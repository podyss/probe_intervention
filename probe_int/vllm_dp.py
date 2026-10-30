import json
import multiprocessing
import os
import shutil
import socket
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def configure_runtime():
    needs_triton_fallback = shutil.which("nvcc") is None
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ and needs_triton_fallback:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return needs_triton_fallback


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _score_rows(scores):
    if scores is None:
        return None
    return [
        None if row is None else {
            str(token_id): None if score is None else score.logprob
            for token_id, score in row.items()
        }
        for row in scores
    ]


class _SharedProgress:
    def __init__(self, rank, progress_counts, max_count):
        self.rank = rank
        self.progress_counts = progress_counts
        self.max_count = max_count
        self.started = time.monotonic()
        self.postfix = ""

    @property
    def n(self):
        return self.progress_counts[self.rank]

    @property
    def format_dict(self):
        return {"elapsed": max(time.monotonic() - self.started, 1e-6)}

    def update(self, count=1):
        with self.progress_counts.get_lock():
            current = self.progress_counts[self.rank]
            increment = min(count, max(0, self.max_count - current))
            self.progress_counts[self.rank] = current + increment

    def refresh(self):
        pass

    def close(self):
        pass


def _rank_progress_factory(rank, progress_counts, max_count):
    def rank_progress(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        return _SharedProgress(rank, progress_counts, max_count)

    return rank_progress


def _worker(rank, dp_size, master_port, prompts, sampling_params,
            engine_options, output_path, error_path,
            progress_counts, use_tqdm):
    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(dp_size),
        "LOCAL_WORLD_SIZE": str(dp_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
    })
    try:
        needs_triton_fallback = configure_runtime()
        from vllm import LLM, SamplingParams

        options = dict(engine_options, seed=engine_options["seed"] + rank)
        if needs_triton_fallback:
            options.setdefault("gdn_prefill_backend", "triton")
        llm = LLM(
            **options,
            data_parallel_size=dp_size,
            distributed_executor_backend="external_launcher",
        )
        requests = [item[1] for item in prompts]
        request_indices = [item[0] for item in prompts]
        params = SamplingParams(**sampling_params)
        rank_progress = (
            _rank_progress_factory(
                rank, progress_counts,
                sum(index is not None for index, _ in prompts)
                * sampling_params.get("n", 1),
            ) if use_tqdm else False
        )

        outputs = llm.generate(
            requests, params, use_tqdm=rank_progress,
        )
        with Path(output_path).open("w", encoding="utf-8") as stream:
            for index, output in zip(request_indices, outputs):
                samples = []
                for sample in output.outputs:
                    samples.append({
                        "text": sample.text,
                        "token_ids": list(sample.token_ids),
                        "finish_reason": sample.finish_reason,
                        "logprobs": _score_rows(sample.logprobs),
                    })
                stream.write(json.dumps({
                    "index": index,
                    "samples": samples,
                    "prompt_logprobs": _score_rows(output.prompt_logprobs),
                }) + "\n")
        # All ranks have left vLLM's global unfinished-request loop. Synchronize
        # before teardown, then explicitly release the native distributed groups.
        import torch.distributed as dist
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory, get_dp_group

        dist.barrier(group=get_dp_group().cpu_group)
        del llm
        cleanup_dist_env_and_memory()
    except BaseException:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _restore_scores(rows):
    if rows is None:
        return None
    return [
        None if row is None else {
            int(token_id): None if score is None else SimpleNamespace(logprob=score)
            for token_id, score in row.items()
        }
        for row in rows
    ]


def partition_prompts(prompts, data_parallel_size):
    if data_parallel_size < 2:
        raise ValueError("data_parallel_size must be at least 2")
    shards = [[] for _ in range(data_parallel_size)]
    for index, prompt in enumerate(prompts):
        shards[index % data_parallel_size].append((index, prompt))
    for shard in shards:
        if not shard:
            shard.append((None, prompts[0]))
    return shards


def generate_with_dp(prompts, sampling_params, engine_options, data_parallel_size,
                     use_tqdm=True):
    """Run one external-launcher vLLM rank per local DP replica."""
    if not prompts:
        return []
    # external_launcher performs its global unfinished check once per frontend
    # step, on the same Gloo group as model-forward synchronization. Async batch
    # queues can drain without a forward on one rank while a peer executes one,
    # interleaving incompatible collectives (completion int32 vs padding tensor).
    # A synchronous scheduler makes each step execute exactly one forward; native
    # vLLM dummy batches keep locally idle ranks participating until global idle.
    # This still uses vLLM's continuous batching, CUDA graphs and native DP.
    engine_options = dict(engine_options, async_scheduling=False)
    shards = partition_prompts(prompts, data_parallel_size)

    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="probe-int-vllm-dp-") as temp_dir:
        processes = []
        paths = []
        progress_counts = context.Array("q", data_parallel_size, lock=True)
        master_port = _free_port()
        for rank, shard in enumerate(shards):
            output_path = Path(temp_dir) / f"rank-{rank}.jsonl"
            error_path = Path(temp_dir) / f"rank-{rank}.error"
            process = context.Process(
                target=_worker,
                args=(rank, data_parallel_size, master_port, shard,
                      sampling_params, engine_options,
                      str(output_path), str(error_path), progress_counts,
                      use_tqdm),
            )
            process.start()
            processes.append(process)
            paths.append((output_path, error_path))

        failed = False
        progress_bar = None
        if use_tqdm:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=len(prompts) * sampling_params.get("n", 1),
                desc="Generating", unit="completion",
                bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )
        last_progress = 0
        total_completions = len(prompts) * sampling_params.get("n", 1)
        try:
            while any(process.is_alive() for process in processes):
                for process in processes:
                    process.join(timeout=0.25)
                if progress_bar is not None:
                    current_progress = min(sum(progress_counts[:]), total_completions - 1)
                    if current_progress > last_progress:
                        progress_bar.update(current_progress - last_progress)
                    last_progress = current_progress
                if any(process.exitcode not in (None, 0) for process in processes):
                    failed = True
                    break
            failed = failed or any(process.exitcode != 0 for process in processes)
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            if progress_bar is not None:
                progress_bar.close()
            raise

        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            if progress_bar is not None:
                current_progress = min(sum(progress_counts[:]), total_completions - 1)
                progress_bar.update(current_progress - last_progress)
                progress_bar.close()
            details = []
            for rank, (process, (_, error_path)) in enumerate(zip(processes, paths)):
                if error_path.exists():
                    details.append(f"DP rank {rank} (exit {process.exitcode}):\n"
                                   + error_path.read_text(encoding="utf-8"))
                elif process.exitcode != 0:
                    details.append(f"DP rank {rank} exited with code {process.exitcode}; "
                                   "no Python traceback (see native worker logs above)")
            raise RuntimeError("vLLM data-parallel worker failed\n" + "\n".join(details))

        ordered = [None] * len(prompts)
        for output_path, _ in paths:
            with output_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["index"] is None:
                        continue
                    samples = [SimpleNamespace(
                        text=sample["text"],
                        token_ids=sample["token_ids"],
                        finish_reason=sample["finish_reason"],
                        logprobs=_restore_scores(sample["logprobs"]),
                    ) for sample in row["samples"]]
                    ordered[row["index"]] = SimpleNamespace(
                        outputs=samples,
                        prompt_logprobs=_restore_scores(row["prompt_logprobs"]),
                    )
        if any(output is None for output in ordered):
            if progress_bar is not None:
                progress_bar.close()
            raise RuntimeError("vLLM data-parallel workers returned incomplete outputs")
        if progress_bar is not None:
            progress_bar.update(total_completions - last_progress)
            progress_bar.close()
        return ordered

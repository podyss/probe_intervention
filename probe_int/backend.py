import math
import os
from numbers import Integral

from .core import tokenizer_fingerprint
from .prompts import PROBE_SUFFIX


QWEN25_PROBE_SUFFIX = "\n\nGive only the final answer in \\boxed{}: "


def token_ids(encoded):
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one chat prompt, got a batched tokenization result")
        encoded = encoded[0]
    if any(not isinstance(token_id, Integral) or isinstance(token_id, bool)
           for token_id in encoded):
        raise TypeError("Chat template must return integer token IDs")
    return [int(token_id) for token_id in encoded]


class VLLMBackend:
    def __init__(self, args):
        if args.tensor_parallel_size > 1:
            # Avoid CUDA symmetric-memory rendezvous failures on some drivers.
            os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
        from .vllm_dp import configure_runtime

        needs_triton_fallback = configure_runtime()
        from vllm import LLM, SamplingParams

        self.args = args
        self.sampling_params = SamplingParams
        self.max_model_len = args.max_model_len
        self.data_parallel_size = getattr(args, "data_parallel_size", 1)
        if self.data_parallel_size < 1:
            raise ValueError("--data-parallel-size must be positive")
        self.engine_options = dict(
            model=args.model, tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=not getattr(args, "disable_prefix_caching", False),
            seed=args.seed, trust_remote_code=False, logprobs_mode="processed_logprobs",
        )
        if needs_triton_fallback:
            self.engine_options["gdn_prefill_backend"] = "triton"
        if self.data_parallel_size > 1:
            from transformers import AutoTokenizer

            self.llm = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=False,
            )
        else:
            self.llm = LLM(**self.engine_options)
            self.tokenizer = self.llm.get_tokenizer()
        self.fingerprint = tokenizer_fingerprint(self.tokenizer)
        self.yes_token_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        self.no_token_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        template_probe = token_ids(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ))
        self.has_thinking_tokens = "<think>\n" in self.tokenizer.decode(
            template_probe, skip_special_tokens=False,
        )
        self.probe_suffix = QWEN25_PROBE_SUFFIX

    def _generate_requests(self, prompts, params, params_kwargs, use_tqdm):
        if getattr(self, "data_parallel_size", 1) > 1:
            from .vllm_dp import generate_with_dp

            return generate_with_dp(
                prompts, params_kwargs, self.engine_options,
                self.data_parallel_size, use_tqdm,
            )
        return self.llm.generate(prompts, params, use_tqdm=use_tqdm)

    def chat(self, content, thinking):
        prompt = token_ids(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking,
        ))
        text = self.tokenizer.decode(prompt, skip_special_tokens=False)
        think_open = "<think>\n"
        if thinking:
            if text.endswith(think_open):
                self.probe_suffix = PROBE_SUFFIX
            elif all(marker in self.tokenizer.all_special_tokens
                     for marker in ["<think>", "</think>"]):
                prompt += self.tokenizer.encode(think_open, add_special_tokens=False)
                self.probe_suffix = PROBE_SUFFIX
        elif text.endswith(think_open):
            open_think_ids = self.tokenizer.encode(think_open, add_special_tokens=False)
            if prompt[-len(open_think_ids):] != open_think_ids:
                raise ValueError("Could not remove the template's open thinking marker")
            prompt = prompt[:-len(open_think_ids)]
        return prompt

    def generate(self, prompts, max_tokens, temperature=0, samples=1, stop=None, top_p=1):
        if not prompts:
            return []
        oversized = [(len(prompt), max_tokens) for prompt in prompts
                     if len(prompt) + max_tokens > self.max_model_len]
        if oversized:
            prompt_tokens, output_tokens = max(oversized, key=lambda request: sum(request))
            raise ValueError(
                f"Request exceeds --max-model-len={self.max_model_len}: "
                f"prompt={prompt_tokens} tokens, requested_output={output_tokens} tokens"
            )
        params_kwargs = dict(
            n=samples, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, stop=stop,
        )
        params = self.sampling_params(**params_kwargs)
        outputs = self._generate_requests(
            [{"prompt_token_ids": prompt} for prompt in prompts], params,
            params_kwargs, use_tqdm=not getattr(self.args, "no_progress", False),
        )
        think_prefix = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        formatted = []
        for prompt, output in zip(prompts, outputs):
            starts_thinking = prompt[-len(think_prefix):] == think_prefix
            samples_out = []
            for sample in output.outputs:
                text = sample.text
                if starts_thinking and not text.startswith("<think>"):
                    text = "<think>\n" + text
                samples_out.append({"text": text, "token_ids": list(sample.token_ids),
                                   "finish_reason": sample.finish_reason})
            formatted.append(samples_out)
        return formatted

    def score_yes_no(self, prompts):
        if not prompts:
            return []
        if len(self.yes_token_ids) != 1 or len(self.no_token_ids) != 1:
            raise ValueError(
                "Judge probes require ' yes' and ' no' to each be a single token "
                "for this tokenizer"
            )
        yes_id, no_id = self.yes_token_ids[0], self.no_token_ids[0]
        if yes_id == no_id:
            raise ValueError("Tokenizer maps ' yes' and ' no' to the same token")
        oversized = [len(prompt) for prompt in prompts if len(prompt) + 1 > self.max_model_len]
        if oversized:
            raise ValueError(
                f"Judge probe exceeds --max-model-len={self.max_model_len}: "
                f"prompt={max(oversized)} tokens, requested_output=1 token"
            )
        request_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]

        params_kwargs = dict(
            temperature=1.0, max_tokens=1, logprobs=2,
            allowed_token_ids=[yes_id, no_id],
        )
        params = self.sampling_params(**params_kwargs)
        outputs = self._generate_requests(
            request_prompts, params, params_kwargs,
            use_tqdm=not getattr(self.args, "no_progress", False),
        )
        def has_finite_logprob(score):
            return score is not None and math.isfinite(score.logprob)

        missing_indices = []
        candidate_scores = []
        for index, output in enumerate(outputs):
            sample = output.outputs[0]
            token_scores = sample.logprobs[0] if sample.logprobs else {}
            yes_score = token_scores.get(yes_id)
            no_score = token_scores.get(no_id)
            candidate_scores.append((yes_score, no_score))
            if not has_finite_logprob(yes_score) or not has_finite_logprob(no_score):
                missing_indices.append(index)

        # vLLM's top-k response can omit a constrained candidate in edge cases.
        # Score only those candidates as prompt tokens, without generation filters.
        if missing_indices:
            fallback_prompts = []
            fallback_owners = []
            for index in missing_indices:
                for token_id in (yes_id, no_id):
                    fallback_prompts.append(prompts[index] + [token_id])
                    fallback_owners.append((index, token_id))
            if any(len(prompt) + 1 > self.max_model_len for prompt in fallback_prompts):
                raise RuntimeError(
                    "Judge-token logprobs were omitted and prompt-logprob fallback "
                    "would exceed --max-model-len"
                )
            fallback_params_kwargs = dict(
                temperature=0, max_tokens=1, prompt_logprobs=1,
            )
            fallback_params = self.sampling_params(**fallback_params_kwargs)
            fallback_outputs = self._generate_requests(
                [{"prompt_token_ids": prompt} for prompt in fallback_prompts],
                fallback_params, fallback_params_kwargs,
                use_tqdm=not getattr(self.args, "no_progress", False),
            )
            fallback_by_index = {index: {} for index in missing_indices}
            for (index, token_id), output in zip(fallback_owners, fallback_outputs):
                prompt_logprobs = output.prompt_logprobs
                token_scores = prompt_logprobs[-1] if prompt_logprobs else None
                score = token_scores.get(token_id) if token_scores else None
                if has_finite_logprob(score):
                    fallback_by_index[index][token_id] = score
            for index in missing_indices:
                yes_score, no_score = candidate_scores[index]
                if not has_finite_logprob(yes_score):
                    yes_score = fallback_by_index[index].get(yes_id)
                if not has_finite_logprob(no_score):
                    no_score = fallback_by_index[index].get(no_id)
                candidate_scores[index] = (yes_score, no_score)

        scored = []
        for index, (yes_score, no_score) in enumerate(candidate_scores):
            if not has_finite_logprob(yes_score) or not has_finite_logprob(no_score):
                missing = [label for label, score in (("yes", yes_score), ("no", no_score))
                           if not has_finite_logprob(score)]
                raise RuntimeError(
                    f"vLLM omitted judge token logprobs for {', '.join(missing)}; "
                    f"prompt-logprob fallback also failed for request {index}"
                )
            yes_logprob, no_logprob = yes_score.logprob, no_score.logprob
            normalizer = max(yes_logprob, no_logprob)
            yes_weight = math.exp(yes_logprob - normalizer)
            no_weight = math.exp(no_logprob - normalizer)
            answer_id = yes_id if yes_logprob >= no_logprob else no_id
            token_ids = [answer_id]
            scored.append({
                "text": self.tokenizer.decode(token_ids, skip_special_tokens=False).strip(),
                "token_ids": token_ids,
                "finish_reason": "scored",
                "answer": "yes" if answer_id == yes_id else "no",
                "correct": None,
                "status": "scored",
                "p_correct": yes_weight / (yes_weight + no_weight),
                "yes_logprob": yes_logprob,
                "no_logprob": no_logprob,
            })
        return scored

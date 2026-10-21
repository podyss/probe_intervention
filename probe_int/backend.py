from .core import tokenizer_fingerprint
from .prompts import PROBE_SUFFIX


QWEN25_PROBE_SUFFIX = "\n\nGive only the final answer in \\boxed{}: "


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.args = args
        self.sampling_params = SamplingParams
        self.max_model_len = args.max_model_len
        self.llm = LLM(
            model=args.model, tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=not getattr(args, "disable_prefix_caching", False),
            seed=args.seed, trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.fingerprint = tokenizer_fingerprint(self.tokenizer)
        self.probe_suffix = QWEN25_PROBE_SUFFIX

    def chat(self, content, thinking):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        if thinking:
            text = self.tokenizer.decode(prompt, skip_special_tokens=False)
            assistant_header = "<|im_start|>assistant\n"
            if assistant_header not in text:
                raise ValueError("Backend requires a Qwen-style assistant header")
            assistant_prefix = text.rsplit(assistant_header, 1)[1]
            if "</think>" in assistant_prefix:
                raise ValueError("Thinking was closed by the chat template")
            if "<think>" in assistant_prefix:
                self.probe_suffix = PROBE_SUFFIX
            else:
                if all(marker in self.tokenizer.all_special_tokens
                       for marker in ["<think>", "</think>"]):
                    prompt += self.tokenizer.encode("<think>\n", add_special_tokens=False)
                    self.probe_suffix = PROBE_SUFFIX
        return prompt

    def generate(self, prompts, max_tokens, temperature=0, samples=1, stop=None):
        if not prompts:
            return []
        if any(len(prompt) + max_tokens > self.max_model_len for prompt in prompts):
            raise ValueError("Request exceeds --max-model-len; shorten input or adjust token budgets")
        params = self.sampling_params(
            n=samples, temperature=temperature, max_tokens=max_tokens, stop=stop,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts], params,
            use_tqdm=not getattr(self.args, "no_progress", True),
        )
        return [[{"text": sample.text, "token_ids": list(sample.token_ids),
                  "finish_reason": sample.finish_reason} for sample in output.outputs]
                for output in outputs]

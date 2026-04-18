from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os


class VLLMModel:
    def __init__(self, model_path: str, token=None, tensor_parallel_size=1, gpu_memory_utilization=0.45, max_model_len=4096):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        print(f"vLLM model loaded: {model_path}")

    def generate(self, system: str, user: str, max_length: int = 2000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
        plain_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
        )

        outputs = self.model.generate([plain_text], sampling_params)
        return outputs[0].outputs[0].text

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 2000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
        plain_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
        )

        outputs = self.model.generate([plain_text], sampling_params)
        return outputs[0].outputs[0].text

    def continue_generate(self, condition: str, system: str, user1: str, assistant1: str, user2: str, max_length: int = 2000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user1},
            {'role': 'assistant', 'content': assistant1},
            {'role': 'user', 'content': user2},
        ]
        plain_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
        )

        outputs = self.model.generate([plain_text], sampling_params)
        return outputs[0].outputs[0].text

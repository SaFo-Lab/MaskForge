"""Client for vLLM OpenAI-compatible server."""
import requests
import json


class VLLMServerModel:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        print(f"VLLMServerModel connected: {base_url} ({model_name})")

    def generate(self, system: str, user: str, max_length: int = 2000, **kwargs):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._chat(messages, max_length, **kwargs)

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 2000, **kwargs):
        if condition:
            if system.strip() or user.strip():
                # Has system/user context — use chat template + condition via completions API
                from transformers import AutoTokenizer
                if not hasattr(self, '_tokenizer'):
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                prompt += condition
            else:
                # No system/user — use condition directly as prompt (for base models)
                prompt = condition
            return self._complete(prompt, max_length, **kwargs)
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self._chat(messages, max_length, **kwargs)

    def continue_generate(self, condition: str, system: str, user1: str, assistant1: str, user2: str, max_length: int = 2000, **kwargs):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
        ]
        if condition:
            from transformers import AutoTokenizer
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt += condition
            return self._complete(prompt, max_length, **kwargs)
        return self._chat(messages, max_length, **kwargs)

    def _complete(self, prompt, max_length, **kwargs):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": min(max_length, 1500),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["text"]
        except requests.exceptions.HTTPError as e:
            print(f"[VLLMServerModel] Completions API error: {e}")
            print(f"[VLLMServerModel] Response: {resp.text[:500]}")
            return ""

    def _chat(self, messages, max_length, **kwargs):
        # Filter out empty messages and ensure valid format
        clean_messages = []
        for m in messages:
            if m.get("content", "").strip():
                clean_messages.append({"role": m["role"], "content": m["content"]})
        if not clean_messages:
            clean_messages = [{"role": "user", "content": "Generate a response."}]

        payload = {
            "model": self.model_name,
            "messages": clean_messages,
            "max_tokens": min(max_length, 1500),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print(f"[VLLMServerModel] Chat API error: {e}")
            print(f"[VLLMServerModel] Response: {resp.text[:500]}")
            return ""

import os
import re
import json
import mimetypes
import base64
from typing import Optional, List, Dict, Any

import requests as http_requests

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockModel:
    def __init__(
        self,
        model_id: str,
        region_name: Optional[str] = "us-east-2",
        bearer_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize a Bedrock model wrapper.

        Auth priority:
            1. bearer_token  → direct HTTP with Authorization: Bearer header
            2. aws credentials → boto3 client with SigV4

        Args:
            model_id: Bedrock model ID
            region_name: AWS region (default us-east-1)
            bearer_token: Bedrock API key (Bearer token), used via HTTP directly
            aws_access_key_id: AWS access key (boto3 path)
            aws_secret_access_key: AWS secret key (boto3 path)
            aws_session_token: AWS session token (boto3 path)
        """
        self.model_id = model_id
        self.region_name = region_name or "us-east-1"
        self.bearer_token = bearer_token
        self.client = None

        if self.bearer_token:
            self.base_url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com"
            print(f"Bedrock model initialized (Bearer token): {self.model_id} @ {self.region_name}")
        else:
            if boto3 is None:
                raise ImportError("boto3 is required when not using bearer_token")
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
            print(f"Bedrock model initialized (boto3): {self.model_id} @ {self.region_name}")

    # ------------------------------------------------------------------ #
    #  Internal: two paths for calling converse API
    # ------------------------------------------------------------------ #

    def _converse(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route to HTTP or boto3 based on auth mode."""
        if self.bearer_token:
            return self._converse_http(messages, system, inference_config)
        else:
            return self._converse_boto3(messages, system, inference_config)

    def _converse_http(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/model/{self.model_id}/converse"
        payload: Dict[str, Any] = {"messages": messages}
        if system:
            payload["system"] = [{"text": system}]
        if inference_config:
            payload["inferenceConfig"] = inference_config

        resp = http_requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Bedrock HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    def _converse_boto3(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if inference_config:
            kwargs["inferenceConfig"] = inference_config
        return self.client.converse(**kwargs)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        content = response["output"]["message"]["content"]
        texts = [block["text"] for block in content if "text" in block]
        return "\n".join(texts).strip()

    @staticmethod
    def _build_inference_config(
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
        for k, v in kwargs.items():
            if v is not None:
                config[k] = v
        return config

    # ------------------------------------------------------------------ #
    #  Public API (unchanged signatures)
    # ------------------------------------------------------------------ #

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        messages = [{"role": "user", "content": [{"text": user}]}]
        config = self._build_inference_config(max_tokens, temperature, top_p, **kwargs)
        response = self._converse(messages, system=system or None, inference_config=config)
        return self._extract_text(response)

    def continue_generate(
        self,
        condition: str,
        system: str,
        user1: str,
        assistant1: str,
        user2: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        final_user = user2 + condition if condition else user2
        messages = [
            {"role": "user", "content": [{"text": user1}]},
            {"role": "assistant", "content": [{"text": assistant1}]},
            {"role": "user", "content": [{"text": final_user}]},
        ]
        config = self._build_inference_config(max_tokens, temperature, top_p, **kwargs)
        response = self._converse(messages, system=system or None, inference_config=config)
        return self._extract_text(response)

    def conditional_generate(
        self,
        condition: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        full_user = user + condition if condition else user
        messages = [{"role": "user", "content": [{"text": full_user}]}]
        config = self._build_inference_config(max_tokens, temperature, top_p, **kwargs)
        response = self._converse(messages, system=system or None, inference_config=config)
        return self._extract_text(response)

    def generate_with_image(
        self,
        image_path: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        media_type, _ = mimetypes.guess_type(image_path)
        if not media_type or not media_type.startswith("image/"):
            raise ValueError(f"Unsupported image type: {media_type}")

        image_format = media_type.split("/")[-1].lower()
        if image_format == "jpeg":
            image_format = "jpg"

        # For HTTP path, image bytes must be base64 encoded in JSON
        if self.bearer_token:
            image_content = {
                "image": {
                    "format": image_format,
                    "source": {"bytes": base64.b64encode(image_bytes).decode("utf-8")}
                }
            }
        else:
            image_content = {
                "image": {
                    "format": image_format,
                    "source": {"bytes": image_bytes}
                }
            }

        messages = [
            {
                "role": "user",
                "content": [image_content, {"text": prompt}]
            }
        ]
        config = self._build_inference_config(max_tokens, temperature, top_p, **kwargs)
        response = self._converse(messages, system=system, inference_config=config)
        return self._extract_text(response)


if __name__ == "__main__":
    # ---- 方式1: Bearer Token (不污染环境变量) ----
    model = BedrockModel(
        model_id="qwen.qwen3-235b-a22b-2507-v1:0",
        region_name="us-east-2",
    )

    output = model.generate(
        system="You are helpful assistant",
        user="Hello"
    )
    print(output)

    # ---- 方式2: 传统 IAM 凭证 (boto3) ----
    # model = BedrockModel(
    #     model_id="qwen.qwen3-235b-a22b-2507-v1:0",
    #     region_name="us-east-2",
    #     aws_access_key_id="YOUR_ACCESS_KEY",
    #     aws_secret_access_key="YOUR_SECRET_KEY",
    # )
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_torch_dtype(dtype_name: str):
    normalized = (dtype_name or "auto").lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported llm_torch_dtype: {dtype_name}")


@dataclass
class LocalModelConfig:
    model_id: str
    max_new_tokens: int = 2048
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False


class LocalModelClient:
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=_resolve_torch_dtype(config.torch_dtype),
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )

    def generate(self, system_prompt: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            rendered_prompt = self._render_prompt(messages)
            inputs = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
            )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @staticmethod
    def _render_prompt(messages) -> str:
        prompt_parts = []
        for message in messages:
            role = message["role"].upper()
            content = message["content"]
            prompt_parts.append(f"{role}:\n{content}")
        prompt_parts.append("ASSISTANT:\n")
        return "\n\n".join(prompt_parts)

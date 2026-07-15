from .model_client import LocalModelClient, LocalModelConfig
from .proposer import Proposer


class LLMProposer(Proposer):
    def __init__(
        self,
        target_val,
        target_prompt,
        model_id,
        max_new_tokens=2048,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False,
    ):
        super().__init__(target_val, target_prompt)
        self.client = LocalModelClient(
            LocalModelConfig(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        )

    def generate(self, system_prompt, prompt):
        return self.client.generate(system_prompt, prompt)

from .llm_proposer import LLMProposer


def load_proposer(args, target_prompt, max_new_tokens=2048):
    llm_model = getattr(args, "llm_model", None)
    if not llm_model:
        raise ValueError(
            "No local LLM model path or Hugging Face model id provided. "
            "Set --llm_model (or --llm_model_path)."
        )

    return LLMProposer(
        target_val=args.target_value,
        target_prompt=target_prompt,
        model_id=llm_model,
        max_new_tokens=max_new_tokens,
        device_map=getattr(args, "llm_device_map", "auto"),
        torch_dtype=getattr(args, "llm_torch_dtype", "auto"),
        trust_remote_code=getattr(args, "llm_trust_remote_code", False),
    )

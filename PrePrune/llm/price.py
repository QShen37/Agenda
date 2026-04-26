import tiktoken

from PrePrune.utils.globals import Cost, PromptTokens, CompletionTokens


def cal_token(model: str, text: str) -> int:
    try:
        if not text:
            return 0

        if model.startswith(("gpt-5", "gpt-4.1", "gpt-4o")):
            encoder = tiktoken.get_encoding("cl100k_base")

        elif model.startswith("gpt-"):
            encoder = tiktoken.encoding_for_model(model)

        else:
            return 0

        return len(encoder.encode(text))

    except Exception as e:
        print(
            f"[cal_token] Warning: Failed to calculate token count ({model}), defaulting to 0. Error: {e}"
        )
        return 0

def cal_qwen_token(text):
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)

    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"[cal_qwen_token] Failed, returning 0: {e}")
        return 0

def cal_deepseek_token(text):
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)

    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"[cal_deepseek_token] Failed, returning 0: {e}")
        return 0

def cal_kimi_token(text):
    """
    Kimi (Moonshot) token approximate estimation (cl100k_base)
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)

    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"[cal_kimi_token] Failed, returning 0: {e}")
        return 0


def get_token_count(prompt, response, model_name):
    """
    Unified token counting interface for papers/experiments
    """

    # GLM: must use real usage
    if "glm" in model_name.lower():
        if hasattr(response, "usage"):
            return (
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        else:
            return 0, 0

    # OpenAI: use tiktoken estimation
    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)
    return prompt_len, completion_len

def cost_count(prompt, response, model_name):
    """
    prompt: str
    response:
        - GPT / Qwen: str
        - GLM: OpenAI-style response object
    """

    model_name_lower = model_name.lower()

    if "glm" in model_name_lower:

        prompt_len = 0
        completion_len = 0

        if hasattr(response, "usage"):
            prompt_len = getattr(response.usage, "prompt_tokens", 0)
            completion_len = getattr(response.usage, "completion_tokens", 0)

        GLM_INPUT_PRICE = 0.000002
        GLM_OUTPUT_PRICE = 0.000008

        price = (
            prompt_len * GLM_INPUT_PRICE
            + completion_len * GLM_OUTPUT_PRICE
        )

        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len

        return price, prompt_len, completion_len

    if "qwen" in model_name_lower:

        if hasattr(response, "usage") and response.usage is not None:
            prompt_len = getattr(response.usage, "prompt_tokens", 0)
            completion_len = getattr(response.usage, "completion_tokens", 0)
        else:
            prompt_len = cal_qwen_token(prompt)
            text = response if isinstance(response, str) else str(response)
            completion_len = cal_qwen_token(text)

        model_key = model_name_lower
        if model_key not in QWEN_MODEL_INFO:
            print(f"[Qwen] Unknown model {model_name}, cost set to 0")
            price = 0.0
        else:
            price = (
                    prompt_len * QWEN_MODEL_INFO[model_key]["input"] / 1000
                    + completion_len * QWEN_MODEL_INFO[model_key]["output"] / 1000
            )

        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len

        return price, prompt_len, completion_len

    if "gemini" in model_name_lower:

        # Use official usage if available
        if hasattr(response, "usage") and response.usage is not None:
            prompt_len = getattr(response.usage, "prompt_tokens", 0)
            completion_len = getattr(response.usage, "completion_tokens", 0)
        else:
            # fallback: approximate estimation
            prompt_len = cal_qwen_token(prompt)
            text = response if isinstance(response, str) else str(response)
            completion_len = cal_qwen_token(text)

        model_key = model_name_lower
        if model_key not in GEMINI_MODEL_INFO:
            print(f"[Gemini] Unknown model {model_name}, cost set to 0")
            price = 0.0
        else:
            price = (
                    prompt_len * GEMINI_MODEL_INFO[model_key]["input"] / 1000
                    + completion_len * GEMINI_MODEL_INFO[model_key]["output"] / 1000
            )

        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len

        return price, prompt_len, completion_len

    elif "deepseek" in model_name_lower:

        # Official usage preferred
        if hasattr(response, "usage") and response.usage is not None:
            prompt_len = getattr(response.usage, "prompt_tokens", 0)
            completion_len = getattr(response.usage, "completion_tokens", 0)
        else:
            # fallback: approximate estimation
            prompt_len = cal_deepseek_token(prompt)
            text = response if isinstance(response, str) else str(response)
            completion_len = cal_deepseek_token(text)

        model_key = model_name_lower
        if model_key not in DEEPSEEK_MODEL_INFO:
            print(f"[DeepSeek] Unknown model {model_name}, cost set to 0")
            price = 0.0
        else:
            price = (
                    prompt_len * DEEPSEEK_MODEL_INFO[model_key]["input"] / 1000
                    + completion_len * DEEPSEEK_MODEL_INFO[model_key]["output"] / 1000
            )

        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len

        return price, prompt_len, completion_len

    if "kimi" in model_name_lower:
        if hasattr(response, "usage") and response.usage is not None:
            prompt_len = getattr(response.usage, "prompt_tokens", 0)
            completion_len = getattr(response.usage, "completion_tokens", 0)
        else:
            prompt_len = cal_kimi_token(prompt)
            text = response if isinstance(response, str) else str(response)
            completion_len = cal_kimi_token(text)

        model_key = model_name_lower
        if model_key not in KIMI_MODEL_INFO:
            print(f"[Kimi] Unknown model {model_name}, cost set to 0")
            price = 0.0
        else:
            price = (
                prompt_len * KIMI_MODEL_INFO[model_key]["input"] / 1000
                + completion_len * KIMI_MODEL_INFO[model_key]["output"] / 1000
            )

        Cost.instance().value += price
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len

        return price, prompt_len, completion_len

    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)

    price = 0.0

    if "gpt-4" in model_name:
        price = (
            prompt_len * OPENAI_MODEL_INFO["gpt-4"][model_name]["input"] / 1000
            + completion_len * OPENAI_MODEL_INFO["gpt-4"][model_name]["output"] / 1000
        )

    elif "gpt-3.5" in model_name:
        price = (
            prompt_len * OPENAI_MODEL_INFO["gpt-3.5"][model_name]["input"] / 1000
            + completion_len * OPENAI_MODEL_INFO["gpt-3.5"][model_name]["output"] / 1000
        )
    elif "gpt-5" in model_name:
        price = (
            prompt_len * OPENAI_MODEL_INFO["gpt-5"][model_name]["input"] / 1000
            + completion_len * OPENAI_MODEL_INFO["gpt-5"][model_name]["output"] / 1000
        )

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    return price, prompt_len, completion_len


OPENAI_MODEL_INFO = {
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-1106": {"input": 0.0010, "output": 0.0020},
    },
    "gpt-5": {
        "current_recommended": "gpt-5-mini",
        "gpt-5-mini": {"input": 0.00025, "output": 0.0020},
    }
}

QWEN_MODEL_INFO = {
    "qwen-plus": {
        "input": 0.0008,   # CNY / 1K tokens
        "output": 0.0020,
    },
    "qwen-turbo": {
        "input": 0.0003,
        "output": 0.0006,
    },
    "qwen-max": {
        "input": 0.0024,
        "output": 0.0072,
    }
}

GEMINI_MODEL_INFO = {
    "gemini-2.5-flash": {
        "input": 0.00035,   # CNY / 1K tokens (example)
        "output": 0.00105,
    },
    "gemini-1.5-pro": {
        "input": 0.0025,
        "output": 0.0075,
    }
}

DEEPSEEK_MODEL_INFO = {
    "deepseek-chat": {
        "input": 0.002,    # example price
        "output": 0.003,
    },
    "deepseek-reasoner": {
        "input": 0.002,
        "output": 0.003,
    }
}

KIMI_MODEL_INFO = {
    "kimi-k2-0905-preview": {
        "input": 0.0040,    # CNY / 1K tokens (example)
        "output": 0.0016,
    },
    "kimi-k2-turbo-preview": {
        "input": 0.0080,
        "output": 0.0058,
    }
}

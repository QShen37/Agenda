import os
from typing import Dict
from typing import List, Union, Optional

from httpx import Timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from PrePrune.llm.format import Message
from PrePrune.llm.llm import LLM
from PrePrune.llm.llm_registry import LLMRegistry
from PrePrune.llm.price import cost_count
from dotenv import load_dotenv

load_dotenv()

# Read provider credentials from .env
MINE_API_KEYS = os.getenv("MINE_API_KEYS", os.getenv("MINE_API_KEY", ""))
MINE_BASE_URL = os.getenv("MINE_BASE_URL", "")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")

KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def achat(
        model: str = "glm-4.6v",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = AsyncOpenAI(
            api_key=MINE_API_KEYS,
            base_url=MINE_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([item.get("content", "") for item in msg])
        response_text = completion.choices[0].message.content
        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[achat] Error: {str(e)}")
        return None

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def chat(
        model: str = "glm-4.6v",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = OpenAI(
            api_key=MINE_API_KEYS,
            base_url=MINE_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([item.get("content", "") for item in msg])
        response_text = completion.choices[0].message.content
        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[chat] Error: {str(e)}")
        return None

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def qwen_achat(
        model: str = "qwen-plus",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    try:
        client = AsyncOpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)
        return response_text

    except Exception as e:
        print(f"[qwen_achat] Error: {e}")
        return ""


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def qwen_chat(
        model: str,
        msg: List[Dict],
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    try:
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)
        return response_text

    except Exception as e:
        print(f"[qwen_chat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def gemini_achat(
        model: str = "gemini-2.5-flash",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[gemini_achat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def gemini_chat(
        model: str = "gemini-2.5-flash",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[gemini_chat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def deepseek_achat(
        model: str = "deepseek-chat",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[deepseek_achat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def deepseek_chat(
        model: str = "deepseek-chat",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)

        return response_text

    except Exception as e:
        print(f"[deepseek_chat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def kimi_achat(
        model: str = "kimi-k2-turbo-preview",
        msg: List[Dict] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 2048,
):
    if msg is None:
        msg = []

    try:
        client = AsyncOpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)
        return response_text

    except Exception as e:
        print(f"[kimi_achat] Error: {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def kimi_chat(
        model: str = "kimi-k2-turbo-preview",
        msg: List[Dict] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 2048,
):
    if msg is None:
        msg = []

    try:
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
            timeout=Timeout(100.0)
        )

        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response_text = completion.choices[0].message.content

        cost_count(prompt_text, completion, model)
        return response_text

    except Exception as e:
        print(f"[kimi_chat] Error: {e}")
        return ""

# OpenAI
@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def openai_achat(
        model: str = "gpt-5-mini",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=Timeout(100.0)
        )

        input_text = "\n".join([m.get("content", "") for m in msg])

        response = await client.responses.create(
            model=model,
            input=input_text,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        output_text = response.output_text or ""

        try:
            cost_count(input_text, output_text, model)
        except Exception as e:
            print(f"[openai_achat] cost_count skipped: {e}")

        return output_text

    except Exception as e:
        print(f"[openai_achat] Error (model={model}): {e}")
        return ""

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
def openai_chat(
        model: str = "gpt-5-mini",
        msg: List[Dict] = None,
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_tokens: int = 10000,
):
    if msg is None:
        msg = []

    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=Timeout(100.0)
        )

        prompt_text = "".join([m.get("content", "") for m in msg])
        response = client.responses.create(
            model=model,
            input=prompt_text,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response_text = response.output_text or ""

        try:
            cost_count(prompt_text, response_text, model)
        except Exception as e:
            print(f"[openai_achat] cost_count skipped: {e}")
        return response_text

    except Exception as e:
        print(f"[openai_chat] Error: {e}")
        return ""

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        conv_messages = []
        for m in messages:
            if isinstance(m, dict):
                conv_messages.append(m)
            else:
                conv_messages.append({"role": m.role, "content": m.content})

        if self.model_name.startswith("glm"):
            return await achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

        elif self.model_name.startswith("qwen"):
            return await qwen_achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("gemini"):
            return await gemini_achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("deepseek"):
            return await deepseek_achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("kimi"):
            return await kimi_achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("gpt"):
            return await openai_achat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        conv_messages = []
        for m in messages:
            if isinstance(m, dict):
                conv_messages.append(m)
            else:
                conv_messages.append({"role": m.role, "content": m.content})

        if self.model_name.startswith("glm"):
            res = chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

        elif self.model_name.startswith("qwen"):
            res = qwen_chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("gemini"):
            res = gemini_chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("deepseek"):
            res = deepseek_chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("kimi"):
            res = kimi_chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif self.model_name.startswith("gpt"):
            res = openai_chat(
                model=self.model_name,
                msg=conv_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if res is None:
            return ""

        if isinstance(res, tuple):
            return res[0] if len(res) > 0 else ""

        return res

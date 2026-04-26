import os
from typing import List
import re
import time
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")


class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def update(self, usage: dict, tag: str = ""):
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)

        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total

        print(f"[Token Usage - {tag}]")
        print(f"  Prompt tokens     : {prompt}")
        print(f"  Completion tokens : {completion}")
        print(f"  Total tokens      : {total}")

        print(f"[Cumulative]")
        print(f"  Prompt total     : {self.total_prompt_tokens}")
        print(f"  Completion total : {self.total_completion_tokens}")
        print(f"  Overall total    : {self.total_tokens}")


class LLMScorer:
    def __init__(
            self,
            use_llm: bool = True,
            model: str = "qwen-plus",
            max_retries: int = 3,
            sleep_base: float = 0.6,
            debug: bool = True
    ):
        self.use_llm = use_llm and (openai is not None)
        self.model = model
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.token_tracker = TokenTracker()
        self.debug = debug

    @staticmethod
    def _build_batch_prompt(task: str, agent_descs: List[str]) -> str:
        agents_str = ""
        for i, desc in enumerate(agent_descs):
            agents_str += f"ID {i}: {desc}\n"

        prompt = f"""
        You are a strict evaluator.

        Task:
        \"\"\"{task}\"\"\"

        Agents:
        {agents_str}

        Scoring rules:
        - Score each agent from 0 to 1
        - 1 = highly suitable, 0 = completely irrelevant
        - Be consistent across agents
        - Use at most 2 decimal places
        - The scores should roughly follow a normal distribution

        IMPORTANT:
        - Output ONLY valid JSON
        - Do NOT include any explanation

        Format:
        {{"scores": [0.85, 0.40, 0.92]}}
        """
        return prompt

    def _safe_parse_scores(self, content: str, n: int) -> List[float]:
        try:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise ValueError("No JSON found")

            data = json.loads(match.group(0))
            scores = [float(s) for s in data.get("scores", [])]

        except Exception as e:
            print(f"[Parse Error]: {e}")
            return [0.0] * n

        if len(scores) < n:
            scores.extend([0.0] * (n - len(scores)))

        return scores[:n]

    def get_batch_scores(self, task: str, agent_descs: List[str]) -> List[float]:
        if not agent_descs:
            return []

        prompt = self._build_batch_prompt(task, agent_descs)

        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )

        for attempt in range(self.max_retries):
            try:
                if self.debug:
                    print("\n" + "=" * 20 + " LLM PROMPT " + "=" * 20)
                    print(prompt)

                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0
                )

                content = completion.choices[0].message.content

                if self.debug:
                    print("\n" + "=" * 20 + " LLM RESPONSE " + "=" * 20)
                    print(content)

                if hasattr(completion, "usage") and completion.usage:
                    usage = {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    }
                    self.token_tracker.update(usage, tag=f"Task: {task[:40]}")

                scores = self._safe_parse_scores(content, len(agent_descs))

                if self.debug:
                    print("\nParsed Scores:", scores)
                    print("=" * 60 + "\n")

                if self.debug:
                    log_data = {
                        "task": task,
                        "prompt": prompt,
                        "response": content,
                        "scores": scores
                    }
                    print("[JSON LOG]")
                    print(json.dumps(log_data, ensure_ascii=False, indent=2))

                return scores

            except Exception as e:
                print(f"[Retry {attempt + 1}] Error: {e}")

                if attempt == self.max_retries - 1:
                    return [0.0] * len(agent_descs)

                sleep_time = self.sleep_base * (2 ** attempt)
                time.sleep(sleep_time)

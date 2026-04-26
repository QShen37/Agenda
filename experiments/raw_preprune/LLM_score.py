from typing import List, Optional, Callable, Tuple
import re
import time
import json
import openai
from openai import OpenAI


class LLMScorer:
    """
    LLMScorer wrapper:
      - Calls LLM to evaluate the suitability of a task for each agent_desc (returns 0-1)
      - If LLM is unavailable or fails, uses embedding_proxy_fn as a fallback (embedding -> cosine -> mapped to [0,1])
      - Provides batch interface score_batch, optionally calling LLM only on top_k to save cost
    Constructor parameters:
      - use_llm: whether to attempt using OpenAI (default True)
      - model: LLM model name (default "qwen-plus")
      - max_retries: number of retry attempts for LLM calls
      - sleep_base: base sleep time for retries (seconds)
    """
    def __init__(
        self,
        use_llm: bool = True,
        model: str = "qwen-plus",
        max_retries: int = 3,
        sleep_base: float = 0.6
    ):
        self.use_llm = use_llm and (openai is not None)
        self.model = model
        self.max_retries = max_retries
        self.sleep_base = sleep_base

    @staticmethod
    def _build_prompt(task: str, agent_desc: str) -> str:
        """
        Construct the prompt: require the model to strictly output JSON, including score (0-1) and reason.
        Using temperature=0 increases stability.
        """
        prompt = f"""
                    You are an assistant who judges how suitable an agent is for solving a given task.
                    Return exactly one JSON object (no extra text) with two fields:
                      "score": a number between 0 and 1 (inclusive) indicating suitability,
                      "reason": a short one-sentence explanation.

                    Task: \"\"\"{task}\"\"\" 
                    Agent description: \"\"\"{agent_desc}\"\"\" 

                    Example:
                    {{"score": 0.85, "reason": "Agent's skills closely match the task requirements."}}

                    Now produce the JSON for the given Task and Agent.
                """
        return prompt

    def extract_score_only(self, text: str) -> float:
        """
        Robustly extract the score field from the JSON text returned by LLM (ignore reason).
        Extra text before or after JSON is allowed.
        Returns score ∈ [0,1]; returns 0 on failure.
        """
        if not text or not isinstance(text, str):
            return 0.0

        # 1. Try regex to extract the innermost JSON
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            json_candidate = match.group(0)
            try:
                data = json.loads(json_candidate)
                score = float(data.get("score", 0.0))
                return max(0.0, min(1.0, score))
            except Exception:
                pass

        # 2. If JSON parsing fails, try directly finding the score field
        match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except Exception:
                pass
        return 0.0

    def achat(self, prompt):
        try:
            client = OpenAI(
                api_key = "sk-efe456d1242c417a8666db7915069c21",
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=15
            )
            completion = client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                top_p = 0.7,
                temperature = 0.9
            )

            score = self.extract_score_only(completion.choices[0].message.content)
            return score

        except Exception as e:
            print("[achat ERROR]:", str(e))
            return 0.0

    def _call_openai_with_retry(self, prompt: str) -> Optional[float]:
        """OpenAI call with retry logic"""
        if not self.use_llm:
            return None
        for attempt in range(self.max_retries):
            score = self.achat(prompt)
            if score is not None:
                return score
            time.sleep(self.sleep_base * (attempt + 1))
        return None

    def llm_score(self, task, agent_descs):
        # agent_descs = [a['desc'] for a in agents]
        score = []
        for i in range(len(agent_descs)):
            prompt = self._build_prompt(task, agent_descs[i])
            score.append(self.achat(prompt))
        return score

    def score_batch(
        self,
        task: str,
        agent_descs: List[str],
        top_k: Optional[int] = None,
        llm_only_for_topk: bool = True,
        sleep_per_call: float = 0.02,
        cossim_task2agent: List[float] = None
    ) -> List[float]:
        """
        Return LLM scores (0-1) for a given task against a list of agent_descs.
        Parameters:
          - top_k: if not None, first sort all agents by embedding proxy, then only call LLM on top_k (if use_openai)
          - llm_only_for_topk: True means only call LLM for top_k; others use proxy
          - llm_weight: here it only returns LLM-based scores (0-1), does not do adjusted fusion (fusion done externally)
        Returns:
          - scores: list of floats (len == len(agent_descs)), each ∈ [0,1]
        Note on cost: calling OpenAI for many agents is expensive, recommend pre-filtering top_k externally
        """

        # Get top-k relevant agents
        n = len(agent_descs)
        scores = cossim_task2agent
        sort_scores = list(enumerate(scores))
        sort_scores.sort(reverse=True, key=lambda x: x[1])
        top_indices = [idx for idx, _ in sort_scores[:top_k]]

        # If OpenAI cannot be used or should not be called, use proxy for all (or as much as possible)
        if not self.use_llm:
            return scores

        # If OpenAI is available: call LLM on top_indices; others use proxy if needed
        for i in range(n):
            if top_k is None or (i in top_indices) or (not llm_only_for_topk):
                # Call LLM if needed
                prompt = self._build_prompt(task, agent_descs[i])
                s = self._call_openai_with_retry(prompt)
                if s is None:
                    # LLM call failed -> use 0
                    s = 0
                scores[i] = float(s)
                # Small sleep to avoid rate limits
                time.sleep(sleep_per_call)
        return scores


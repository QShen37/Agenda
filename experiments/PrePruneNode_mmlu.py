import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sentence_transformers import SentenceTransformer
import time
import asyncio
from typing import List, Dict, Tuple

from LLM_score import LLMScorer
from PrePrune.prompt import MMLUPromptSet, HumanEvalPromptSet, GSM8KPromptSet
from MLT.threshold_predictor import ThresholdPredictor
from PrePrune.tools.reader.readers import JSONLReader
from datasets.gsm8k_dataset import gsm_data_process

class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
scorer = LLMScorer(use_llm=True, max_retries=3, sleep_base=0.6)

def normalize(x: np.ndarray):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cosine_similarity(task: str, agent_descs: List[str]) -> np.ndarray:
    task_emb = embedding_model.encode([task])
    agent_emb = embedding_model.encode(agent_descs)

    task_emb = normalize(task_emb)[0]
    agent_emb = normalize(agent_emb)

    sim = agent_emb @ task_emb
    sim = (sim + 1) / 2
    return sim

def select_agents(
        task: str,
        agent_names: List[str],
        agent_descs: List[str],
        llm_weight: float = 0.5,
        threshold: float = 0.3,
        alpha: float = 1.0,
        max_agents: int = 3,
        use_llm: bool = True
) -> List[Tuple[str, float]]:
    cos_scores = cosine_similarity(task, agent_descs)

    llm_scores = np.array(scorer.get_batch_scores(task, agent_descs), dtype=float) if use_llm else np.zeros(
        len(agent_descs))

    final_scores = (1 - llm_weight) * cos_scores + llm_weight * llm_scores

    sorted_idx = np.argsort(final_scores)[::-1]

    candidates = []
    cum_score = 0.0

    for i in sorted_idx:
        score = float(final_scores[i])

        if score < threshold and len(candidates) > 0:
            continue

        candidates.append((i, score))
        cum_score += score

        if cum_score >= alpha:
            break

    if cum_score < alpha:
        for i in sorted_idx:
            if i in [idx for idx, _ in candidates]:
                continue
            candidates.append((i, float(final_scores[i])))
            cum_score += float(final_scores[i])

            if cum_score >= alpha:
                break

    candidates = candidates[:max_agents]

    selected = [(agent_names[i], score) for i, score in candidates]

    selected.sort(key=lambda x: x[1], reverse=True)

    return selected[:max_agents]

def get_agents(dataset_type: str) -> Tuple[List[str], List[str]]:
    dataset_type = dataset_type.lower()
    if dataset_type == "mmlu":
        p = MMLUPromptSet()
    elif dataset_type == "humaneval":
        p = HumanEvalPromptSet()
    elif dataset_type == "gsm8k":
        p = GSM8KPromptSet()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return p.split_role_label()

def get_role_num(dataset_type):
    role_num = {}
    if dataset_type.lower() == "mmlu":
        pass
    elif dataset_type.lower() == "humaneval":
        role_num = HumanEvalPromptSet().get_role_num()
    elif dataset_type.lower() == "gsm8k":
        role_num = GSM8KPromptSet().get_role_num()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return role_num

def prepruneNode(task: str, dataset_type: str):
    predictor = ThresholdPredictor("experiments/MLT/model.pt")
    beta, thr = predictor.predict(task)
    agent_names, agent_descs = get_agents(dataset_type)
    selected_agents = select_agents(
        task,
        agent_names,
        agent_descs,
        llm_weight=0.5,
        threshold=beta,
        alpha=thr,
        max_agents=5,
        use_llm=True
    )
    print("Task:", task)
    print("beta:", beta)
    print("thr:", thr)
    print("\nSelected Agents:")
    for name, score in selected_agents:
        print(f"  - {name:<25} : {score:.3f}")
    print("-" * 100)
    return  selected_agents


def GetMMLUAgent(dataset_val, limit_questions):
    def eval_loader():
        records = []
        for i_record, record in enumerate(dataset_val):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
        if len(records) > 0:
            return records
        return None

    tasks = []
    answers_task = {}

    for item in eval_loader():
        input_dict = dataset_val.record_to_input(item)
        task = input_dict["task"]
        tasks.append(task)
        answers_task[task] = dataset_val.record_to_target_answer(item)
    dataset_type = "mmlu"
    for task in tasks:
        prepruneNode(task, dataset_type)
    tracker = scorer.token_tracker
    print("\n" + "=" * 60)
    print("FINAL TOKEN USAGE")
    print(f"Total prompt tokens     : {tracker.total_prompt_tokens}")
    print(f"Total completion tokens : {tracker.total_completion_tokens}")
    print(f"Total tokens            : {tracker.total_tokens}")

def GetGsm8kAgent(dataset_json, limit_questions):
    def dataloader(data_list, limit_questions):
        if limit_questions is None:
            return data_list
        return data_list[:limit_questions]

    dataset = JSONLReader.parse_file(dataset_json)
    dataset = gsm_data_process(dataset)
    questions = dataloader(dataset, limit_questions)

    print(f"Loaded {len(questions)} questions")

    tasks = []
    answers = {}

    for i in questions:
        if isinstance(i, dict) and "task" in i and "answer" in i:
            tasks.append(i["task"])
            answers[i["task"]] = i["answer"]
        else:
            print(f"Skipping invalid item: {i}")

    dataset_type = "gsm8k"

    for task in tasks:
        prepruneNode(task, dataset_type)

    tracker = scorer.token_tracker
    print("\n" + "=" * 60)
    print("FINAL TOKEN USAGE")
    print(f"Total prompt tokens     : {tracker.total_prompt_tokens}")
    print(f"Total completion tokens : {tracker.total_completion_tokens}")
    print(f"Total tokens            : {tracker.total_tokens}")

def GetHumanevalAgent(dataset_json, limit_questions):
    def dataloader(data_list, limit_questions):
        if limit_questions is None:
            return data_list
        return data_list[:limit_questions]

    dataset = JSONLReader.parse_file(dataset_json)
    questions = dataloader(dataset, limit_questions)

    print(f"Loaded {len(questions)} questions")

    tasks = []
    for i in questions:
        if isinstance(i, dict) and "prompt" in i and "test" in i:
            tasks.append(i["prompt"])
        else:
            print(f"Skipping invalid item: {i}")

    dataset_type = "humaneval"

    for task in tasks:
        prepruneNode(task, dataset_type)

    tracker = scorer.token_tracker
    print("\n" + "=" * 60)
    print("FINAL TOKEN USAGE")
    print(f"Total prompt tokens     : {tracker.total_prompt_tokens}")
    print(f"Total completion tokens : {tracker.total_completion_tokens}")
    print(f"Total tokens            : {tracker.total_tokens}")

async def async_prepruneNode(task: str, dataset_type: str, thr: float):
    start_time = time.time()
    selected_agents = await asyncio.to_thread(prepruneNode, task, dataset_type)
    elapsed = time.time() - start_time

    agent_names = [name for name, score in selected_agents]
    agent_scores = [score for name, score in selected_agents]

    print(f"Task '{task[:40]}...' done in {elapsed:.2f}s")
    return agent_names, agent_scores, elapsed


async def prepruneNode_batch(
    tasks: List[str],
    dataset_type: str,
    max_concurrency: int = 6,
    thr: float = 1.5,
) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], List[str]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(task: str):
        async with semaphore:
            return task, await async_prepruneNode(task, dataset_type, thr)

    coros = [_run_one(task) for task in tasks]

    task2agents: Dict[str, List[str]] = {}
    task2scores: Dict[str, List[float]] = {}
    all_agents = set()
    task_times: Dict[str, float] = {}

    total_start = time.time()
    for fut in asyncio.as_completed(coros):
        task, (agent_names, agent_scores, elapsed) = await fut
        task2agents[task] = agent_names
        task2scores[task] = agent_scores
        all_agents.update(agent_names)
        task_times[task] = elapsed

    total_elapsed = time.time() - total_start
    print("\n" + "="*60)
    print("Pre-pruning Summary:")
    for t in tasks:
        print(f"Task '{t[:40]}...' -> {task_times[t]:.2f}s")
    print(f"\nTotal time for all tasks: {total_elapsed:.2f}s")
    tracker = scorer.token_tracker
    print("\n" + "=" * 60)
    print("FINAL TOKEN USAGE")
    print(f"Total prompt tokens     : {tracker.total_prompt_tokens}")
    print(f"Total completion tokens : {tracker.total_completion_tokens}")
    print(f"Total tokens            : {tracker.total_tokens}")
    print("-"*100)
    return task2agents, task2scores, list(all_agents)
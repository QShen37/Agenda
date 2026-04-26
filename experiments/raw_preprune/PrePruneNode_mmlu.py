import sys
import os

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import asyncio
import pandas as pd
import os
from LLM_score import LLMScorer
from PrePrune.prompt import PromptSetRegistry, MMLUPromptSet, HumanEvalPromptSet, GSM8KPromptSet

scorer = LLMScorer(
    use_llm=True,
    model="qwen-plus",
    max_retries=3,
    sleep_base=0.6
)

# ------------------ Load tasks from CSV ------------------
def load_task_from_csv(csv_path: str) -> List[Dict]:
    """
    Load multiple choice questions from a CSV file.
    Expected CSV format (6 columns): stem, A, B, C, D, answer
    Returns a list of tasks:
        {'stem':..., 'options': {'A':..., 'B':..., 'C':..., 'D':...}, 'answer': 'A'}
    """
    df = pd.read_csv(csv_path, header=0, encoding='utf-8', dtype=str)
    if df.shape[1] < 6:
        raise ValueError(f"CSV file has less than 6 columns: {csv_path}")
    cols = df.columns.tolist()[:6]
    tasks = []
    for _, row in df.iterrows():
        stem = str(row[cols[0]])
        options = {
            'A': str(row[cols[1]]),
            'B': str(row[cols[2]]),
            'C': str(row[cols[3]]),
            'D': str(row[cols[4]])
        }
        answer = str(row[cols[5]]).strip()
        tasks.append({'stem': stem, 'options': options, 'answer': answer})
    return tasks

# Load m random questions from all CSV files in a folder
def load_task_from_csvs_random(folder: str, m: int) -> List[Dict]:
    """
    Randomly sample m multiple-choice questions from all CSV files in the folder.
    Each row must contain at least 6 columns: stem, A, B, C, D, answer.
    Returns a list of tasks:
        {'stem':..., 'options': {'A':..., 'B':..., 'C':..., 'D':...}, 'answer': ...}
    """
    all_csv_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".csv")
    ]
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in folder {folder}.")

    all_rows = []
    for csv_path in all_csv_files:
        try:
            df = pd.read_csv(csv_path, header=0, encoding="utf-8", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, header=0, encoding="gbk", dtype=str)

        if df.shape[1] < 6:
            print(f"[Warning] File {csv_path} has less than 6 columns, skipped.")
            continue

        cols = df.columns.tolist()[:6]

        for _, row in df.iterrows():
            task = {
                "stem": str(row[cols[0]]),
                "options": {
                    "A": str(row[cols[1]]),
                    "B": str(row[cols[2]]),
                    "C": str(row[cols[3]]),
                    "D": str(row[cols[4]]),
                },
                "answer": str(row[cols[5]]).strip()
            }
            all_rows.append(task)

    if len(all_rows) == 0:
        raise ValueError("No CSV data available. Please check the file contents.")

    if m > len(all_rows):
        print(f"[Info] Requested {m} rows but only {len(all_rows)} available. Using all.")
        m = len(all_rows)

    sampled = random.sample(all_rows, m)
    return sampled

# Load embedding model
def load_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Successfully loaded model all-MiniLM-L6-v2!")
        return model
    except Exception:
        print("sentence-transformers not available, falling back to TF-IDF.")
        return None

_MODEL = load_model()

# Encode a list of texts into vectors.
# Prefer SentenceTransformer, fallback to TF-IDF + TruncatedSVD if not available.
def embed_texts(texts: List[str]) -> np.ndarray:
    if _MODEL is not None:
        return _MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    n_comp = min(128, X.shape[1]-1 if X.shape[1] > 1 else 1)
    svd = TruncatedSVD(n_components=n_comp)
    return svd.fit_transform(X)

# L2 normalize rows for cosine similarity computation
def normalize_rows(X: np.ndarray, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)

# Simple keyword matching score: ratio of task words appearing in agent description
def score_by_keywords(task: str, agent_desc: str) -> float:
    kws = [w.lower() for w in task.split() if len(w) >= 2]
    desc = agent_desc.lower()
    match = sum(1 for k in kws if k in desc)
    return float(match) / (len(kws) + 1e-6)

# Compute cosine similarity between task and agent descriptions
def cos_sim_task2agent(task: str,
                       agents_descs: List[str],
                       method: str,
                       precomputed_agent_emb: Optional[np.ndarray]):
    agent_emb = None
    if method in ("embedding", "hybrid"):
        if precomputed_agent_emb is None:
            agent_emb = embed_texts(agents_descs)
            agent_emb = normalize_rows(agent_emb)
        else:
            agent_emb = precomputed_agent_emb
    if method == "keyword":
        base_scores = np.array([score_by_keywords(task, d) for d in agents_descs])
    else:
        task_emb = normalize_rows(embed_texts([task]))[0]
        base_scores = agent_emb @ task_emb
    adjusted = (1 + base_scores) / 2
    return adjusted, agent_emb

# Get LLM evaluation scores between task and agents
def get_llm_score(task: str,
                  agents_descs: List[str]):
    llm_scores = scorer.llm_score(task, agents_descs)
    llm_scores = np.asarray(llm_scores, dtype=float)
    return llm_scores

# Select agents for a task
# selected: agents chosen for current task
# available: remaining available agents
# Ensures multiple similar agents are not chosen for the same task
def select_agent(task: str,
                 r: int,
                 thr: float,
                 method: str,
                 llm_weight: float,
                 use_llm: bool,
                 topk: bool,
                 dataset_type: str,
                 inthr: float):

    # Choose prompt set according to dataset type
    if dataset_type.lower() == "mmlu":
        p = MMLUPromptSet()
    elif dataset_type.lower() == "humaneval":
        p = HumanEvalPromptSet()
    elif dataset_type.lower() == "gsm8k":
        p = GSM8KPromptSet()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    names, agent_descs = p.split_role_label()
    m = len(agent_descs)
    if m == 0:
        return [], [], [], [], []

    # Compute similarity
    agent_emb = normalize_rows(embed_texts(agent_descs))
    cos_sim, agent_emb = cos_sim_task2agent(task, agent_descs, method, agent_emb)
    cos_sim = np.clip(cos_sim, 0.0, 1.0)

    # Integrate LLM scores
    llm_scores = get_llm_score(task, agent_descs) if use_llm else np.zeros(m)
    adjusted = (1 - llm_weight) * cos_sim + llm_weight * llm_scores
    print(f"adjusted: {adjusted}")

    # Keep only candidates with adjusted score >= inthr
    available = {i for i in range(m) if adjusted[i] >= inthr}
    if not available:
        return [], llm_scores, cos_sim, adjusted, names

    print(f"available: {available}")
    selected = []
    total_score = 0.0

    # Selection loop
    while available:
        best_j = max(available, key=lambda j: adjusted[j])
        selected.append((best_j, float(adjusted[best_j])))
        total_score += float(adjusted[best_j])
        available.remove(best_j)

        # Stop if threshold reached
        if total_score >= thr:
            break

        # Optional topk limit
        if topk and len(selected) >= min(r, m):
            break

    return selected, llm_scores, cos_sim, adjusted, names

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

def prepruneNode(task: str, dataset_type: str, thr: float):

    # predictor = ThresholdPredictor("MLT/model.pt")
    # beta, thr = predictor.predict(task)
    r = 3
    beta = 0.5
    method = "hybrid"
    llm_weight = 0.5
    use_llm = True
    topk = False

    results, llm_scores, cos_sim, adjusted, agent_names = select_agent(
        task,
        r=r,
        thr=thr,
        method=method,
        llm_weight=llm_weight,
        use_llm=use_llm,
        topk=topk,
        dataset_type=dataset_type,
        inthr=beta
    )
    agent_res_name = [agent_names[x[0]] for x in results]
    agent_res_score = [x[1] for x in results]
    print(f"\ntask: {task}")
    print(f"thr: {thr}")
    print(f"beta: {beta}")
    print(f"agent_res_name: {agent_res_name}")
    print(f"agent_res_score: {agent_res_score}")
    return agent_res_name, agent_res_score

# ----------------- Async versions -----------------
async def async_prepruneNode(task: str, dataset_type: str, thr: float):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, prepruneNode, task, dataset_type, thr)

async def prepruneNode_batch(
    tasks: List[str],
    dataset_type: str,
    max_concurrency: int = 6,
    thr: float = 1.5
):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(task: str):
        async with semaphore:
            return task, await async_prepruneNode(task, dataset_type, thr)

    coros = [_run_one(task) for task in tasks]

    task2agents: Dict[str, List[str]] = {}
    task2scores: Dict[str, List[float]] = {}
    all_agents = set()

    for fut in tqdm(
        asyncio.as_completed(coros),
        total=len(coros),
        desc="Pre-pruning tasks"
    ):
        task, (agent_names, agent_scores) = await fut
        task2agents[task] = agent_names
        task2scores[task] = agent_scores
        all_agents.update(agent_names)

    return task2agents, task2scores, list(all_agents)

def test_prepruneNode():
    """
    Simple test for prepruneNode function
    """

    # Example tasks
    tasks = [
        "What is the time complexity of binary search?",
        "Write a Python function to reverse a linked list.",
        "If a train travels 60 km in 1 hour, what is its speed in m/s?"
    ]

    dataset_type = "humaneval"   # choose from: mmlu / humaneval / gsm8k
    thr = 1.5

    print("=" * 60)
    print("Testing prepruneNode")
    print("=" * 60)

    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}: {task}")

        try:
            agent_names, agent_scores = prepruneNode(
                task=task,
                dataset_type=dataset_type,
                thr=thr
            )

            print("Selected Agents:")
            for name, score in zip(agent_names, agent_scores):
                print(f"  {name}  ->  {score:.4f}")

        except Exception as e:
            print(f"Error processing task: {e}")

    print("\nTest finished.")

if __name__ == "__main__":
    test_prepruneNode()
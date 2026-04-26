import time
import asyncio
from typing import Optional, Iterator, List, Any, Dict
from tqdm import tqdm
import copy
import torch

from PrePrune.graph.vgae_graph import Graph
from PrePrune.prompt import MMLUPromptSet
from experiments.PrePruneNode_mmlu import prepruneNode_batch
from experiments.accuracy import Accuracy
from PrePrune.utils.globals import Cost, PromptTokens, CompletionTokens


async def evaluate(
    graph: Graph,
    dataset,
    num_rounds: int = 1,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 4,
    domain: str = "mmlu",
) -> float:

    print(f"[VGAE EVAL] {dataset.__class__.__name__}")

    accuracy = Accuracy()

    prompt_set = MMLUPromptSet()
    all_agents, _ = prompt_set.split_role_label()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    tasks: List[str] = []
    answers_task: Dict[str, str] = {}

    for batch in eval_loader(eval_batch_size):
        for record in batch:
            input_dict = dataset.record_to_input(record)
            task = input_dict["task"]
            tasks.append(task)
            answers_task[task] = dataset.record_to_target_answer(record)

    task2agents, task2scores, total_agents = await prepruneNode_batch(
        tasks,
        domain,
        max_concurrency=6,
        thr=2.5
    )

    agent_locks: Dict[str, asyncio.Lock] = {
        a: asyncio.Lock() for a in all_agents
    }

    async def run_one_task(task: str, agents: List[str]):

        locks = [agent_locks[a] for a in sorted(agents)]
        for lk in locks:
            await lk.acquire()

        try:
            realized_graph = copy.deepcopy(graph)

            input_dict = dataset.str_task_to_input(task)
            correct_answer = answers_task[task]

            out = await realized_graph.node_run(
                input_dict,
                agents,
                num_rounds
            )

            if isinstance(out, tuple) and len(out) == 3:
                raw_answer, log_prob, edge_weight = out
            else:
                raw_answer = out
                log_prob = None
                edge_weight = None

            answer = dataset.postprocess_answer(raw_answer)
            print("\n" + "=" * 60)
            print(task)
            print("agents:", agents)
            print("answer:", answer)
            print("correct:", correct_answer)
            accuracy.update(answer, correct_answer)
            print(f"cost: {Cost.instance().value}")
            print(f"prompt tokens: {PromptTokens.instance().value}")
            print(f"completion tokens: {CompletionTokens.instance().value}")
            accuracy.print()

            return {
                "task": task,
                "agents": agents,
                "raw_answer": raw_answer,
                "log_prob": log_prob,
                "edge_weight": edge_weight,
                "correct_answer": correct_answer,
                "graph": realized_graph
            }

        finally:
            for lk in reversed(locks):
                lk.release()

    start_ts = time.time()

    coros = [
        run_one_task(task, agents)
        for task, agents in task2agents.items()
    ]
    last_graph = None

    for fut in tqdm(
        asyncio.as_completed(coros),
        total=len(coros),
        desc="VGAE eval running"
    ):
        res = await fut

        last_graph = res["graph"]


    print("\n" + "=" * 60)
    print("VGAE FINAL RESULT")
    print(f"time: {time.time() - start_ts:.3f}s")
    print(f"cost: {Cost.instance().value}")
    print(f"prompt tokens: {PromptTokens.instance().value}")
    print(f"completion tokens: {CompletionTokens.instance().value}")
    accuracy.print()
    #
    # if last_graph is not None:
    #     last_graph.graph_update_memory()

    return accuracy.get()
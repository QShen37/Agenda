import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List, Union, Literal
import random
import sys
import os
from datetime import datetime
from collections import Counter

from PrePruneNode_mmlu import prepruneNode_batch,get_role_num

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "logs/run_humaneval")
os.makedirs(LOG_DIR, exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PrePrune.graph.vgae_graph import Graph
from PrePrune.tools.reader.readers import JSONLReader
from PrePrune.tools.coding.python_executor import PyExecutor
from PrePrune.utils.globals import Time
from PrePrune.utils.const import GDesigner_ROOT
from PrePrune.utils.globals import Cost, PromptTokens, CompletionTokens


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size:i_batch * batch_size + batch_size]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="PrePrune Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="kimi-k2-0905-preview")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25, help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10, help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",
                        help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the PrePrune')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')

    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args


async def main():
    args = parse_args()
    result_file = None

    log_filename = datetime.now().strftime(
        f"{args.llm_name}_output_%Y-%m-%d_%H-%M-%S.txt"
    )
    log_filepath = os.path.join(LOG_DIR, log_filename)

    sys.stdout = Tee(sys.stdout, open(log_filepath, "w", encoding="utf-8"))
    sys.stderr = sys.stdout

    print("程序开始运行...")
    print(f"日志文件保存到: {log_filepath}")

    ROLE_NUM = get_role_num(args.domain)

    dataset = JSONLReader.parse_file(args.dataset_json)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]

    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(domain="humaneval",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.vgae.train()
    optimizer = torch.optim.Adam(graph.vgae.parameters(), lr=args.lr)

    num_batches = int( 80 / args.batch_size)
    total_solved, total_executed = (0, 0)
    Total_batch_time = 0.0
    num_batches = 1
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}", 80 * '-')
        start_ts = time.time()

        current_batch_tasks = dataloader(dataset, args.batch_size, i_batch)
        tasks = []
        tests = {}
        for i in current_batch_tasks:
            tasks.append(i["prompt"])
            tests[i["prompt"]] = i["test"]
        if current_batch_tasks is None:
            print("No more data available.")
            break
        task2agents, task2scores, all_agents = await prepruneNode_batch(
            tasks,
            args.domain,
            max_concurrency=6,
            thr = 1.5
        )

        role_semaphores = {
            role: asyncio.Semaphore(num)
            for role,num in ROLE_NUM.items()
        }

        start_event = asyncio.Event()
        role_in_use = {role: 0 for role in ROLE_NUM}

        def print_role_state(prefix=""):
            states = []
            for role, limit in ROLE_NUM.items():
                used = role_in_use.get(role, 0)
                states.append(f"{role}: {used}/{limit}")
            print(f"{prefix}[STATE] " + " | ".join(states))

        async def acquire_roles(agents):
            role_count = Counter(agents)
            acquired = []

            print(f"\n[WAIT ] roles={agents}")
            print_role_state("  ")

            try:
                for role in sorted(role_count.keys()):
                    for _ in range(role_count[role]):
                        await role_semaphores[role].acquire()
                        role_in_use[role] += 1
                        acquired.append(role)

                        print(f"[ACQ  ] +{role}")
                        print_role_state("  ")

            except Exception:
                for role in acquired:
                    role_semaphores[role].release()
                    role_in_use[role] -= 1
                raise

        async def release_roles(agents):
            role_count = Counter(agents)

            for role in role_count:
                for _ in range(role_count[role]):
                    role_semaphores[role].release()
                    role_in_use[role] -= 1

                    print(f"[REL  ] -{role}")
                    print_role_state("  ")

        raw_answers = {}
        log_probs = []
        async def run_single_task(task_code, agents):
            await start_event.wait()

            await acquire_roles(agents)

            await asyncio.sleep(0)

            try:
                realized_graph = copy.deepcopy(graph)
                realized_graph.vgae = graph.vgae

                input_dict = {"task": task_code}
                print(f"test: {tests[task_code]}")

                print(f"[RUN  ] task={task_code}")
                final_answer, log_prob, edge_weight = await realized_graph.node_arun(
                    input_dict,
                    agents,
                    args.num_rounds
                )
                print(f"[DONE ] task={task_code}\n final answer: {final_answer}")

                raw_answers[task_code] = final_answer
                log_probs.append(log_prob)

            finally:
                await release_roles(agents)

        tasks = [
            run_single_task(task_code, agents)
            for task_code, agents in zip(
                task2agents.keys(),
                task2agents.values()
            )
        ]

        gather_task = asyncio.gather(*tasks)

        await asyncio.sleep(0.1)
        print("\n🚦 ALL TASKS READY — START ACQUIRE 🚦\n")
        start_event.set()

        await gather_task

        loss_list = []
        utilities = []
        data = load_result(result_file)

        for task, log_prob in zip(current_batch_tasks, log_probs):
            test = tests[task["prompt"]]
            answer = raw_answers[task["prompt"]]

            code = answer[0].lstrip("```python\n").rstrip("\n```")

            print("=" *80)
            print(f"task:\n{task}")
            print(f"test:\n{test}")

            def clean_code(answer):

                if isinstance(answer, list):
                    if len(answer) == 0:
                        return ""
                    answer = answer[0]

                code = answer.strip()

                if code.startswith("```"):
                    code = code.split("```")[1]
                    if code.startswith("python"):
                        code = code[len("python"):]

                return code.strip()

            print(f"answer:\n{clean_code(answer)}")

            is_solved, _, _ = PyExecutor().execute(
                code,
                [test],
                timeout=100
            )

            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed

            utility = is_solved
            utilities.append(utility)
            loss_list.append(-log_prob * utility)

            data.append({
                "Question": task,
                "Tests": test,
                "Attempt answer": code,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            })

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        Total_batch_time += time.time() - start_ts
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print("loss:", total_loss.item())

        if i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.vgae.eval()
            print("Start Eval")

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
    print(f"Total Batch time {Total_batch_time:.3f}")


def get_kwargs(mode: Union[
    Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'], Literal['Debate'], Literal[
        'Layered'], Literal['Star']],
               N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs}

if __name__ == '__main__':
    asyncio.run(main())

import torch
from typing import Dict, List
import numpy as np
import time
import copy

from PrePrune.graph.vgae_graph import Graph
from experiments.PrePruneNode_mmlu import prepruneNode_batch
from experiments.accuracy import Accuracy
from PrePrune.utils.globals import Cost, PromptTokens, CompletionTokens


async def train(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.0005,
    batch_size: int = 4,
    domain: str = "mmlu",
    beta: float = 0.01,
) -> None:
    """
    VGAE-based training loop for MAS topology optimization.

    Loss:
        total_loss = policy_loss + beta * KL_loss
    """

    def infinite_data_loader():
        while True:
            perm = np.random.permutation(len(dataset))
            for idx in perm:
                yield dataset[idx.item()]

    loader = infinite_data_loader()

    optimizer = torch.optim.Adam(
        graph.vgae.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    graph.vgae.train()

    tasks: List[str] = []
    answers_task: Dict[str, str] = {}

    for _, record in zip(range(batch_size * num_iters), loader):
        input_dict = dataset.record_to_input(record)
        task = input_dict["task"]

        tasks.append(task)
        answers_task[task] = dataset.record_to_target_answer(record)

    task2agents, task2scores, all_agents = await prepruneNode_batch(
        tasks,
        domain,
        max_concurrency=4,
        thr=2.5,
    )

    start_ts = time.time()

    correct_answers = []
    raw_answers = []
    log_probs = []
    edge_weights = []
    kl_losses = []

    cost_total = 0
    PromptTokens_total = 0
    CompletionTokens_total = 0

    for task, agents in task2agents.items():
        realized_graph = copy.deepcopy(graph)

        # Share same VGAE parameters
        realized_graph.vgae = graph.vgae

        input_dict = dataset.str_task_to_input(task)

        print(f"\nTask: {task}")
        print("Selected agents:", agents)

        correct_answer = answers_task[task]
        correct_answers.append(correct_answer)

        raw_answer, log_prob, edge_weight = await realized_graph.node_run(
            input_dict,
            agents,
            num_rounds,
        )

        kl_loss = realized_graph.kl_loss

        print("Edge weights:")
        print(edge_weight)

        print("Correct answer:", correct_answer)
        print("Predicted answer:", dataset.postprocess_answer(raw_answer))

        raw_answers.append(raw_answer)
        log_probs.append(log_prob)
        edge_weights.append(edge_weight)
        kl_losses.append(kl_loss)

        cost_total += Cost.instance().value
        PromptTokens_total += PromptTokens.instance().value
        CompletionTokens_total += CompletionTokens.instance().value

    loss_list: List[torch.Tensor] = []
    utilities: List[float] = []
    answers: List[str] = []

    for raw_answer, log_prob, kl_loss, correct_answer in zip(
        raw_answers,
        log_probs,
        kl_losses,
        correct_answers,
    ):
        answer = dataset.postprocess_answer(raw_answer)
        answers.append(answer)

        accuracy = Accuracy()
        accuracy.update(answer, correct_answer)

        utility = accuracy.get()
        utilities.append(utility)

        # Policy Gradient Loss
        policy_loss = -log_prob * utility

        # VGAE total loss
        total_sample_loss = policy_loss + beta * kl_loss

        loss_list.append(total_sample_loss)

    total_loss = torch.mean(torch.stack(loss_list))

    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        graph.vgae.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    avg_kl = torch.mean(torch.stack(kl_losses)).item()

    print("\nTRAINING SUMMARY：")
    print("Correct Answers:", correct_answers)
    print("Raw Answers:", raw_answers)
    print("Processed Answers:", answers)

    print(f"Batch Time: {time.time() - start_ts:.3f}s")

    print("Utilities:", utilities)

    print(f"Policy + KL Loss: {total_loss.item():.6f}")
    print(f"Average KL Loss: {avg_kl:.6f}")

    print(f"Cost: {cost_total}")
    print(f"PromptTokens: {PromptTokens_total}")
    print(f"CompletionTokens: {CompletionTokens_total}")

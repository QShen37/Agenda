from abc import ABC
from typing import List, Optional, Dict, Any

import numpy as np
import shortuuid
import torch
from torch_geometric.utils import dense_to_sparse

from PrePrune.agents.agent_registry import AgentRegistry
from PrePrune.gnn.vgae import VGAE
from PrePrune.graph.node import Node
from PrePrune.llm.profile_embedding import get_sentence_embedding
from PrePrune.prompt.prompt_set_registry import PromptSetRegistry


class Graph(ABC):
    def __init__(
        self,
        domain: str,
        llm_name: Optional[str],
        agent_names: List[str],
        decision_method: str,
        optimized_spatial: bool = False,
        initial_spatial_probability: float = 0.5,
        fixed_spatial_masks: List[List[int]] = None,
        optimized_temporal: bool = False,
        initial_temporal_probability: float = 0.5,
        fixed_temporal_masks: List[List[int]] = None,
        node_kwargs: List[Dict] = None,
    ):

        if fixed_spatial_masks is None:
            fixed_spatial_masks = [
                [1 if i != j else 0 for j in range(len(agent_names))]
                for i in range(len(agent_names))
            ]

        if fixed_temporal_masks is None:
            fixed_temporal_masks = [
                [1 for _ in range(len(agent_names))]
                for _ in range(len(agent_names))
            ]

        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)

        self.id = shortuuid.ShortUUID().random(length=4)
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal

        self.decision_node = AgentRegistry.get(
            decision_method,
            **{"domain": self.domain, "llm_name": self.llm_name},
        )

        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges = []
        self.potential_temporal_edges = []

        self.node_kwargs = (
            node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        )

        self.init_nodes()
        self.init_potential_edges()
        self.init_memory()

        self.prompt_set = PromptSetRegistry.get(domain)
        self.node_format_json = self.prompt_set.get_conclude_knowledge_format()

        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()

        # VGAE replacing GCN+MLP
        self.vgae = VGAE(
            in_channels=self.features.size(1) * 2,
            hidden_channels=128,
            latent_dim=64,
        )

        self.kl_loss = torch.tensor(0.0)

        init_spatial_logit = (
            torch.log(
                torch.tensor(
                    initial_spatial_probability / (1 - initial_spatial_probability)
                )
            )
            if optimized_spatial
            else 10.0
        )

        self.spatial_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_spatial_edges)) * init_spatial_logit,
            requires_grad=optimized_spatial,
        )

        self.spatial_masks = torch.nn.Parameter(
            fixed_spatial_masks,
            requires_grad=False,
        )

        init_temporal_logit = (
            torch.log(
                torch.tensor(
                    initial_temporal_probability / (1 - initial_temporal_probability)
                )
            )
            if optimized_temporal
            else 10.0
        )

        self.temporal_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_temporal_edges)) * init_temporal_logit,
            requires_grad=optimized_temporal,
        )

        self.temporal_masks = torch.nn.Parameter(
            fixed_temporal_masks,
            requires_grad=False,
        )

    def construct_adj_matrix(self):
        role_connect = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes, num_nodes))
        role_2_id = {}

        for edge in role_connect:
            in_role, out_role = edge
            role_2_id.setdefault(in_role, [])
            role_2_id.setdefault(out_role, [])

        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            if role in role_2_id:
                role_2_id[role].append(i)

        for edge in role_connect:
            in_role, out_role = edge
            for in_id in role_2_id[in_role]:
                for out_id in role_2_id[out_role]:
                    role_adj[in_id][out_id] = 1

        edge_index, _ = dense_to_sparse(role_adj)
        return edge_index

    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        return torch.tensor(np.array(features), dtype=torch.float32)

    def construct_new_feature(self, query):
        query_embedding = torch.tensor(
            get_sentence_embedding(query),
            dtype=torch.float32,
        )
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes, 1))
        return torch.cat((self.features, query_embedding), dim=1)

    def generate_spatial_logits(self, task: str):
        new_feature = self.construct_new_feature(task)

        z, mu, logstd = self.vgae.encode(
            new_feature,
            self.role_adj_matrix,
        )

        prob_matrix = self.vgae.decode(
            z,
            task_idx=self.num_nodes - 1,
        )

        self.kl_loss = self.vgae.kl_loss(mu, logstd)

        self.spatial_logits = min_max_norm(
            torch.flatten(prob_matrix)
        )

        return prob_matrix

    @property
    def num_nodes(self):
        return len(self.nodes)

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def find_node(self, node_id: str):
        return self.nodes[node_id]

    def init_nodes(self):
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_memory(self):
        print("Memory Initialization...")
        for node_id in self.nodes:
            self.nodes[node_id].get_persistent_memory(self.nodes[node_id].role)
        print("Memory Initialization Done...")

    def init_potential_edges(self):
        for node1_id in self.nodes:
            for node2_id in self.nodes:
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        for node_id in self.nodes:
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def construct_spatial_connection(
        self,
        task: str = "",
        agent_res_name: list = None,
        temperature: float = 1.0,
        threshold: float = None,
    ):
        if agent_res_name is None:
            agent_res_name = []

        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

        if not agent_res_name:
            return torch.sum(torch.stack(log_probs)), np.zeros((0, 0))

        role_to_idx = {
            role.lower(): idx for idx, role in enumerate(agent_res_name)
        }

        edge_weight = np.zeros(
            (len(agent_res_name), len(agent_res_name)),
            dtype=float,
        )

        for potential_connection, edge_logit, edge_mask in zip(
            self.potential_spatial_edges,
            self.spatial_logits,
            self.spatial_masks,
        ):
            out_node = self.find_node(potential_connection[0])
            in_node = self.find_node(potential_connection[1])

            out_role = out_node.role.lower()
            in_role = in_node.role.lower()

            if out_role not in role_to_idx or in_role not in role_to_idx:
                continue

            if edge_mask == 0:
                continue

            if out_node == in_node:
                continue

            if self.check_cycle(in_node, {out_node}):
                continue

            edge_prob = torch.sigmoid(edge_logit / temperature)
            raw_prob = float(edge_prob.detach().cpu().item())

            edge_weight[
                role_to_idx[out_role],
                role_to_idx[in_role],
            ] = raw_prob

            sample_prob = raw_prob
            if threshold is not None:
                sample_prob = 1.0 if raw_prob > threshold else 0.0

            sample_prob = max(min(sample_prob, 1.0), 1e-9)

            if torch.rand(1).item() < sample_prob:
                out_node.add_successor(in_node, "spatial")
                log_probs.append(torch.log(torch.tensor(sample_prob)))
            else:
                log_probs.append(
                    torch.log(torch.tensor(max(1 - sample_prob, 1e-9)))
                )

        return torch.sum(torch.stack(log_probs)), edge_weight

    def connect_decision_node(self):
        for node_id in self.nodes:
            self.nodes[node_id].add_successor(self.decision_node)

    def update_memory(self):
        for _, node in self.nodes.items():
            node.update_memory()

    def compute_total_loss(self, reward, log_probs, beta=0.01):
        policy_loss = -reward * log_probs
        return policy_loss + beta * self.kl_loss

    async def node_arun(
        self,
        input: Dict[str, str],
        agent_res_name: List[str],
        num_rounds: int = 3,
        max_tries: int = 3,
    ):
        log_probs = 0

        self.generate_spatial_logits(input["task"])

        active_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node.role in agent_res_name
        }

        for _ in range(num_rounds):
            res_prob, edge_weight = self.construct_spatial_connection(
                input["task"],
                agent_res_name,
            )

            log_probs += res_prob

            in_degree = {}
            for node_id in active_nodes:
                node = self.nodes[node_id]
                in_degree[node_id] = sum(
                    1
                    for pred in node.spatial_predecessors
                    if pred.id in active_nodes
                )

            queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while queue:
                current_node_id = queue.pop(0)

                tries = 0
                while tries < max_tries:
                    try:
                        await self.nodes[current_node_id].async_execute(input)
                        break
                    except Exception:
                        tries += 1

                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in active_nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        queue.append(successor.id)

                self.nodes[current_node_id].update_memorybank(
                    input["task"],
                    self.node_format_json[
                        self.nodes[current_node_id].role
                    ],
                )

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(input)

        final_answers = self.decision_node.outputs
        if not final_answers:
            final_answers = ["No answer of the decision node"]

        return final_answers, log_probs, edge_weight

    async def node_run(
            self,
            input: Dict[str, str],
            agent_res_name: List[str],
            num_rounds: int = 3,
            max_tries: int = 3,
    ):

        log_probs = 0.0

        self.generate_spatial_logits(input["task"])

        active_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node.role in agent_res_name
        }

        if len(active_nodes) == 0:
            return ["No active nodes"], log_probs, np.zeros((0, 0))

        for _ in range(num_rounds):
            res_prob, edge_weight = self.construct_spatial_connection(
                input["task"],
                agent_res_name,
            )

            log_probs += res_prob

            in_degree = {
                node_id: sum(
                    1 for pred in self.nodes[node_id].spatial_predecessors
                    if pred.id in active_nodes
                )
                for node_id in active_nodes
            }

            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while zero_in_degree_queue:

                current_node_id = zero_in_degree_queue.pop(0)
                node = self.nodes[current_node_id]

                if node is None:
                    continue

                tries = 0
                success = False

                while tries < max_tries:
                    try:
                        node.execute(input)
                        success = True
                        break
                    except Exception as e:
                        print(f"[ERROR] Node {current_node_id} failed: {e}")
                        tries += 1

                if not success or len(node.outputs) == 0:
                    print(f"[WARNING] Node {current_node_id} produced empty output")
                else:
                    node.update_memorybank(
                        input["task"],
                        self.node_format_json.get(node.role, {})
                    )

                for successor in node.spatial_successors:
                    if successor.id not in active_nodes:
                        continue

                    in_degree[successor.id] -= 1

                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            self.update_memory()

        self.connect_decision_node()
        self.decision_node.execute(input)
        final_answers = self.decision_node.outputs
        if not final_answers:
            final_answers = ["No answer of the decision node"]

        return final_answers, log_probs, edge_weight

    def graph_update_memory(self):
        for node_id in self.nodes:
            self.nodes[node_id].save_memory_to_persistent()


def min_max_norm(tensor: torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-9)
    return normalized * 2 - 1

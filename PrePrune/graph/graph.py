import asyncio
from abc import ABC
from typing import Any, List, Optional, Dict, Tuple

import numpy as np
import shortuuid
import torch
from torch_geometric.utils import dense_to_sparse

from PrePrune.agents.agent_registry import AgentRegistry
from PrePrune.gnn.gcn import GCN, MLP
from PrePrune.graph.node import Node
from PrePrune.llm.profile_embedding import get_sentence_embedding
from PrePrune.prompt.prompt_set_registry import PromptSetRegistry


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.node_format_json = self.prompt_set.get_conclude_knowledge_format()
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
        self.mlp = MLP(384,16,16)

        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
                                                 requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features

    def construct_new_feature(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features, query_embedding), dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self,
                                     task: str = '',
                                     agent_res_name : list = [],
                                     temperature: float = 1.0,
                                     threshold: float = None,
                                     debug: bool = False):
        """
        Build spatial connections but ONLY compute probabilities / sampling / log_probs
        for node pairs that are both in agent_res_name.

        Returns:
            total_logprob (torch.Tensor), edge_weight (np.ndarray)
        Notes:
            - edge_weight is a numpy matrix of shape (N_prepruned, N_prepruned) containing
              the raw sigmoid probabilities for edges between prepruned agents.
            - log_probs (and thus training signal) comes ONLY from those edges.
            - If you want a differentiable probability matrix for direct gradient updates,
              return a torch tensor prob_matrix as well (see comment below).
        """
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

        # --- get prepruned agents (replace the hardcoded list with your actual prepruneNode call) ---
        # print('selected agent: ', agent_res_name)
        # agent_res_name = ['Mathematician', 'Psychologist', 'Historian', 'Doctor', 'Lawyer']
        if not agent_res_name:
            # nothing to train on
            return torch.sum(torch.stack(log_probs)), np.zeros((0, 0))

        # normalized name -> index mapping for O(1) lookup
        def _norm(x):
            return None if x is None else str(x).strip().lower()

        name_to_idx = {}
        for idx, name in enumerate(agent_res_name):
            key = _norm(name)
            if key in name_to_idx and debug:
                print(f"Warning: duplicate normalized name {key} at idx {idx}")
            name_to_idx[key] = idx
        agent_res_set = set(name_to_idx.keys())

        # prepare numpy edge_weight for the prepruned subgraph
        edge_weight = np.zeros((len(agent_res_name), len(agent_res_name)), dtype=float)

        # (optional) if you want a torch tensor of probabilities for direct differentiation,
        # you can keep a torch tensor 'prob_matrix_tensor' and return it as well.
        # prob_matrix_tensor = torch.zeros((len(agent_res_name), len(agent_res_name)), dtype=torch.float32, device=self.spatial_logits.device)

        # helper: robustly extract and normalize node id
        def _node_id_from_node(node):
            for attr in ("role", "name", "node_name", "agent_name", "idnumber", "id"):
                if hasattr(node, attr):
                    return _norm(getattr(node, attr))
            return None

        # iterate all candidate edges but only act when both endpoints are in agent_res_set
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges,
                                                               self.spatial_logits,
                                                               self.spatial_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])

            out_id_s = _node_id_from_node(out_node)
            in_id_s = _node_id_from_node(in_node)

            # skip edges which endpoints cannot be identified or are not in prepruned set
            if out_id_s is None or in_id_s is None:
                if debug:
                    print(f"skip: cannot id node: out={out_id_s}, in={in_id_s}")
                continue
            if out_id_s not in agent_res_set or in_id_s not in agent_res_set:
                # NOTE: edges with endpoints outside prepruned list are ignored for training
                if debug:
                    if out_id_s not in agent_res_set:
                        print(f"skip edge: out_id '{out_id_s}' not in prepruned set")
                    if in_id_s not in agent_res_set:
                        print(f"skip edge: in_id '{in_id_s}' not in prepruned set")
                continue

            # edge_mask handling (identical semantics to your previous logic)
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial == False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue

            # prevent cycles
            if not self.check_cycle(in_node, {out_node}):
                # compute probability (torch tensor)
                edge_prob_tensor = torch.sigmoid(edge_logit / temperature)

                # store raw probability (float) into numpy matrix (for analysis / loss weighting)
                try:
                    raw_prob = float(edge_prob_tensor.detach().cpu().numpy())
                except Exception:
                    raw_prob = float(edge_prob_tensor.item())
                out_idx = name_to_idx[out_id_s]
                in_idx = name_to_idx[in_id_s]
                edge_weight[out_idx, in_idx] = raw_prob
                # prob_matrix_tensor[out_idx, in_idx] = edge_prob_tensor  # if you want differentiable matrix

                # only these edges contribute to log_probs (training signal)
                if threshold is not None:
                    sample_prob_tensor = torch.tensor(1.0 if raw_prob > threshold else 0.0)
                else:
                    sample_prob_tensor = torch.tensor(raw_prob)

                sample_prob_tensor = sample_prob_tensor.clamp(min=1e-9, max=1.0)
                if torch.rand(1) < sample_prob_tensor:
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(torch.log(sample_prob_tensor))
                else:
                    log_probs.append(torch.log((1 - sample_prob_tensor).clamp(min=1e-9)))

        return torch.sum(torch.stack(log_probs)), edge_weight
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))

    async def node_arun(self, input: Dict[str, str],
                       agent_res_name: List[str],
                       num_rounds:int = 3,
                       max_tries: int = 3,
                       max_time: int = 600,
                       retraining_count: int = 0, ) -> List[Any]:
        log_probs = 0
        new_feature = self.construct_new_feature(input['task'])
        logits = self.gcn(new_feature,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        active_nodes = set()

        for node_id, node in self.nodes.items():
            if node.role in agent_res_name:
                active_nodes.add(node_id)

        for round in range(num_rounds):
            # res_prob, edge_weight, mask = self.construct_spatial_node_connection(input['task'],retraining_count = retraining_count)
            # res_prob, edge_weight = self.construct_preprune_edge_spatial_connection(input['task'],retraining_count = retraining_count)
            res_prob, edge_weight = self.construct_spatial_connection(input['task'], agent_res_name)
            log_probs += res_prob

            in_degree = {}

            for node_id in active_nodes:
                node = self.nodes[node_id]
                in_degree[node_id] = sum(
                    1 for pred in node.spatial_predecessors
                    if pred.id in active_nodes
                )

            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)

                tries = 0
                while tries < max_tries:
                    try:
                        await self.nodes[current_node_id].async_execute(input)

                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1

                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in active_nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
                self.nodes[current_node_id].update_memorybank(input['task'],self.node_format_json[self.nodes[current_node_id].role])
            self.update_memory()

        print("=== Spatial Graph ===")
        for node_id, node in self.nodes.items():
            if node_id in active_nodes:
                print(node_id, "->", [s.id for s in node.spatial_successors])

        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, log_probs, edge_weight

    async def node_run(self, input: Dict[str, str],
                       agent_res_name: List[str],
                       num_rounds:int = 3,
                       max_tries: int = 3,
                       max_time: int = 600,
                       retraining_count: int = 0, ) -> List[Any]:
        log_probs = 0
        new_feature = self.construct_new_feature(input['task'])
        logits = self.gcn(new_feature,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        active_nodes = set()

        for node_id, node in self.nodes.items():
            if node.role in agent_res_name:
                active_nodes.add(node_id)

        for round in range(num_rounds):
            # res_prob, edge_weight, mask = self.construct_spatial_node_connection(input['task'],retraining_count = retraining_count)
            # res_prob, edge_weight = self.construct_preprune_edge_spatial_connection(input['task'],retraining_count = retraining_count)
            res_prob, edge_weight = self.construct_spatial_connection(input['task'], agent_res_name)
            log_probs += res_prob

            in_degree = {}

            for node_id in active_nodes:
                node = self.nodes[node_id]
                in_degree[node_id] = sum(
                    1 for pred in node.spatial_predecessors
                    if pred.id in active_nodes
                )

            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)

                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(input)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1

                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in active_nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

                self.nodes[current_node_id].update_memorybank(input['task'], self.node_format_json[self.nodes[current_node_id].role])
            self.update_memory()


        print("=== Spatial Graph ===")
        for node_id, node in self.nodes.items():
            if node_id in active_nodes:
                print(node_id, "->", [s.id for s in node.spatial_successors])

        self.connect_decision_node()
        self.decision_node.execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, log_probs, edge_weight

    async def run(self, input: Dict[str, str],
                  agent_res_name: List[str],
                  num_rounds:int = 3,
                  max_tries: int = 3,
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0

        for round in range(num_rounds):
            log_probs_temp, edge_weight = self.construct_spatial_connection(input['task'], agent_res_name)
            log_probs += log_probs_temp
            log_probs += self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(input) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        return final_answers, log_probs, edge_weight

    async def arun(self, input: Dict[str,str],
                  num_rounds:int = 3,
                  max_tries: int = 3,
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        new_features = self.construct_new_features(input['task'])
        logits = self.gcn(new_features,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs_temp, edge_weight = self.construct_spatial_connection(input['task'])
            log_probs += log_probs_temp
            log_probs += self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs, edge_weight
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
            # print("Memory:", node.memory_bank)
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    
import asyncio
import os
import json
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict

import shortuuid


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.

    Methods:
        add_predecessor(operation):
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation):
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs):
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs):
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self,
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="",
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}
        self.memory_bank: List[Any] = []
        self.memory_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                "..",
                "..",
                "Memory",
                "intrinsic_memory.json"
            )
        )

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []

    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def get_persistent_memory(self, role):
        if not os.path.exists(self.memory_file):
            return self.memory_bank

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                all_memory = json.load(f)

            stored_memory = all_memory.get(self.role, {})

            if stored_memory is None:
                self.memory_bank = []
            elif isinstance(stored_memory, list):
                self.memory_bank = stored_memory
            else:
                self.memory_bank = [stored_memory]

            print("[Memory Load]", role, self.memory_bank)

        except Exception as e:
            print(f"[Memory Load Error] {e}")
            self.memory_bank = []

        return self.memory_bank

    def update_memorybank(self, input, Knowledge):
        if self.memory_bank is None:
            self.memory_bank = []

        if len(self.memory_bank) == 0:
            self.execute_conclude(input, Knowledge)
        else:
            self.execute_conclude(input, self.memory_bank[0])

        self.save_memory_to_persistent()

    def save_memory_to_persistent(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        all_memory = {}

        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    all_memory = json.load(f)
            except Exception as e:
                print(f"[Memory Load Error] {e}")
                all_memory = {}

        all_memory[self.role] = self.memory_bank[0] if self.memory_bank else {}

        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(all_memory, f, indent=4, ensure_ascii=False)

            print(f"[Memory Saved] {self.role}")
            print("[Memory]", all_memory[self.role])

        except Exception as e:
            print(f"[Memory Save Error] {e}")

    def get_spatial_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return temporal_info

    def execute(self, input: Any, **kwargs):
        self.outputs = []
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()
        prompt_input = input
        if isinstance(input, dict):
            prompt_input = dict(input)
            if self.memory_bank:
                prompt_input["memory"] = self.memory_bank[0]
        elif isinstance(input, str):
            if self.memory_bank:
                prompt_input = input + f"\n[Memory]: {self.memory_bank[0]}"
        else:
            prompt_input = str(input)
        results = [self._execute(prompt_input, spatial_info, temporal_info, **kwargs)]
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    def execute_conclude(self, input: Any, Knowledge, **kwargs):
        result = self._execute_conclude(
            input,
            self.outputs[0] if self.outputs else "",
            Knowledge,
            self.role
        )
        if self.memory_bank is None:
            self.memory_bank = []
        if result is not None:
            self.memory_bank = [result]
        return self.memory_bank

    async def async_execute(self, input:Any, **kwargs):

        self.outputs = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """

    @abstractmethod
    def _execute_conclude(self, input:str, output:str, knowledge: dict[str,Any], role:str, **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

from typing import List, Any, Dict, Tuple

from PrePrune.graph.node import Node
from PrePrune.agents.agent_registry import AgentRegistry
from PrePrune.llm.llm_registry import LLMRegistry
from PrePrune.prompt.prompt_set_registry import PromptSetRegistry
from PrePrune.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        spatial_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):  # is python code
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}.\n At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response

    def _execute_conclude(self,  input, answer, Knowledge, role, **kwargs):
        """
        Conclude and summarize experience into structured memory (JSON format)
        """

        system_prompt = self.prompt_set.get_description(role)

        import json
        knowledge_schema_str = json.dumps(Knowledge, indent=2, ensure_ascii=False)

        user_prompt = f"""
        ## Task
        You are updating an agent's structured memory.

        Given the current memory (JSON schema) and a new answer,
        refine and update the memory by extracting only the most valuable and generalizable knowledge.

        ## Input
        Question:
        {input}

        Final Answer:
        {answer}

        Current Memory (JSON Schema):
        {knowledge_schema_str}

        ## Instructions
        1. DO NOT change the JSON structure (keys must remain exactly the same).
        2. Update the existing fields using the new answer.
        3. Keep only high-value, reusable knowledge.
        4. REMOVE redundant, duplicate, or low-information content.
        5. Prefer concise and general insights over specific details.
        6. Limit the total number of items in each list (keep it compact).

        ## Output Requirements
        - Output ONLY valid JSON
        - No explanation
        - No extra text
        """

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm.gen(message)

        if response == "" or response == "None":
            return Knowledge

        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            # fallback
            return Knowledge


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                        debug: bool = False, **kwargs) -> Tuple[str, str]:
        """
        Robust version of _process_inputs.
        Returns: (system_prompt, user_prompt)
        - If some node info has missing/None 'output', treat it as empty string (or '<no output>' if debug).
        - If raw_inputs is not a dict or doesn't contain 'task', fallback to str(raw_inputs).
        - debug=True will print helpful diagnostics.
        """
        # assemble system prompt as before
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{self.role}.\n {self.constraint}"

        # helper to safely extract output string from info
        def _safe_output(node_id, info):
            if info is None:
                if debug:
                    print(f"[_process_inputs] warning: spatial_info[{node_id}] is None")
                return ""  # or return f"<no output from {node_id}>"
            if not isinstance(info, dict):
                if debug:
                    print(
                        f"[_process_inputs] warning: spatial_info[{node_id}] is not dict (type={type(info)}). Converting to str.")
                return str(info)
            # info is dict-like
            out_val = info.get('output', None)
            if out_val is None:
                if debug:
                    print(f"[_process_inputs] warning: spatial_info[{node_id}].get('output') is None or missing.")
                return ""
            return str(out_val)

        # build spatial string robustly
        spatial_str = ""
        if spatial_info:
            for node_id, info in spatial_info.items():
                node_id_str = str(node_id)
                out_text = _safe_output(node_id_str, info)
                spatial_str += f"{node_id_str}: {out_text}\n\n"

        # you previously didn't use temporal_info in prompt, but we keep it available (optional)
        # temporal_str = ""
        # if temporal_info:
        #     for t_id, t_info in temporal_info.items():
        #         temporal_str += f"{t_id}: {_safe_output(t_id, t_info)}\n\n"

        decision_few_shot = self.prompt_set.get_decision_few_shot()

        # safe extraction of task text
        task_text = ""
        if isinstance(raw_inputs, dict):
            task_text = raw_inputs.get('task', "")
            if task_text is None:
                if debug:
                    print("[_process_inputs] warning: raw_inputs['task'] is None")
                task_text = ""
        else:
            # fallback: raw_inputs may be a string or other structure
            task_text = str(raw_inputs)

        user_prompt = (
            f"{decision_few_shot} The task is:\n\n {task_text}.\n "
            f"At the same time, the output of other agents is as follows:\n\n{spatial_str}"
        )

        if debug:
            print("[_process_inputs] system_prompt preview:\n", system_prompt[:1000])
            print("[_process_inputs] user_prompt preview:\n", user_prompt[:1000])

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        user_prompt += 'If all experts output "ACCEPT", then summarize; otherwise, summarize after reasoning.'

        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        # print("final refer:")
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response

    def _execute_conclude(self, input, answer, Knowledge, role, **kwargs):
        """
        Conclude and summarize experience into structured memory (JSON format)
        """

        system_prompt = self.prompt_set.get_description(role)

        import json
        knowledge_schema_str = json.dumps(Knowledge, indent=2, ensure_ascii=False)

        user_prompt = f"""
        ## Task
        You are updating an agent's structured memory.

        Given the current memory (JSON schema) and a new answer,
        refine and update the memory by extracting only the most valuable and generalizable knowledge.

        ## Input
        Question:
        {input}

        Final Answer:
        {answer}

        Current Memory (JSON Schema):
        {knowledge_schema_str}

        ## Instructions
        1. DO NOT change the JSON structure (keys must remain exactly the same).
        2. Update the existing fields using the new answer.
        3. Keep only high-value, reusable knowledge.
        4. REMOVE redundant, duplicate, or low-information content.
        5. Prefer concise and general insights over specific details.
        6. Limit the total number of items in each list (keep it compact).

        ## Output Requirements
        - Output ONLY valid JSON
        - No explanation
        - No extra text
        """

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm.gen(message)

        if response == "" or response == "None":
            return Knowledge

        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            # fallback
            return Knowledge

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output

    def _execute_conclude(self,  input, answer, Knowledge, role, **kwargs):
        """
        Conclude and summarize experience into structured memory (JSON format)
        """

        system_prompt = self.prompt_set.get_description(role)

        import json
        knowledge_schema_str = json.dumps(Knowledge, indent=2, ensure_ascii=False)

        user_prompt = f"""
        ## Task
        You are updating an agent's structured memory.

        Given the current memory (JSON schema) and a new answer,
        refine and update the memory by extracting only the most valuable and generalizable knowledge.

        ## Input
        Question:
        {input}

        Final Answer:
        {answer}

        Current Memory (JSON Schema):
        {knowledge_schema_str}

        ## Instructions
        1. DO NOT change the JSON structure (keys must remain exactly the same).
        2. Update the existing fields using the new answer.
        3. Keep only high-value, reusable knowledge.
        4. REMOVE redundant, duplicate, or low-information content.
        5. Prefer concise and general insights over specific details.
        6. Limit the total number of items in each list (keep it compact).

        ## Output Requirements
        - Output ONLY valid JSON
        - No explanation
        - No extra text
        """

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm.gen(message)

        if response == "" or response == "None":
            return Knowledge

        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            # fallback
            return Knowledge


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            print(processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

    def _execute_conclude(self,  input, answer, Knowledge, role, **kwargs):
        """
        Conclude and summarize experience into structured memory (JSON format)
        """

        system_prompt = self.prompt_set.get_description(role)

        import json
        knowledge_schema_str = json.dumps(Knowledge, indent=2, ensure_ascii=False)

        user_prompt = f"""
        ## Task
        You are updating an agent's structured memory.

        Given the current memory (JSON schema) and a new answer,
        refine and update the memory by extracting only the most valuable and generalizable knowledge.

        ## Input
        Question:
        {input}

        Final Answer:
        {answer}

        Current Memory (JSON Schema):
        {knowledge_schema_str}

        ## Instructions
        1. DO NOT change the JSON structure (keys must remain exactly the same).
        2. Update the existing fields using the new answer.
        3. Keep only high-value, reusable knowledge.
        4. REMOVE redundant, duplicate, or low-information content.
        5. Prefer concise and general insights over specific details.
        6. Limit the total number of items in each list (keep it compact).

        ## Output Requirements
        - Output ONLY valid JSON
        - No explanation
        - No extra text
        """

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm.gen(message)

        if response == "" or response == "None":
            return Knowledge

        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            # fallback
            return Knowledge
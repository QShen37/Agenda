from typing import List,Any,Dict
import re

from PrePrune.graph.node import Node
from PrePrune.agents.agent_registry import AgentRegistry
from PrePrune.llm.llm_registry import LLMRegistry
from PrePrune.prompt.prompt_set_registry import PromptSetRegistry
from PrePrune.tools.search.wiki import search_wiki_main

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None =None, role:str = None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""

    # def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
    #     """ To be overriden by the descendant class """
    #     """ Process the raw_inputs(most of the time is a List[Dict]) """
    #     system_prompt = f"{self.constraint}"
    #     user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
    #     user_prompt += "Please continue to complete this task based on the previous conversation."
    #     return system_prompt, user_prompt

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict],
                              temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = f"{self.constraint}" if self.role != 'Fake' else "You are a liar who only tells lies."
        # system_prompt = f"{self.constraint}" if self.role == "Normal" else "You are a gibbering idiot"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(
            raw_inputs['task'])
        if self.role == 'Fake':
            return system_prompt, user_prompt
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(
            spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(
            temporal_str) else ""
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        user_prompt += 'If you have not received any such knowledge or if you have already learned from previous experts and strongly agree with their views, output "ACCEPT" after the reasoning. Otherwise, do not output "ACCEPT" after the reasoning.'
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        if response == "" or response == "None":
            response = "I am sorry, I do not have any information about this topic."
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        if response == "" or response == "None":
            response = "I am sorry, I do not have any information about this topic."
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
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
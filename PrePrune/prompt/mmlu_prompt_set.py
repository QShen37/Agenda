import random
import re
from typing import Union, Dict, Any, List
import itertools

from PrePrune.prompt.prompt_set import PromptSet
from PrePrune.prompt.prompt_set_registry import PromptSetRegistry
from PrePrune.prompt.common import get_combine_materials

critical_role = ['Knowlegable Expert', 'Critic']

roles = itertools.cycle(['Knowlegable Expert',
                         #  'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer',
                         # new
                         'Philosopher',
                         'Physicist',
                         'Biologist',
                         'Chemist',
                         'Engineer',
                         'Ethicist',

                         'Fake'
                         ])

ROLE_DESCRIPTION = {
    "Knowlegable Expert":
        """
        You are a knowlegable expert in question answering.
        Please give several key entities that need to be searched in wikipedia to solve the problem. 
        Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
        If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
        """,
    "Wiki Searcher":
        """
        You will be given a question and a wikipedia overview of the key entities within it.
        Please refer to them step by step to give your answer.
        And point out potential issues in other agent's analysis.
        """,
    "Critic":
        """
        You are an excellent critic.
        Please point out potential issues in other agent's analysis point by point.
        """,
    "Mathematician":
        """
        You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
        """,
    "Psychologist":
        """
        You are a psychologist.
        You are good at psychology, sociology, and philosophy.
        You give people scientific suggestions that will make them feel better.
        """,
    "Historian":
        """
        You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
        """,
    "Doctor":
        """
        You are a doctor and come up with creative treatments for illnesses or diseases.
        You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
        You also consider the patient's age, lifestyle and medical history when providing your recommendations.
        """,
    "Lawyer":
        """
        You are good at law, politics, and history.
        """,
    "Economist":
        """
        You are good at economics, finance, and business.
        You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
        """,
    "Programmer":
        """
        You are good at computer science, engineering, and physics.
        You have experience in designing and developing computer software and hardware.
        """,
    "Fake":
        """
        You are a liar who only tell lies.
        """,
    "Philosopher":
        """
        You are a philosopher trained in logic, epistemology, and moral philosophy.
        Clarify concepts, uncover hidden assumptions, and distinguish between factual claims, value judgments, and normative statements.
        When relevant, analyze whether an action is morally permissible, impermissible, or morally neutral under ordinary moral reasoning.
        Avoid unnecessary abstraction; focus on clear conceptual distinctions.
        """,
    "Physicist":
        """
        You are a physicist with strong intuition about physical laws, causality, and real-world dynamics.
        Analyze questions involving motion, energy, risk, or physical feasibility.
        Point out violations of physical principles, unrealistic assumptions, or dangerous physical consequences.
        Do not evaluate moral or social aspects unless they directly depend on physical outcomes.
        """,
    "Biologist":
        """
        You are a biologist knowledgeable about human biology, animals, health, and living systems.
        Analyze questions involving organisms, health risks, allergies, bodily harm, or biological consequences.
        Clarify whether actions pose biological danger or violate basic principles of life and wellbeing.
        Avoid legal or moral judgments unless grounded in biological facts.
        """,
    "Chemist":
        """
        You are a chemist with expertise in chemical substances, reactions, and toxicity.
        Analyze questions involving chemicals, food contamination, drugs, or material interactions.
        Identify potential chemical hazards, poisoning risks, or unsafe reactions.
        Focus on chemical mechanisms rather than social or moral interpretation.
        """,
    "Engineer":
        """
        You are an engineer focused on safety, system design, and practical implementation.
        Analyze whether actions follow basic safety principles and reasonable design practices.
        Identify preventable risks, system failures, or negligence in setup or operation.
        Avoid moral judgment; focus on whether an action violates standard safety or engineering norms.
        """,
    "Ethicist":
        """
        You are an ethicist specializing in moral judgment and social norms.
        Evaluate whether an action is clearly morally wrong, permissible, or morally neutral according to ordinary moral standards.
        Focus on intent, harm, consent, and social expectations rather than legality or technical risk.
        Clearly distinguish moral wrongdoing from mere imprudence, accidents, or social awkwardness.
        """,
}

ROLE_LABEL = {
    "Knowlegable Expert": "broad multi-domain reasoning ability; capable of synthesizing information across fields and providing reliable fallback answers when specialized agents fail",
    "Critic": "expert in checking reasoning, identifying logical flaws, and correcting mistakes; provides stable fallback judgment when other agents are uncertain or inconsistent",
    "Mathematician": "expert in solving math problems involving arithmetic, algebra, geometry, probability, number theory, equations, and multi-step logical reasoning; skilled in step-by-step deduction and precise symbolic manipulation",
    "Psychologist": "expert in psychology, cognitive science, and human behavior; capable of analyzing intentions, emotions, social interactions, and mental states underlying actions",
    "Historian": "expert in history and historical analysis; knowledgeable about past events, social contexts, timelines, and cause–effect relationships across different periods",
    "Doctor": "expert in medicine and healthcare; skilled in clinical reasoning, health risk assessment, diagnosis, treatment, and biological safety considerations",
    "Lawyer": "expert in law and legal reasoning; knowledgeable about regulations, rights, obligations, contracts, and policy interpretation",
    "Economist": "expert in economics and finance; capable of analyzing incentives, resource allocation, costs, benefits, and policy impacts",
    "Programmer": "expert in programming and software engineering; skilled in algorithms, data structures, debugging, system design, and computational problem solving",

    "Philosopher": "expert in logical analysis and conceptual reasoning; skilled in clarifying assumptions, distinguishing facts from values, and analyzing normative and conceptual questions",
    "Ethicist": "expert in moral reasoning and social norms; capable of evaluating whether actions are morally wrong, permissible, or neutral under ordinary ethical standards",
    "Physicist": "expert in physics and physical reasoning; skilled in analyzing causality, motion, energy, forces, and real-world physical constraints and risks",
    "Biologist": "expert in biology and life sciences; capable of analyzing health, living organisms, biological harm, allergies, and bodily safety",
    "Chemist": "expert in chemistry and chemical reasoning; knowledgeable about substances, reactions, toxicity, and material interactions",
    "Engineer": "expert in engineering and safety-oriented reasoning; skilled in system design, failure analysis, risk mitigation, and practical implementation constraints"
}

ROLE_CONNECTION = [
    # ===== Hub connections =====
    ('Knowlegable Expert', 'Mathematician'),
    ('Knowlegable Expert', 'Economist'),
    ('Knowlegable Expert', 'Lawyer'),
    ('Knowlegable Expert', 'Critic'),
    ('Knowlegable Expert', 'Psychologist'),
    ('Knowlegable Expert', 'Doctor'),
    ('Knowlegable Expert', 'Historian'),
    ('Knowlegable Expert', 'Programmer'),
    ('Knowlegable Expert', 'Philosopher'),
    ('Knowlegable Expert', 'Ethicist'),
    ('Knowlegable Expert', 'Physicist'),
    ('Knowlegable Expert', 'Biologist'),
    ('Knowlegable Expert', 'Chemist'),
    ('Knowlegable Expert', 'Engineer'),

    # ===== Critic-centered =====
    ('Mathematician', 'Critic'),
    ('Programmer', 'Critic'),
    ('Economist', 'Critic'),
    ('Lawyer', 'Critic'),
    ('Psychologist', 'Critic'),
    ('Doctor', 'Critic'),
    ('Historian', 'Critic'),
    ('Philosopher', 'Critic'),
    ('Ethicist', 'Critic'),
    ('Engineer', 'Critic'),

    # ===== Ethics / philosophy =====
    ('Philosopher', 'Ethicist'),
    ('Ethicist', 'Lawyer'),
    ('Ethicist', 'Psychologist'),
    ('Ethicist', 'Historian'),
    ('Philosopher', 'Historian'),
    ('Philosopher', 'Psychologist'),

    # ===== Psychology & social reasoning =====
    ('Psychologist', 'Doctor'),
    ('Psychologist', 'Economist'),
    ('Psychologist', 'Lawyer'),
    ('Doctor', 'Psychologist'),

    # ===== Law / economics =====
    ('Economist', 'Lawyer'),
    ('Lawyer', 'Economist'),
    ('Doctor', 'Lawyer'),

    # ===== Science & safety =====
    ('Physicist', 'Engineer'),
    ('Engineer', 'Physicist'),
    ('Engineer', 'Programmer'),
    ('Physicist', 'Doctor'),

    ('Biologist', 'Doctor'),
    ('Biologist', 'Ethicist'),

    ('Chemist', 'Doctor'),
    ('Chemist', 'Engineer'),

    # ===== Technical reasoning =====
    ('Programmer', 'Mathematician'),
    ('Mathematician', 'Programmer'),
    ('Programmer', 'Engineer'),

    # ===== Cross-domain grounding =====
    ('Historian', 'Economist'),
    ('Historian', 'Philosopher'),
    ('Historian', 'Mathematician'),
]

Knowledge = {
    "Knowlegable Expert": {
            "profile": {
                "expertise_domains": [],
                "confidence_level": 0.0
            },
            "working_memory": {
                "current_question": "",
                "key_points": [],
                "uncertainties": []
            },
            "knowledge": {
                "facts": [],
                "concepts": [],
                "cross_domain_links": []
            },
            "reflection": {
                "knowledge_gaps": [],
                "improvement_notes": []
            }
    },
    "Critic" :{
            "profile": {
                "critique_style": "strict/logical"
            },
            "working_memory": {
                "target_argument": "",
                "detected_issues": [],
                "severity_scores": []
            },
            "knowledge": {
                "common_fallacies": [],
                "evaluation_rules": []
            },
            "reflection": {
                "missed_issues": [],
                "false_positives": []
            }
    },
    "Mathematician": {
            "profile": {
                "fields": ["algebra", "calculus", "logic"]
            },
            "working_memory": {
                "problem": "",
                "equations": [],
                "assumptions": []
            },
            "knowledge": {
                "theorems": [],
                "proof_patterns": []
            },
            "reflection": {
                "calculation_errors": [],
                "better_methods": []
            }
    },
    "Psychologist": {
            "profile": {
                "focus": ["cognition", "emotion", "behavior"]
            },
            "working_memory": {
                "observed_behaviors": [],
                "mental_states": [],
                "biases": []
            },
            "knowledge": {
                "psychological_theories": [],
                "patterns": []
            },
            "reflection": {
                "misjudgments": [],
                "new_patterns": []
            }
    },
    "Historian": {
            "profile": {
                "time_periods": []
            },
            "working_memory": {
                "events": [],
                "context": [],
                "causality": []
            },
            "knowledge": {
                "historical_cases": [],
                "comparisons": []
            },
            "reflection": {
                "anachronisms": [],
                "missing_context": []
            }
    },
    "Doctor": {
            "profile": {
                "specialty": []
            },
            "working_memory": {
                "symptoms": [],
                "diagnosis": [],
                "risk_factors": []
            },
            "knowledge": {
                "diseases": [],
                "treatments": []
            },
            "reflection": {
                "misdiagnosis": [],
                "uncertain_cases": []
            }
    },
    "Lawyer": {
            "profile": {
                "jurisdiction": []
            },
            "working_memory": {
                "case_facts": [],
                "legal_issues": [],
                "arguments": []
            },
            "knowledge": {
                "laws": [],
                "precedents": []
            },
            "reflection": {
                "weak_arguments": [],
                "missing_evidence": []
            }
    },
    "Economist": {
            "profile": {
                "models": ["micro", "macro"]
            },
            "working_memory": {
                "variables": [],
                "assumptions": [],
                "predictions": []
            },
            "knowledge": {
                "economic_models": [],
                "data_patterns": []
            },
            "reflection": {
                "model_failures": [],
                "biases": []
            }
    },
    "Programmer": {
            "profile": {
                "languages": []
            },
            "working_memory": {
                "task": "",
                "code_snippets": [],
                "bugs": []
            },
            "knowledge": {
                "algorithms": [],
                "design_patterns": []
            },
            "reflection": {
                "bug_patterns": [],
                "optimization_points": []
            }
    },
    "Philosopher": {
            "profile": {
                "schools": []
            },
            "working_memory": {
                "questions": [],
                "arguments": [],
                "paradoxes": []
            },
            "knowledge": {
                "theories": [],
                "thought_experiments": []
            },
            "reflection": {
                "logical_gaps": [],
                "deeper_questions": []
            }
    },
    "Physicist": {
            "profile": {
                "fields": []
            },
            "working_memory": {
                "models": [],
                "variables": [],
                "assumptions": []
            },
            "knowledge": {
                "laws": [],
                "equations": []
            },
            "reflection": {
                "invalid_models": [],
                "approximation_limits": []
            }
    },
    "Biologist": {
            "profile": {
                "focus": []
            },
            "working_memory": {
                "organisms": [],
                "processes": [],
                "interactions": []
            },
            "knowledge": {
                "biological_systems": [],
                "patterns": []
            },
            "reflection": {
                "misinterpretations": [],
                "new_hypotheses": []
            }
    },
    "Chemist": {
            "profile": {
                "fields": []
            },
            "working_memory": {
                "reactions": [],
                "compounds": [],
                "conditions": []
            },
            "knowledge": {
                "reaction_rules": [],
                "properties": []
            },
            "reflection": {
                "failed_reactions": [],
                "uncertainties": []
            }
    },
    "Engineer": {
            "profile": {
                "domains": []
            },
            "working_memory": {
                "requirements": [],
                "designs": [],
                "constraints": []
            },
            "knowledge": {
                "solutions": [],
                "tradeoffs": []
            },
            "reflection": {
                "design_flaws": [],
                "improvements": []
            }
    },
    "Ethicist": {
            "profile": {
                "frameworks": []
            },
            "working_memory": {
                "dilemmas": [],
                "stakeholders": [],
                "conflicts": []
            },
            "knowledge": {
                "ethical_principles": [],
                "case_studies": []
            },
            "reflection": {
                "biases": [],
                "unresolved_conflicts": []
            }
    },
    "Fake":{
            "profile": {
                "style": "misleading"
            },
            "working_memory": {
                "fake_claims": [],
                "manipulation_strategy": []
            },
            "knowledge": {
                "misinformation_patterns": []
            },
            "reflection": {
                "caught_by_others": [],
                "successful_deceptions": []
            }
        }
}

@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    def get_role_connection(self):
        return ROLE_CONNECTION

    def get_description(self, role):
        return ROLE_DESCRIPTION[role]

    def split_role_label(self):
        roles = [k for k, v in ROLE_LABEL.items() if k not in critical_role]
        descriptions = [v for k, v in ROLE_LABEL.items() if k not in critical_role]
        return roles, descriptions

    def split_critical_role_label(self):
        roles = [k for k, v in ROLE_LABEL.items() if k in critical_role]
        descriptions = [v for k, v in ROLE_LABEL.items() if k in critical_role]
        return roles, descriptions

    def get_conclude_knowledge_format(self):
        return Knowledge

    def conclude_knowledge_format(self,role):
        return Knowledge[role]

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """

    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else "" + """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """

    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot():
        return ""

    def postprocess_answer(self, answer: Union[str, List[Any]]) -> str:
        """
        Normalize model answers into a single option letter (A/B/C/D).
        If no option is found, randomly generate one.
        """

        # --- flatten one-level lists/tuples and pick first non-empty element ---
        if isinstance(answer, (list, tuple)):
            picked = ""
            for item in answer:
                if item is None:
                    continue
                if isinstance(item, (list, tuple)):
                    if len(item) == 0:
                        continue
                    item = item[0]
                s = "" if item is None else str(item)
                if s.strip():
                    picked = s
                    break
            answer = picked

        # ensure string
        if answer is None:
            return random.choice(["A", "B", "C", "D"])

        if not isinstance(answer, str):
            answer = str(answer)

        s = answer.strip()
        if not s:
            return random.choice(["A", "B", "C", "D"])

        # remove box/control tokens
        s = s.replace("<|begin_of_box|>", " ").replace("<|end_of_box|>", " ")

        # remove common prefixes
        s = re.sub(r'(?i)^\s*answer\s*(is|:)?\s*', '', s)
        s = re.sub(r'(?i)^\s*option\s*[:\.]?\s*', '', s)

        # explicit "Option X"
        m = re.search(r'(?i)\boption\b\s*[:\.]?\s*([A-Da-d])\b', s)
        if m:
            return m.group(1).upper()

        # standalone A-D
        m = re.search(r'(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])', s)
        if m:
            return m.group(1).upper()

        # very permissive fallback
        m = re.search(r'([A-Da-d])', s)
        if m:
            return m.group(1).upper()

        # 🚨 FINAL fallback: random guess
        return random.choice(["A", "B", "C", "D"])
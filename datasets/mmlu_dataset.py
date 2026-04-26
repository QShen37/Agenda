import glob
import os
import random
import re

import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"./datasets/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def str_task_to_input(self, task: str) -> Dict[str, Any]:
        return {"task": task}

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

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer

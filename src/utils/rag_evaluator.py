from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
    summarization_score,
)
import pandas as pd
from datasets import Dataset


class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def prepare_evaluation_data(self, df: pd.DataFrame) -> Dataset:
        eval_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truths": []
        }
        
        for _, row in df.iterrows():
            eval_data["question"].append(row['question'])
            eval_data["contexts"].append([row['retrieved_context']])  
            eval_data["answer"].append(row['generated_response'])
            eval_data["ground_truths"].append([row['answer']])  
        
        return Dataset.from_dict(eval_data)

    def evaluate(self, df: pd.DataFrame):
        eval_dataset = self.prepare_evaluation_data(df)
        evaluation_result = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness,
                context_recall,
                context_precision,
                summarization_score,
            ]
        )
        return evaluation_result

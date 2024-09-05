import pandas as pd
from typing import Dict, List, Union, Tuple
from datasets import load_dataset
from tqdm import tqdm

class DatasetLoader:
    def __init__(self):
        print("DatasetLoads initialized")
        
    def overall_datasets_processing(self, datasets: List[str], splits: List[str]):
        processed_datasets = {}
        for dataset_name, split in zip(datasets, splits):
            documents, processed_df = self.overall_dataset_processing(dataset_name, split)
            processed_datasets[dataset_name] = {
                "documents": documents,
                "processed_df": processed_df
            }
        
    def overall_dataset_processing(self, dataset_name: str, split: str = 'train') -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        """
        Process a single dataset and return a list of documents and a dataframe.

        Args:
            dataset_name (str): Name of the dataset to process.
                squad, natural_questions, trivia_qa, hotpot_qa, deepmind/narrativeqa
            split (str): Split of the dataset to process. Defaults to 'train'.

        Returns:
            Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]: A tuple containing a list of documents and a processed dataframe.
        """
        df = self.load_dataset(dataset_name, split)
        documents, processed_df = self.process_dataset(df, dataset_name)
        
        return documents, processed_df

    def load_dataset(self, dataset_name: str, split: str = 'train', **kwargs) -> pd.DataFrame:
        if dataset_name == 'squad':
            dataset = load_dataset(dataset_name, split=split)
        elif dataset_name == 'natural_questions':
            dataset = load_dataset(dataset_name, split=split)
        elif dataset_name == 'trivia_qa':
            dataset = load_dataset(dataset_name, 'unfiltered', split=split)
        elif dataset_name == 'hotpot_qa':
            dataset = load_dataset(dataset_name, 'distractor', split=split)
        elif dataset_name == 'deepmind/narrativeqa':
            dataset = load_dataset(dataset_name, split=split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return pd.DataFrame(dataset)

    def process_dataset(self, df: pd.DataFrame, dataset_name: str) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        if dataset_name == 'squad':
            return self._process_squad(df)
        elif dataset_name == 'natural_questions':
            return self._process_natural_questions(df)
        elif dataset_name == 'trivia_qa':
            return self._process_trivia_qa(df)
        elif dataset_name == 'hotpot_qa':
            return self._process_hotpot_qa(df)
        elif dataset_name == 'deepmind/narrativeqa':
            return self._process_narrativeqa(df)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _process_squad(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        documents = []
        previous_document = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SQuAD"):
            document = {
                "content": row['context'],
                "metadata": {
                    "dataset": "squad",
                    "title": row['title'],
                }
            }
            if document == previous_document:
                continue
            documents.append(document)
            previous_document = document

        df['generated_response'] = ''
        df['retrieved_context'] = ''

        return documents, df

    def _process_natural_questions(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        documents = []
        previous_document = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Natural Questions"):
            # 將 tokens 轉換為文本
            content = ' '.join(row['document']['tokens'])
            
            document = {
                "content": f"Title: {row['document']['title']}\nContent: {content}",
                "metadata": {
                    "dataset_name": "natural_questions",
                    "title": row['document']['title'],
                }
            }
            if document == previous_document:
                continue
            documents.append(document)
            previous_document = document

        df['generated_response'] = ''
        df['retrieved_context'] = ''
        
        return documents, df

    def _process_trivia_qa(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        documents = []
        previous_document = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Trivia QA"):
            # 使用 wiki_context 作為主要內容，如果可用的話
            if 'entity_pages' in row and 'wiki_context' in row['entity_pages']:
                for title, context in zip(row['entity_pages']['title'], row['entity_pages']['wiki_context']):
                    document = {
                        "content": f"Title: {title}\nContent: {context}",
                        "metadata": {
                            "dataset_name": "trivia_qa",
                            "title": title,
                        }
                    }
                    if document == previous_document:
                        continue
                    documents.append(document)
                    previous_document = document
                    
            # 使用 search_results 作為備用或額外的內容
            if 'search_results' in row and 'search_context' in row['search_results']:
                for title, context in zip(row['search_results']['title'], row['search_results']['search_context']):
                    document = {
                        "content": f"Title: {title}\nContent: {context}",
                        "metadata": {
                            "dataset_name": "trivia_qa",
                            "title": title,
                            "type": "search_context"
                        }
                    }
                    if document == previous_document:
                        continue
                    documents.append(document)
                    previous_document = document
                    
        df['generated_response'] = ''
        df['retrieved_context'] = ''
        
        return documents, df

    def _process_hotpot_qa(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        documents = []
        previous_document = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Hot Pot QA"):
            for title, sentences in zip(row['context']['title'], row['context']['sentences']):
                content = f"Title: {title}\nContent: {' '.join(sentences)}"
                
                document = {
                    "content": content.strip(),
                    "metadata": {
                        "dataset_name": "hotpot_qa",
                        # "level": row['level'],
                        # "type": row['type'],
                        "title": title
                    }
                }
                if document == previous_document:
                    continue
                documents.append(document)
                previous_document = document
                
        # 添加新列到原始 DataFrame
        df['generated_response'] = ''
        df['retrieved_context'] = ''
        
        return documents, df
    
    def _process_narrativeqa(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Union[str, Dict]]], pd.DataFrame]:
        documents = []
        previous_document = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Narrative QA"):
            document = {
                "content": f"Title: {row['title']}\nContent: {row['document']}",
                "metadata": {
                    "dataset_name": "narrativeqa",
                    "title": row['title']
                }
            }
            if document == previous_document:
                continue
            documents.append(document)
            previous_document = document
            
        # 添加新列到原始 DataFrame
        df['generated_response'] = ''
        df['retrieved_context'] = ''
        
        return documents, df

if __name__ == "__main__":
    preprocessor = DatasetLoader()
    
    datasets = [ 'trivia_qa', 'natural_questions',  'hotpot_qa',]
    # datasets = ['squad'] #, 'natural_questions', 'trivia_qa', 'hotpot_qa']
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} dataset:")
        df = preprocessor.load_dataset(dataset_name)
        # documents, processed_df = preprocessor.process_dataset(df, dataset_name)
        
        # print(f"Sample document for {dataset_name}:")
        # print(documents[0])
        
        # print(f"\nProcessed DataFrame for {dataset_name}:")
        # print(processed_df.head())
        # print(processed_df.columns)
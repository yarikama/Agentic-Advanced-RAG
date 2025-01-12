from pydantic import Field, BaseModel
from utils import *
from crewai_tools import BaseTool
from typing import List, Dict, Any, Tuple
import statistics
import config.constants as const
import json

class ListAllCollectionsTool(BaseTool):
    name: str = "list_all_collections"
    description: str = """Lists all collection names available in the vector database.
    Arguments:
    - None required
    """
    
    def _run(self) -> List[str]:
        vecDB = VectorDatabase()
        return vecDB.list_collections()

class RetrieveDataTool(BaseTool):
    name: str = "retrieve_data"
    description: str = """Retrieves data for multiple queries from specified collections in the vector database.
    - collection_names: List[str] - Names of collections to search in
    - queries: List[str] - List of query strings
    - top_k: int (optional) - Number of top results to retrieve for each query (default: 5)
    Example input: {"collection_names": ["col1", "col2"], "queries": ["query1", "query2"], "top_k": 3}
    And it will return the top 3 results for each query from the specified collections and aggregate them with no duplicates.
    """
    
    def _run(self, collection_names: List[str], queries: List[str], top_k: int) -> Dict[str, Any]:
        retriever = Retriever()  
        content = []
        metadata = []
        seen_content = set()
        for collection_name, query in zip(collection_names, queries):
            if collection_name and query:
                retrieved_data = retriever.hybrid_retrieve(collection_name, query, top_k)
                for result_list in retrieved_data:
                    for doc in result_list:
                        doc_content = doc.get('content', '')
                        doc_metadata = doc.get('metadata', {})
                        
                        if doc_content not in seen_content:
                            seen_content.add(doc_content)  
                            content.append(doc_content)
                            metadata.append(doc_metadata)
        return {
            "content": content,
            "metadata": metadata
        }

class DenseRetrieveDataTool(BaseTool):
    name: str = "dense_retrieve_data"
    description: str = """Retrieves data for multiple queries from specified collections in the vector database using dense retrieval.
    - collection_names: List[str] - Names of collections to search in
    - queries: List[str] - List of query strings
    - top_k: int (optional) - Number of top results to retrieve for each query (default: 5)
    Example input: {"collection_names": ["col1", "col2"], "queries": ["query1", "query2"], "top_k": 3}
    """
    
    def _run(self, collection_names: List[str], queries: List[str], top_k: List[str]) -> Dict[str, Any]:
        retriever = Retriever()
        content = []
        metadata = []
        seen_content = set()
        for collection_name, query in zip(collection_names, queries):
            if collection_name and query:
                retrieved_data = retriever.dense_retrieve(collection_name, query, top_k)
                for result_list in retrieved_data:
                    for doc in result_list:
                        doc_content = doc.get('content', '')
                        doc_metadata = doc.get('metadata', {})
                        if doc_content not in seen_content:
                            seen_content.add(doc_content)  
                            content.append(doc_content)
                            metadata.append(doc_metadata)
        return {
            "content": content,
            "metadata": metadata
        }

class GlobalRetrieveTopicTool(BaseTool):
    name: str = "global_retrieve_topic"
    description: str = """Retrieves community data for a given query and level using a global retriever.
    - query: str - The query string to search for
    - level: int - The level of community data to retrieve (0-3, where 0 is most general least information and  and 3 is most specific and detailed)
    Example input: ("AI ethics", 1)
    And it will return the community data for the query "AI ethics" at level 1.
    """

    def _run(self, query: str, level: int) -> str:
        retriever = Retriever()  
        return retriever.global_retrieve(query, level)
    
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = """Performs basic arithmetic calculations.
    Arguments:
    - expression: str - A mathematical expression as a string
    Example input: "2 + 3 * 4"
    """
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except (SyntaxError, NameError, TypeError, ZeroDivisionError):
            return "Error: Invalid expression"

class BasicStatisticsTool(BaseTool):
    name: str = "basic_statistics"
    description: str = """Calculates basic statistical measures for a given set of numbers.
    Arguments:
    - numbers: str - A comma-separated string of numbers
    Example input: "1,2,3,4,5"
    """
    
    def _run(self, numbers: str) -> str:
        try:
            num_list = [float(num.strip()) for num in numbers.split(',')]
            if not num_list:
                return "Error: Empty list"
            
            stats = {
                'mean': statistics.mean(num_list),
                'median': statistics.median(num_list),
                'mode': statistics.multimode(num_list),
                'std_dev': statistics.stdev(num_list) if len(num_list) > 1 else 0,
                'min': min(num_list),
                'max': max(num_list)
            }
            return "\n".join(f"{k}: {v}" for k, v in stats.items())
        except statistics.StatisticsError:
            return "Error: Invalid data for statistical calculation"
        except ValueError:
            return "Error: Invalid input. Please provide comma-separated numbers."

class RerankTool(BaseTool):
    name: str = "rerank"
    description: str = """Reranks the retrieved data based on relevance scores.
    - metadata: List[Dict[str, Any]] - List of metadata dictionaries
    - content: List[str] - List of content strings
    - relevance_scores: List[float] - List of relevance scores
    Example input: {"metadata": [{"id": 1}, {"id": 2}], "content": ["text1", "text2"], "relevance_scores": [0.9, 0.7]}
    """
    def _run(self, metadata: List[Dict[str, Any]], content: List[str], relevance_scores: List[float]) -> str:
        ranked_scores = sorted([(i, score) for i, score in enumerate(relevance_scores) if score > 0], key=lambda x: x[1], reverse=True)
        ranked_content = [content[i] for i, _ in ranked_scores]
        ranked_metadata = [metadata[i] for i, _ in ranked_scores]
        ranked_scores = [score for _, score in ranked_scores]

        return json.dumps({
            "ranked_content": ranked_content,
            "ranked_metadata": ranked_metadata,
            "ranked_scores": ranked_scores
        })

class InsertQAIntoDBTool(BaseTool):
    name: str = "insert_qa_into_db"
    description: str = """Inserts a question-answer pair into the specified collection in the vector database.
    - collection_name: str - Name of the collection to insert into
    - question: str - The question to be inserted
    - answer: str - The answer corresponding to the question
    Example input: {"collection_name": "my_collection", "question": "What is AI?", "answer": "Artificial Intelligence is..."}
    And it will return "Successfully inserted Q&A pair into collection 'my_collection'."
    """
    
    def _run(self, collection_name: str, question: str, answer: str) -> str:
        data_processor = DataProcessor()  # 假设这是获取 DataProcessor 的方式
        qa_content = f"question: {question} answer: {answer}"
        qa_metadata = {"type": "QA", "source": "user and agent"}
        qa_document = [{"content": qa_content, "metadata": qa_metadata}]
        data_processor.document_process(collection_name, qa_document, False, False)
        
        return f"Successfully inserted Q&A pair into collection '{collection_name}'."
class Tools:
    def __init__(self):
        self.tools_map = {
            "list_all_collections": ListAllCollectionsTool,
            "retrieve_data": RetrieveDataTool,
            "dense_retrieve_data": DenseRetrieveDataTool,
            "global_retrieve_topic": GlobalRetrieveTopicTool,
            "calculator": CalculatorTool,
            "basic_statistics": BasicStatisticsTool,
            "rerank": RerankTool,
            "insert_qa_into_db": InsertQAIntoDBTool,
        }

    def get_tools(self, **kwargs):
        """
        options:
            "list_all_collections": ListAllCollectionsTool,
            "retrieve_data": RetrieveDataTool,
            "dense_retrieve_data": DenseRetrieveDataTool,
            "global_retrieve_topic": GlobalRetrieveTopicTool,
            "calculator": CalculatorTool,
            "basic_statistics": BasicStatisticsTool,
            "rerank": RerankTool,
            "insert_qa_into_db": InsertQAIntoDBTool,
        """
        tools = []
        for tool_name, is_result in kwargs.items():
            if tool_name not in self.tools_map:
                raise ValueError(f"Tool '{tool_name}' does not exist.")
            tool_class = self.tools_map[tool_name]
            if is_result:
                tools.append(tool_class(result_as_answer=True))
            else:
                tools.append(tool_class())  
        return tools
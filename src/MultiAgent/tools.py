import statistics
from Utils import *
import Config.constants as const
from langchain.tools import StructuredTool
from typing import List, Dict, Any, Callable, Tuple
from langchain.pydantic_v1 import Field, create_model

def create_tool(callable: Callable):
    method = callable
    args = {k:v for k,v in method.__annotations__.items() if k not in ["self", "return"]}
    name = method.__name__
    doc = method.__doc__
    func_desc = doc[doc.find("<desc>") + len("<desc>"):doc.find("</desc>")]
    arg_desc = dict()
    for arg in args.keys():
        desc = doc[doc.find(f"{arg}: ")+len(f"{arg}: "):]
        desc = desc[:desc.find("\n")]
        arg_desc[arg] = desc
    arg_fields = dict()
    for k,v in args.items():
        arg_fields[k] = (v, Field(description=arg_desc[k]))

    Model = create_model('Model', **arg_fields)

    tool = StructuredTool.from_function(
        func=method,
        name=name,
        description=func_desc,
        args_schema=Model,
        return_direct=False,
    )
    return tool

class Tools:
    def __init__(self, vectordatabase: VectorDatabase, embedder: Embedder, retriever: Retriever, data_processor: DataProcessor):
        # Utils
        self.vectordatabase = vectordatabase 
        self.embedder = embedder
        self.retriever = retriever
        self.data_processor = data_processor
        # Tools
        self.create_list_all_collections_tool = create_tool(self._list_all_collections)
        self.create_retrieve_data_tool = create_tool(self._retrieve_data)
        self.create_dense_retrieve_data_tool = create_tool(self._dense_retrieve_data)
        self.create_ranker_tool = create_tool(self._rerank)
        self.create_calculator_tool = create_tool(self._calculator)
        self.create_basic_statistics_tool = create_tool(self._basic_statistics)
        self.create_insert_qa_into_db_tool = create_tool(self._insert_qa_into_db)
        print("Tools initialized")
    
    # Getters for all tools
    def get_retriever_toolkit(self):
        return [
            self.create_list_all_collections_tool,
            self.create_retrieve_data_tool,
        ]
        
    def get_generator_toolkit(self):
        return [
            self.create_calculator_tool,
            self.create_basic_statistics_tool,
        ]

    def get_database_updater_toolkit_wo_collection(self):
        return [
            self.create_list_all_collections_tool,
            self.create_dense_retrieve_data_tool,
            self.create_insert_qa_into_db_tool,
        ]
        
    def get_database_updater_toolkit_w_collection(self):
        return [
            self.create_dense_retrieve_data_tool,
            self.create_insert_qa_into_db_tool,
        ]
    
    def get_classifiction_toolkit_wo_collection(self):
        return [
            self.create_list_all_collections_tool,
        ]
        
    def get_classifiction_toolkit_w_collection(self):
        return []
        
        
    def get_retrieve_toolkit(self):
        return [
            self.create_retrieve_data_tool,
        ]
        
    def get_reranker_toolkit(self):
        return [
            self.create_ranker_tool
        ]
    
    # Tools definitions
    def _list_all_collections(self) -> List[str]:
        """
        <desc>
        Lists all collection names available in the vector database.
        
        Use case: When you need to know what collections are available in the database for searching or inserting data.
        
        Usage: Simply call this function without any arguments.
        
        Example:
        collections = list_all_collections()
        print("Available collections:", collections)
        
        Returns an empty list if no collections are found.
        </desc>
        """
        return self.vectordatabase.list_collections()
    
    def _retrieve_data(self, collection_names: List[str], queries: List[str], top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        """
        <desc>Retrieves data for multiple queries from specified collections in the vector database</desc>
        
        collection_names: A list of collection names to search in
        queries: A list of query strings to search for
        top_k: The number of top results to retrieve for each query, suggested value: 5
        """
        content = []
        metadata = []
        seen_content = set() 
        for collection_name, query in zip(collection_names, queries):
            if collection_name and query:
                retrieved_data = self.retriever.hybrid_retrieve(collection_name, query, top_k)
                for result_list in retrieved_data:
                    for doc in result_list:
                        doc_content = content.append(doc.get('content', ''))
                        doc_metadata = metadata.append(doc.get('metadata', {}))
                        
                        if doc_content not in seen_content:
                            seen_content.add(doc_content)  
                            content.append(doc_content)
                            metadata.append(doc_metadata)
                # results.append({
                #     "query": query,
                #     "collection_name": collection_name,
                #     "retrieved_data": retrieved_data
                # })
        return {
            "content": content,
            "metadata": metadata
        }
        
    def _dense_retrieve_data(self, collection_names: List[str], queries: List[str], top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        """
        <desc>Retrieves data for multiple queries from specified collections in the vector database</desc>
        
        collection_names: A list of collection names to search in
        queries: A list of query strings to search for
        top_k: The number of top results to retrieve for each query, suggested value: 5
        """
        content = []
        metadata = []
        seen_content = set() 
        for collection_name, query in zip(collection_names, queries):
            if collection_name and query:
                retrieved_data = self.retriever.dense_retrieve(collection_name, query, top_k)
                for result_list in retrieved_data:
                    for doc in result_list:
                        doc_content = content.append(doc.get('content', ''))
                        doc_metadata = metadata.append(doc.get('metadata', {}))
                        
                        if doc_content not in seen_content:
                            seen_content.add(doc_content)  
                            content.append(doc_content)
                            metadata.append(doc_metadata)
                # results.append({
                #     "query": query,
                #     "collection_name": collection_name,
                #     "retrieved_data": retrieved_data
                # })
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _calculator(self, expression: str) -> str:
        """
        <desc>Performs basic arithmetic calculations</desc>
        expression: A mathematical expression as a string (e.g., "2 + 3 * 4")
        """
        try:
            result = eval(expression)
            return str(result)
        except (SyntaxError, NameError, TypeError, ZeroDivisionError):
            return "Error: Invalid expression"
        
    def _basic_statistics(self, numbers: str) -> str:
        """
        <desc>Calculates basic statistical measures for a given set of numbers</desc>
        numbers: A comma-separated string of numbers (e.g., "1,2,3,4,5")
        """
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
        
    def _rerank(self, metadata: List[Dict[str, Any]], content: List[str], relevance_scores: List[float]) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Reranks the retrieved data based on relevance scores.

        Args:
            metadata (List[Dict[str, Any]]): A list of dictionaries containing metadata for each retrieved document.
            content (List[str]): A list of content strings corresponding to the metadata.
            relevance_scores (List[float]): A list of relevance scores for each retrieved document.

        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[float]]: A tuple containing:
                - ranked_data (List[str]): The content strings reordered based on relevance scores.
                - ranked_metadata (List[Dict[str, Any]]): The metadata dictionaries reordered based on relevance scores.
                - ranked_scores (List[float]): The relevance scores reordered in descending order.
        """
        ranked_scores = sorted(enumerate(relevance_scores), key=lambda x: x[1], reverse=True)
        ranked_data = [content[i] for i, _ in ranked_scores]
        ranked_metadata = [metadata[i] for i, _ in ranked_scores]
        return ranked_data, ranked_metadata, [score for _, score in ranked_scores]

    def _insert_qa_into_db(self, collection_name: str, question: str, answer: str) -> str:
        """
        <desc>Inserts a question-answer pair into the specified collection in the vector database</desc>
        collection_name: The name of the collection to insert into
        question: The question to be inserted
        answer: The answer corresponding to the question
        """
        qa_cotent = "question: " + question + " answer: " + answer
        qa_metadata = {"type": "QA", "source": "user and agent"}
        qa_document = [{"content": qa_cotent, "metadata": qa_metadata}]
        self.data_processor.document_process(collection_name, qa_document, False, False)
        
        return f"Successfully inserted Q&A pair into collection '{collection_name}'."
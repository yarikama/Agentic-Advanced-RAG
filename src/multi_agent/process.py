from dotenv import load_dotenv
from crewai import Crew, Process

from utils import *
import config.constants as const

from .tools import Tools
from .tasks import Tasks
from .agents import Agents

from config.output_pydantic import TopicRerankingResult, RerankingResult
from config.rag_config import RAGConfig
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
class MultiAgent_RAG:
    def __init__(self, rag_config: RAGConfig = RAGConfig()):
        """
        Initialize the MultiAgent_RAG system.

        Args:
            rag_config (RAGConfig): Configuration for the RAG system.
        """
        # LLM Settings
        self.model_name = rag_config.model_name if rag_config.model_name else const.MODEL_NAME
        self.model_temperature = rag_config.model_temperature if rag_config.model_temperature else const.MODEL_TEMPERATURE  
        self.user_query = rag_config.user_query if rag_config.user_query else None
        self.specific_collection = rag_config.specific_collection if rag_config.specific_collection else None
        # Callback
        # self.callback_function = rag_config.callback_function if rag_config.callback_function else None
        
        # Tools, Agents, Tasks
        self.tools = Tools()
        self.agents = Agents(self.model_temperature, self.model_name, self.tools)
        self.tasks = Tasks(self.agents, self.tools)
        print("MultiAgent RAG System initialized")
        # self.agents = Agents(self.model_temperature, self.model_name, self.tools, self.callback_function)

                
    def run_crew(self, **kwargs):
        """
        Run a Crew task.

        Args:
            **kwargs: May include node_agents, node_tasks, node_process, node_inputs, etc.

        Returns:
            The result of the task execution.
        """
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "Hierarchical" else Process.sequential,
            verbose=const.CREWAI_PROCESS_VERBOSE,
        )
        node_inputs = kwargs.get("node_inputs")
        return self.crew.kickoff(inputs=node_inputs) if node_inputs else self.crew.kickoff()
        
    def run_crew_batch(self, **kwargs):
        """
        Run Crew tasks in batches.

        Args:
            **kwargs: May include node_agents, node_tasks, node_process, node_batch_inputs, etc.

        Returns:
            The results of the batch task execution.
        """
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "Hierarchical" else Process.sequential,
            verbose=const.CREWAI_AGENT_VERBOSE,
        )
        node_batch_inputs = kwargs.get("node_batch_inputs")
        return self.crew.kickoff_for_each(inputs=node_batch_inputs) if node_batch_inputs else self.crew.kickoff()
    
    def run_crew_batch_async(self, **kwargs):
        """
        Run Crew tasks asynchronously in batches.

        Args:
            **kwargs: May include node_agents, node_tasks, node_process, node_batch_inputs, etc.

        Returns:
            The results of the asynchronous batch task execution.
        """
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "Hierarchical" else Process.sequential,
            verbose=const.CREWAI_AGENT_VERBOSE,
        )
        node_batch_inputs = kwargs.get("node_batch_inputs")

        async def run_async():
            if node_batch_inputs:
                return await self.crew.kickoff_for_each_async(inputs=node_batch_inputs)
            else:
                return await self.crew.kickoff_async()

        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(run_async())

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
        
    def user_query_classification_run(self, **kwargs):
        """
        Run the user query classification task.

        Args:
            **kwargs: Must include 'user_query' (str).

        Returns:
            UserQueryClassificationResult: A Pydantic model containing:
                - needs_retrieval (bool): Indicates if retrieval is needed.
                - justification (str): Explanation for the classification.
        """
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["User Query Classification"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return self.tasks.create_user_query_classification_task.output.pydantic
    
    def plan_coordination_run(self, **kwargs):
        """
        Run the plan coordination task.

        Args:
            **kwargs: Must include 'user_query' (str).

        Returns:
            PlanCoordinationResult: A Pydantic model containing:
                - plan (str): The coordinated plan.
        """
        self.run_crew(   
            node_agents=["Plan Coordinator"],
            node_tasks=["Plan Coordination"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return self.tasks.create_plan_coordination_task.output.pydantic
        
    def query_process_run(self, **kwargs):
        """
        Run the query processing task.

        Args:
            **kwargs: Must include 'user_query' (str).

        Returns:
            QueriesProcessResult: A Pydantic model containing:
                - original_query (str): The original user query.
                - transformed_queries (Optional[List[str]]): List of transformed queries, if any.
                - decomposed_queries (Optional[List[str]]): List of decomposed queries, if any.
        """
        self.run_crew(   
            node_agents=["Query Processor"],
            node_tasks=["Query Process"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return self.tasks.create_query_process_task.output.pydantic
    
    def sub_queries_classification_with_specification_run(self, **kwargs):
        """
        Run the sub-queries classification task with a specific collection.

        Args:
            **kwargs: Must include 'specific_collection' (str).

        Returns:
            SubQueriesClassificationResult: A Pydantic model containing:
                - queries (List[str]): List of classified sub-queries.
                - collection_name (List[Optional[str]]): Corresponding collection names for each query.
        """
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["Sub Queries Classification w/ sc"],
            node_process="Sequential",
            node_inputs={"specific_collection": kwargs.get("specific_collection")}
        )
        return self.tasks.create_sub_queries_classification_task_with_specific_collection.output.pydantic
        
    def sub_queries_classification_without_specification_run(self, **kwargs):
        """
        Run the sub-queries classification task without a specific collection.
        
        Args:
            no args

        Returns:
            SubQueriesClassificationResult: A Pydantic model containing:
                - queries (List[str]): List of classified sub-queries.
                - collection_name (List[Optional[str]]): Corresponding collection names for each query.
        """
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["Sub Queries Classification w/o sc"],
            node_process="Sequential",
        )
        return self.tasks.create_sub_queries_classification_task_without_specific_collection.output.pydantic
        
            
    # def global_topic_reranking_run_batch_async(self, **kwargs):
    #     """
    #     Run the global topic reranking task asynchronously in batches.

    #     Args:
    #         **kwargs: Must include 'node_batch_inputs' (List[Dict]).
    #         node_batch_inputs: List of Dicts containing:
    #             user_query (str): The user query.
    #             sub_queries (Optional[List[str]]): The sub-queries.
    #             batch_communities (List[str]): The batch communities.
    #             batch_size (int): The batch size.

    #     Returns:
    #         TopicRerankingResult: A Pydantic model containing:
    #             - relevant_scores (List[int]): List of relevance scores for topics.
    #     """
    #     results = self.run_crew_batch_async(   
    #         node_agents=["Reranker"],
    #         node_tasks=["Global Topic Reranking"],
    #         node_process="Sequential",
    #         node_batch_inputs=kwargs.get("node_batch_inputs")
    #     )
    #     all_scores = []
    #     for result in results:
    #         batch_scores = result.pydantic.relevant_scores
    #         all_scores.extend(batch_scores)
    #     return TopicRerankingResult(relevant_scores=all_scores)
    
    def global_mapping_run(self, **kwargs):
        """
        Run the global mapping task.

        Args:
            **kwargs: Must include 'node_batch_inputs' (List[Dict]).
            node_batch_inputs: List of Dicts containing:
                user_query (str): The user query.
                batch_communities (List[str]): The batch communities.
        """
        self.run_crew(   
            node_agents=["Synthesizer"],
            node_tasks=["Global Mapping"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query"), 
                        "batch_data": kwargs.get("batch_data"),
                        }
        )
        return self.tasks.create_global_mapping_task.output.pydantic
        
        
    def global_topic_searching_run(self, **kwargs):
        """
        Run the global topic searching task.

        Args:
            **kwargs: Must include: 
                'user_query' (str)
                'sub_queries' (List[str])
                'data' (List[str])
        Returns:
            GlobalTopicSearchingResult: A Pydantic model containing:
                - communities_summaries (List[str]): Summaries of relevant communities.
                - possible_answers (List[str]): Potential answers based on the topics.
        """
        self.run_crew(   
            node_agents=["Topic Searcher"],
            node_tasks=["Global Topic Searching"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query"), 
                        "sub_queries": kwargs.get("sub_queries"),
                        "data": kwargs.get("data"),
                        }
        )
        return self.tasks.create_global_topic_searching_task.output.pydantic
        
    def local_topic_searching_run(self, **kwargs):
        """
        Run the local topic searching task.

        Args:
            **kwargs: Must include: 
                'user_query' (str)
                'sub_queries' (List[str])
                'data' (List[str])
        """
        self.run_crew(   
            node_agents=["Topic Searcher"],
            node_tasks=["Local Topic Searching"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query"), 
                        "sub_queries": kwargs.get("sub_queries"),
                        "data": kwargs.get("data"),
                        }
        )
        return self.tasks.create_local_topic_searching_task.output.pydantic
        
        
    # def retrieval_detail_data_from_topic_run(self, **kwargs):
    #     self.run_crew(
    #         node_agents=["Retriever"],
    #         node_tasks=["Retrieval Detail Data From Topic"],
    #         node_process="Sequential",
    #         node_inputs={""}123123123123123
    #     )
    #     return self.tasks.create_retrieval_detail_data_from_topic_task.output.pydantic
        
    def reranking_run_batch_async(self, **kwargs):
        """
        Run the reranking task asynchronously in batches.

        Args:
            **kwargs: Must include 'node_batch_inputs' (List[Dict]).

        Returns:
            RerankingResult: A Pydantic model containing:
                - relevance_scores (List[int]): List of relevance scores for reranked items.
        """
        results = self.run_crew_batch_async(   
            node_agents=["Reranker"],
            node_tasks=["Reranking"],
            node_process="Sequential",
            node_batch_inputs=kwargs.get("node_batch_inputs")
        )
        all_scores = []
        for result in results:
            all_scores.extend(result.pydantic.relevance_scores)
        return RerankingResult(relevance_scores=all_scores)
    
    
    def information_organization_run(self, **kwargs):
        """
        Run the information organization task.
        
        Args:
            **kwargs: Must include 'user_query' (str), 'retrieved_data' (Dict), Optional 'sub_queries' (List[str]).
            
        Returns:
            str: The organized information.
        """
        self.run_crew(   
            node_agents=["Information Organizer"],
            node_tasks=["Information Organization"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query"), 
                         "retrieved_data": kwargs.get("retrieved_data"),
                         "sub_queries": kwargs.get("sub_queries")}
        )
        return self.tasks.create_information_organization_task.output.pydantic
    
    def generation_run(self, **kwargs):
        """
        Run the generation task.

        Args:
            **kwargs: Must include 'user_query' (str), Optional 'sub_queries' (List[str]).

        Returns:
            Dict: A dictionary containing:
                - generation_result (str): The generated response.
        """
        self.run_crew(   
            node_agents=["Generator"],
            node_tasks=["Generation"],
            node_process="Sequential",
            node_inputs={"user_query": kwargs.get("user_query"),
                         "sub_queries": kwargs.get("sub_queries"),
                         "information": kwargs.get("information"),
                         "retrieval_needed": kwargs.get("retrieval_needed")
                         } 
        )
        return self.tasks.create_generation_task.output.raw
    

        
        
        
    
    
    # def retrieval_and_generation_run(self):
    #     # Crew with process
    #     self.crew = Crew(  
    #         agents=self.agents.get_retrieval_and_generation_node_agent(),
    #         tasks=self.tasks.get_retrieval_and_generation_node_tasks(),
    #         process=Process.sequential,
    #         verbose=const.CREWAI_AGENT_VERBOSE,
    #         output_log_file="logs.txt",
    #     )
    #     self.crew.kickoff()
    #     return {
    #         "queries": self.tasks.create_query_processor_task.output.pydantic,
    #         "queries_identification": self.tasks.create_classification_task.output.pydantic,
    #         "refined_retrieval_data": self.tasks.create_retrieval_task.output.pydantic,
    #         "ranked_retrieval_data": self.tasks.create_rerank_task.output.pydantic,
    #         "result": self.tasks.create_summarizer_task.output,
    #         "audit_result": self.tasks.create_response_auditor_task.output.pydantic,
    #     }
    
    def database_update_run(self):
        """
        Run the database update task.

        Returns:
            Dict: A dictionary containing:
                - update_condition (DatabaseUpdateResult): A Pydantic model with:
                    - success (bool): Indicates if the update was successful.
                    - reason (str): Explanation for the update result.
        """
        # Any update Here
        #
        
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_database_update_node_agent(),
            tasks=self.tasks.get_database_update_node_task(),
            process=Process.sequential,
            verbose=const.CREWAI_AGENT_VERBOSE,
            output_log_file="logs.txt",
        )
        self.crew.kickoff()
        return {
            "update_condition": self.tasks.create_database_updater_task.output.pydantic
        }
    
    def overall_run(self, mode=str, new_query=None, new_collection=None):
        """
        Run the overall RAG process.

        Args:
            mode (str): The execution mode ('sequential' or 'hierarchical').
            new_query (Optional[str]): A new user query to process.
            new_collection (Optional[str]): A new collection to use.

        Returns:
            Dict: A dictionary containing multiple Pydantic models:
                - user_query_classification (UserQueryClassificationResult)
                - queries (QueriesProcessResult)
                - queries_identification_list (SubQueriesClassificationResult)
                - refined_retrieval_data (RetrievalResult)
                - ranked_retrieval_data (RerankingResult)
                - result (str): The final generated result.
                - audit_result (ResponseAuditResult)
                - update_condition (DatabaseUpdateResult)
        """
        # Update the task with new query and collection
        self.tasks.update_task(new_query, new_collection)
        
        # Crew with process
        self.crew = Crew(  
            manager_agent=self.agents.create_plan_coordinator if mode != "Sequential" else None,
            agents=self.agents.get_sequential_agents() if mode == "sequential" else self.agents.get_hierarchical_agents(),
            tasks=self.tasks.get_sequential_tasks() if mode == "sequential" else self.tasks.get_hierarchical_tasks(),
            process=Process.sequential if mode == "sequential" else Process.hierarchical,
            output_log_file="logs.txt",
            verbose=const.CREWAI_AGENT_VERBOSE,
        )        
        self.crew.kickoff()        
        return {
            "user_query_classification": self.tasks.create_user_query_classification_task.output.pydantic,
            "queries": self.tasks.create_query_processor_task.output.pydantic,
            "queries_identification_list": self.tasks.create_classification_task.output.pydantic,
            "refined_retrieval_data": self.tasks.create_retrieval_task.output.pydantic,
            "ranked_retrieval_data": self.tasks.create_rerank_task.output.pydantic,
            "result": self.tasks.create_summarizer_task.output,
            "audit_result": self.tasks.create_response_auditor_task.output.pydantic,
            "update_condition": self.tasks.create_database_updater_task.output
        }
        
if __name__ == "__main__":
    pass
from dotenv import load_dotenv
from crewai import Crew, Process

from Utils import *
import Config.constants as const

from .tools import Tools
from .tasks import Tasks
from .agents import Agents

from Config.rag_config import RAGConfig

import asyncio

load_dotenv()

class MultiAgent_RAG:
    def __init__(self, rag_config: RAGConfig = RAGConfig()):
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
        print("LLMMA RAG System initialized")
        
        # self.agents = Agents(self.model_temperature, self.model_name, self.tools, self.callback_function)

                
    def run_crew(self, **kwargs):
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "hierarchical" else Process.sequential,
            verbose=True,
        )
        node_inputs = kwargs.get("node_inputs")
        return self.crew.kickoff(inputs=node_inputs) if node_inputs else self.crew.kickoff()
        
    def run_crew_batch(self, **kwargs):
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "hierarchical" else Process.sequential,
            verbose=True,
        )
        node_batch_inputs = kwargs.get("node_batch_inputs")
        return self.crew.kickoff_for_each(inputs=node_batch_inputs) if node_batch_inputs else self.crew.kickoff()
    
    async def run_crew_batch_async(self, **kwargs):
        self.crew = Crew(
            agents=self.agents.get_agents(*kwargs.get("node_agents", [])),
            tasks=self.tasks.get_tasks(*kwargs.get("node_tasks", [])),
            process=Process.hierarchical if kwargs.get("node_process") == "hierarchical" else Process.sequential,
            verbose=True,
        )
        node_batch_inputs = kwargs.get("node_batch_inputs")
        return await self.crew.kickoff_for_each_async(inputs=node_batch_inputs) if node_batch_inputs else await self.crew.kickoff_async()
        
    def user_query_classification_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["User Query Classification"],
            node_process="sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return {
            "user_query_classification_result": self.tasks.create_user_query_classification_task.output.pydantic
        }
    
    def plan_coordination_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Plan Coordinator"],
            node_tasks=["Plan Coordination"],
            node_process="sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return {
            "plan_coordination_result": self.tasks.create_plan_coordination_task.output.pydantic
        }
        
    def query_process_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Query Processor"],
            node_tasks=["Query Process"],
            node_process="sequential",
            node_inputs={"user_query": kwargs.get("user_query")}
        )
        return {
            "queries_process_result": self.tasks.create_query_process_task.output.pydantic
        }
    
    def sub_queries_classification_with_specification_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["Sub Queries Classification w/ sc"],
            node_process="sequential",
            node_inputs={"specific_collection": kwargs.get("specific_collection")}
        )
        return {
            "sub_queries_classification_result": self.tasks.create_sub_queries_classification_task_with_specific_collection.output.pydantic
        }
        
    def sub_queries_classification_without_specification_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Classifier"],
            node_tasks=["Sub Queries Classification w/o sc"],
            node_process="sequential",
        )
        return {
            "sub_queries_classification_result": self.tasks.create_sub_queries_classification_task_without_specific_collection.output.pydantic
        }
    
    def topic_searching_run(self, **kwargs):
        self.run_crew(   
            node_agents=["Topic Searcher"],
            node_tasks=["Topic Searching"],
            node_process="sequential",
            node_inputs={"user_query": kwargs.get("user_query")} #123
        )
        return {
            "topic_search_result": self.tasks.create_topic_searching_task.output.pydantic
        }
        
    def topic_reranking_run_batch(self, **kwargs):
        return self.run_crew_batch(   
            node_agents=["Reranker"],
            node_tasks=["Topic Reranking"],
            node_process="sequential",
            node_batch_inputs=kwargs.get("node_batch_inputs")
        )
        
    async def topic_reranking_run_batch_async(self, **kwargs):
        result = await self.run_crew_batch_async(   
            node_agents=["Reranker"],
            node_tasks=["Topic Reranking"],
            node_process="sequential",
            node_batch_inputs=kwargs.get("node_batch_inputs")
        )
        return result
    
    
    
        
        
        
    
    
    # def retrieval_and_generation_run(self):
    #     # Crew with process
    #     self.crew = Crew(  
    #         agents=self.agents.get_retrieval_and_generation_node_agent(),
    #         tasks=self.tasks.get_retrieval_and_generation_node_tasks(),
    #         process=Process.sequential,
    #         verbose=True,
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
    
    def generation_run(self):
        # Any update Here
        
        
        # Specific node agents and tasks
        node_agents = ["Generator", "Summarizer", "Response Auditor"]
        node_tasks = ["Generation", "Summarization", "Response Audit"]
        
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_agents(*node_agents),
            tasks=self.tasks.get_tasks(*node_tasks),
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.txt",
        )
        self.crew.kickoff()
        return {
            "result": self.tasks.create_summarization_task.output.pydantic,
            "audit_result": self.tasks.create_response_audit_task.output.pydantic,
        }
    
    def database_update_run(self):
        # Any update Here
        #
        
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_database_update_node_agent(),
            tasks=self.tasks.get_database_update_node_task(),
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.txt",
        )
        self.crew.kickoff()
        return {
            "update_condition": self.tasks.create_database_updater_task.output.pydantic
        }
    
    def overall_run(self, mode=str, new_query=None, new_collection=None):
        # Update the task with new query and collection
        self.tasks.update_task(new_query, new_collection)
        
        # Crew with process
        self.crew = Crew(  
            manager_agent=self.agents.create_plan_coordinator if mode != "Sequential" else None,
            agents=self.agents.get_sequential_agents() if mode == "sequential" else self.agents.get_hierarchical_agents(),
            tasks=self.tasks.get_sequential_tasks() if mode == "sequential" else self.tasks.get_hierarchical_tasks(),
            process=Process.sequential if mode == "sequential" else Process.hierarchical,
            output_log_file="logs.txt",
            verbose=True,
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
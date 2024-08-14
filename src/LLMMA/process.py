from dotenv import load_dotenv
from crewai import Crew, Process

from Utils import *
import Utils.constants as const


from .tools import Tools
from .tasks import Tasks
from .agents import Agents

load_dotenv()

class LLMMA_RAG_System:
    def __init__(self, vectordatabase: VectorDatabase = None, embedder: Embedder = None):
        # Utils
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.embedder = embedder if embedder else Embedder()
        self.retriever = Retriever(self.vectordatabase, self.embedder)
        self.data_processor = DataProcessor(self.vectordatabase, self.embedder)
                
        # Tools, Agents, Tasks
        self.tools = Tools(self.vectordatabase, self.embedder, self.retriever, self.data_processor)
        self.agents = Agents(const.MODEL_TEMPERATURE, const.MODEL_NAME, self.tools)
        self.tasks = Tasks(self.agents, self.tools)
        
        print("LLMMA RAG System initialized")

    def user_query_classification_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_user_query_classification_node_agent(),
            tasks=self.tasks.get_user_query_classification_node_task(),
            process=Process.sequential,
            verbose=True,
        )
        return {
            "user_query_classification": self.crew.kickoff().pydantic
        }
    
    def retrieval_and_generation_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_retrieval_and_generation_node_agent(),
            tasks=self.tasks.get_retrieval_and_generation_node_tasks(),
            process=Process.sequential,
            verbose=True,
        )
        self.crew.kickoff()
        return {
            "queries": self.tasks.create_query_processor_task.output.pydantic,
            "queries_identification_list": self.tasks.create_classification_task.output.pydantic,
            "refined_retrieval_data": self.tasks.create_retrieval_task.output.pydantic,
            "ranked_retrieval_data": self.tasks.create_rerank_task.output.pydantic,
            "result": self.tasks.create_summarizer_task.output,
            "audit_result": self.tasks.create_response_auditor_task.output.pydantic,
        }
    
    def generation_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_generation_node_agent(),
            tasks=self.tasks.get_generation_node_tasks(),
            process=Process.sequential,
            verbose=True,
        )
        self.crew.kickoff()
        return {
            "queries": None,
            "queries_identification_list": None,
            "refined_retrieval_data": None,
            "ranked_retrieval_data": None,
            "result": self.tasks.create_summarizer_task.output,
            "audit_result": self.tasks.create_response_auditor_task.output.pydantic,
        }
    
    def database_update_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_database_update_node_agent(),
            tasks=self.tasks.get_database_update_node_task(),
            process=Process.sequential,
            verbose=True,
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
            agents=self.agents.get_sequential_agents() if mode == "sequential" else self.agents.get_hierarchical_agents(),
            tasks=self.tasks.get_sequential_tasks() if mode == "sequential" else self.tasks.get_hierarchical_tasks(),
            process=Process.sequential if mode == "sequential" else Process.hierarchical,
            output_log_file=True,
            verbose=True,
        )        
        self.crew.kickoff()        
        result = self.tasks.create_summarizer_task.output
        context = self.tasks.create_retrieval_task.output.pydantic
        return result, context
        
if __name__ == "__main__":
    pass
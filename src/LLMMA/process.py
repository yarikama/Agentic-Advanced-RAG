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
            agents=self.agents.get_classification_node_agent(),
            tasks=self.tasks.get_user_query_classification_task(),
            process=Process.sequential,
            verbose=True,
        )
        return self.crew.kickoff().pydantic
    
    def retrieval_and_generation_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_retrieval_and_generation_node_agent(),
            tasks=self.tasks.retrieval_and_generation_tasks(),
            process=Process.sequential,
            verbose=True,
        )
        return self.crew.kickoff().pydantic, self.tasks.create_retrieval_task.output.pydantic
    
    def generation_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_generation_node_agent(),
            tasks=self.tasks.generation_tasks(),
            process=Process.sequential,
            verbose=True,
        )
        return self.crew.kickoff().pydantic, None
    
    def database_update_run(self):
        # Crew with process
        self.crew = Crew(  
            agents=self.agents.get_update_node_agent(),
            tasks=self.tasks.update_tasks(),
            process=Process.sequential,
            verbose=True,
        )
        return self.crew.kickoff().pydantic, None
    
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
        
        # Run the crew
        self.crew.kickoff()
        
        # Get the result and context
        result = self.tasks.create_summarizer_task.output
        context = self.tasks.create_retrieval_task.output.pydantic
        return result, context
        
if __name__ == "__main__":
    pass
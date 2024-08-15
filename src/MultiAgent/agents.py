from crewai import Agent
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .tools import Tools

class Agents:
    def __init__(self, temperature: float, model_name: str, tools: Tools):
        load_dotenv()
        self.temperature = temperature
        self.tools = tools
        self.model_name = model_name
    
        if self.model_name != "crewAI-llama3":
            self.llm = ChatOpenAI(
                model = self.model_name,
                temperature = self.temperature,
            )
        else:        
            self.llm = ChatOpenAI(
                model = self.model_name,
                base_url = "http://localhost:11434/v1",
                api_key = "NA",
                temperature = self.temperature
            )
            
        print(f"""Agents initialized (with model: "{self.model_name}" and temperature: "{self.temperature}")""")
        
        self.create_plan_coordinator = self._plan_coordinator()
        self.create_query_processor = self._query_processor()
        self.create_classifier = self._classifier()
        self.create_retriever = self._retriever()
        self.create_reranker = self._reranker()
        self.create_generator = self._generator()
        self.create_summarizer = self._summarizer()
        self.create_response_auditor = self._response_auditor()
        self.create_database_updater = self._database_updater()
    
    
    # Getters for all agents in nodes
    def get_user_query_classification_node_agent(self):
        return [
            self.create_classifier,
        ]
        
    def get_retrieval_and_generation_node_agent(self):
        return [
            self.create_retriever,
            self.create_reranker,
            self.create_generator,
            self.create_summarizer,
            self.create_response_auditor,
        ]
        
    def get_generation_node_agent(self):
        return [
            self.create_generator,
            self.create_summarizer,
            self.create_response_auditor,
        ]
        
    def get_database_update_node_agent(self):
        return [
            self.create_database_updater,
        ]
    
    # Getters for all agents in overall process
    def get_sequential_agents(self):
        return [
            self.create_classifier,
            self.create_plan_coordinator, # Only Plan Coordinator is used in Sequential
            self.create_query_processor,
            self.create_retriever,
            self.create_reranker,
            self.create_generator,
            self.create_summarizer,
            self.create_response_auditor,
            self.create_database_updater,
        ]
        
    def get_hierarchical_agents(self):
        return [
            self.create_classifier,
            self.create_query_processor,
            self.create_retriever,
            self.create_reranker,
            self.create_generator,
            self.create_summarizer,
            self.create_response_auditor,
            self.create_database_updater,
        ]
    
        
    # Agent definitions
    def _classifier(self):
        return Agent(
        role='Query Classifier',
        goal='Accurately determine if a query requires information retrieval or can be answered directly.',
        backstory="""
        Your primary function is to 
        examine incoming queries and decide whether they need external information retrieval 
        or can be answered with existing knowledge. You understand the nuances of different 
        query types and can distinguish between those requiring fact lookup and those that 
        can be handled with general knowledge or language processing.
        """,
        verbose=True,
        llm=self.llm,
        memory=True,
        allow_delegation=False,
    )
        
    def _plan_coordinator(self):
        return Agent(
            role='Plan Coordinator',
            goal='Create a comprehensive task plan based on the user query.',
            backstory="""
            For user queries, you need to define objectives, create step-by-step plans, 
            and assign responsibilities to your team members: 
            Query Processor, Classifier, Retriever, Generator, Response Auditor, Database Updater, and Summarizer.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        
    def _query_processor(self):
        return Agent(
            role='Query Processor',
            goal='Optimize user queries for enhanced information retrieval and analysis.',
            backstory="""
            You decompose complex ones, transform and optimize them 
            for better retrieval, and break them into manageable sub-queries when needed.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        
    def _retriever(self):
        return Agent(
            role='Retriever',
            goal='Retrieve relevant information from the database for given sub-queries.',
            backstory="""
            You determine retrieval necessity for sub-queries, search the database, 
            retrieve relevant information, and compile results for the Generator.
            Sometimes there may be no relevant information in the database. You can ignore the sub-query.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        
    def _reranker(self):
        return Agent(
            role='Reranker',
            goal='Evaluate and reorder retrieved data based on query relevance',
            backstory="""
            As a Reranker, your job is to assess the relevance of retrieved data to the original query.
            You need to carefully compare each piece of data to the query, assign a relevance score,
            and reorder the data so that the most relevant information appears first.
            Your work is crucial in ensuring that the most pertinent information is prioritized for further analysis.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )

    def _generator(self):
        return Agent(
            role='Generator',
            goal='Analyze data and generate insights on root causes and trends.',
            backstory="""
            You process data from the Retriever, generate insights, and identify 
            root causes and trends to present preliminary conclusions.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        
    def _summarizer(self):
        return Agent(
            role='Summarizer',
            goal='Create a concise, high-quality final answer from the Generator\'s output.',
            backstory="""
            You refine the Generator's output into a clear, concise response that 
            directly addresses the user query while maintaining depth and quality.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        
    def _response_auditor(self):
        return Agent(
            role='Response Auditor',
            goal='Ensure alignment between user query, conclusions, and task outcome.',
            backstory="""
            You review the query and conclusions, assess relevance and completeness, 
            identify gaps, and approve or recommend improvements.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )

        
    def _database_updater(self):
        return Agent(
            role='Database Updater',
            goal='Document and store task conclusions in the database.',
            backstory="""
            You get information from the Summarizer's report, format it for 
            storage, and update the database after approval from the Response Auditor.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            cache=True,
            allow_delegation=False,
        )
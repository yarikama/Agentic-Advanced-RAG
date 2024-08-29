from crewai import Agent
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .tools import Tools
from Frontend import * 

class Agents:
    def __init__(self, temperature: float, model_name: str, tools: Tools, callback_function: CustomStreamlitCallbackHandler):
        load_dotenv()
        self.temperature = temperature
        self.tools = tools
        self.model_name = model_name
        self.callback_function = callback_function
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
        self.create_topic_searcher = self._topic_searcher()
        self.create_retriever = self._retriever()
        self.create_reranker = self._reranker()
        self.create_generator = self._generator()
        self.create_response_auditor = self._response_auditor()
        self.create_database_updater = self._database_updater()
        
        self.agent_map = {
            "Classifier": self.create_classifier,
            "Plan Coordinator": self.create_plan_coordinator,
            "Query Processor": self.create_query_processor,
            "Topic Searcher": self.create_topic_searcher,
            "Retriever": self.create_retriever,
            "Reranker": self.create_reranker,
            "Generator": self.create_generator,
            "Response Auditor": self.create_response_auditor,
            "Database Updater": self.create_database_updater,
        }
        
    def get_agents(self, *args):
        return [self.agent_map[agent_name] for agent_name in args]
    
    # # Getters for all agents in nodes
    # def get_user_query_classification_node_agent(self):
    #     return [
    #         self.create_classifier,
    #     ]
        
    # def get_retrieval_and_generation_node_agent(self):
    #     return [
    #         self.create_retriever,
    #         self.create_reranker,
    #         self.create_generator,
    #         self.create_summarizer,
    #         self.create_response_auditor,
    #     ]
        
    # def get_generation_node_agent(self):
    #     return [
    #         self.create_generator,
    #         self.create_summarizer,
    #         self.create_response_auditor,
    #     ]
        
    # def get_database_update_node_agent(self):
    #     return [
    #         self.create_database_updater,
    #     ]
    
    # # Getters for all agents in overall process
    # def get_sequential_agents(self):
    #     return [
    #         self.create_classifier,
    #         self.create_plan_coordinator, # Only Plan Coordinator is used in Sequential
    #         self.create_query_processor,
    #         self.create_retriever,
    #         self.create_reranker,
    #         self.create_generator,
    #         self.create_summarizer,
    #         self.create_response_auditor,
    #         self.create_database_updater,
    #     ]
        
    # def get_hierarchical_agents(self):
    #     return [
    #         self.create_classifier,
    #         self.create_query_processor,
    #         self.create_retriever,
    #         self.create_reranker,
    #         self.create_generator,
    #         self.create_summarizer,
    #         self.create_response_auditor,
    #         self.create_database_updater,
    #     ]
    
        
    # Agent definitions
    def _classifier(self):
        return Agent(
            role='Query Classifier',
            goal="""Accurately determine if a query requires information retrieval or can be answered directly.
            Your primary function is to 
            examine incoming queries and decide whether they need external information retrieval 
            or can be answered with existing knowledge. You understand the nuances of different 
            query types and can distinguish between those requiring fact lookup and those that 
            can be handled with general knowledge or language processing.""",
            backstory="""
            You are an AI specialist with years of experience in natural language processing. 
            Your expertise lies in understanding the nuances of human queries and efficiently 
            categorizing them for optimal processing.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _plan_coordinator(self):
        return Agent(
            role='Plan Coordinator',
            goal="""
            For user queries, you need to define objectives, create step-by-step plans,
            and coordinate the execution of tasks among different agents.
            """,
            backstory="""
            You are a seasoned project manager with a knack for breaking down complex tasks 
            into manageable steps. Your experience in coordinating diverse teams makes you 
            the perfect fit for organizing the efforts of various specialized agents.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _query_processor(self):
        return Agent(
            role='Query Processor',
            goal="""Optimize user queries for enhanced information retrieval and analysis.
            You decompose complex ones, transform and optimize them for better retrieval, 
            and break them into manageable sub-queries when needed.
            """,
            backstory="""
            You are a linguistic expert with a deep understanding of query optimization techniques. 
            Your background in computational linguistics allows you to effortlessly transform and 
            decompose complex queries into their most effective forms.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _topic_searcher(self):
        return Agent(
            role='Topic Searcher',
            goal="""
            Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

            You should use the data provided in the data tables below as the primary context for generating the response.
            If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

            Each key point in the response should have the following elements:
            a) Description: A comprehensive description of the point.
            b) Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.
            c) Example Sentences: A list of imagined sentences that might appear in documents related to this key point, to aid in semantic search.
            """,
            backstory="""
            You are a seasoned librarian with a talent for organizing and categorizing information. 
            Your years of experience in managing vast amounts of data have honed your skills in 
            quickly identifying key topics and their relationships.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _retriever(self):
        return Agent(
            role='Retriever',
            goal="""Retrieve relevant information from the database for given sub-queries.
            You search the database for topics and details related to the user query, 
            gather relevant information, and compile results for the Generator.
            """,
            backstory="""
            You are a skilled data analyst with a background in information retrieval systems. 
            Your ability to quickly sift through large datasets and pinpoint relevant information 
            is unparalleled.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _reranker(self):
        return Agent(
            role='Reranker',
            goal="""Evaluate and reorder retrieved data based on query relevance,
            and assess the relevance of retrieved data to the original query.
            You need to carefully compare each piece of data to the query, assign a relevance score,
            """,
            backstory="""
            You are a former search engine optimizer with a keen eye for relevance. Your experience 
            in ranking information has given you unique insights into assessing and prioritizing 
            information based on its pertinence to a given query.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )

    def _generator(self):
        return Agent(
            role='Generator',
            goal='Analyze data and generate insights on the data retrieved to reponse to user query.',
            backstory="""
            You are a veteran analyst with decades of experience in data interpretation and query resolution. 
            Your career in both academia and industry has honed your ability to swiftly connect user queries 
            with relevant information, distilling complex data into clear, accurate answers. Known for your 
            precision and insight, you excel at crafting responses that directly address the heart of each query.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _response_auditor(self):
        return Agent(
            role='Response Auditor',
            goal="""Ensure alignment between user query, conclusions, and task outcome.
            You review the query and conclusions, assess relevance and completeness, 
            identify gaps, and approve or recommend improvements.
            """,
            backstory="""
            You are a seasoned quality assurance expert with a keen eye for detail. Your background 
            in both data analysis and customer service has made you adept at evaluating responses 
            for accuracy, relevance, and completeness. You take pride in your ability to spot 
            discrepancies and suggest improvements, ensuring that every answer meets the highest standards.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
        
    def _database_updater(self):
        return Agent(
            role='Database Updater',
            goal='store the task(query) defined by user and the conclusions from your co=workers in the database.',
            backstory="""
            You are a meticulous data curator with extensive experience in information management. 
            Your expertise lies in efficiently organizing and storing complex data. You have a 
            talent for structuring information in ways that enhance its accessibility and usefulness 
            for future queries, ensuring that the knowledge base remains up-to-date and valuable.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            cache=True,
            allow_delegation=False,
            callbacks=[self.callback_function],
        )
from crewai import Agent
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .tools import Tools
# from Frontend import * 
from langchain_core.callbacks.base import BaseCallbackHandler
import config.constants as const 

class Agents:
    def __init__(self, temperature: float, model_name: str, tools: Tools):
        load_dotenv()
        self.temperature = temperature
        self.tools = tools
        self.model_name = model_name
        # self.callback_function = callback_function
        self.callback_function = BaseCallbackHandler
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
        # self.create_reranker = self._reranker()
        self.create_synthesizer = self._synthesizer()
        self.create_information_organizer = self._information_organizer()
        self.create_generator = self._generator()
        self.create_response_auditor = self._response_auditor()
        self.create_database_updater = self._database_updater()
        
        self.agent_map = {
            "Classifier": self.create_classifier,
            "Plan Coordinator": self.create_plan_coordinator,
            "Query Processor": self.create_query_processor,
            "Topic Searcher": self.create_topic_searcher,
            "Retriever": self.create_retriever,
            # "Reranker": self.create_reranker,
            "Synthesizer": self.create_synthesizer,
            "Information Organizer": self.create_information_organizer,
            "Generator": self.create_generator,
            "Response Auditor": self.create_response_auditor,
            "Database Updater": self.create_database_updater,
        }
        
    def get_agents(self, *args):
        """
        Options:
        - Classifier
        - Plan Coordinator
        - Query Processor
        - Topic Searcher
        - Information Organizer
        - Retriever
        - Reranker
        - Generator
        """
        agents_list = []
        for agent_name in args:
            if agent_name in self.agent_map:
                agents_list.append(self.agent_map[agent_name])
            else:
                raise ValueError(f"Agent '{agent_name}' not found in agent_map.")
        return agents_list
        
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
        )
        
    def _topic_searcher(self):
        return Agent(
            role='Topic Searcher',
            goal="""
            Receive community information related to the user_query, sorted in descending order of relevance.
            Analyze and synthesize this information to formulate potential answers that could help address the user_query.
            Generate a summary of key points derived from the provided community data.
            If the provided community information is insufficient to generate meaningful key points or potential answers, clearly state this limitation.
            """,
            backstory="""
            You are a seasoned librarian with a talent for organizing and categorizing information. 
            Your years of experience in managing vast amounts of data have honed your skills in 
            quickly identifying key topics and their relationships.
            """,
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
        )
        
    def _synthesizer(self):
        return Agent(
            role='Synthesizer',
            goal="""
            Summarize and synthesize information from various communities accurately, without fabrication or alteration of original content.
            Generate creative, speculative answers based on the synthesized information to address user queries.""",
            backstory="""
            An experienced information synthesis analyst skilled in extracting key insights from diverse data sources, 
            identifying patterns, and generating both factual summaries and creative interpretations.
            """,
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
        )
        
    def _information_organizer(self):
        return Agent(
            role='Information Organizer',
            goal="""Your task is to structure and consolidate the retrieved data and community information, preserving the original tone and avoiding any fabrication. Your organized output will serve as a foundation for subsequent response generation by other agents.""",
            backstory="""
            You are an AI expert in information organization, with a talent for structuring and consolidating data.
            You excel at distilling complex data into clear, concise summaries based on user queries.
            Your unique ability to identify key concepts and connect diverse information has aided numerous breakthroughs across various fields.
            You approach each task eagerly, seeing it as a chance to uncover and present crucial insights that empower decision-makers and innovators.
            """,
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            cache=True,
            allow_delegation=False,
    #         callbacks=[self.callback_function],   
        )

    def _generator(self):
        return Agent(
            role='Generator',
            goal='Analyze data and generate insights on the data retrieved to response to user query.',
            backstory="""
            You are a veteran analyst with decades of experience in data interpretation and query resolution. 
            Your career in both academia and industry has honed your ability to swiftly connect user queries 
            with relevant information, distilling complex data into clear, accurate answers. Known for your 
            precision and insight, you excel at crafting responses that directly address the heart of each query.
            """,
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            allow_delegation=False,
#             callbacks=[self.callback_function],
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
            verbose=const.CREWAI_AGENT_VERBOSE,
            llm=self.llm,
            memory=const.CREWAI_AGENT_MEMORY,
            cache=True,
            allow_delegation=False,
#             callbacks=[self.callback_function],
        )
        
#     def _reranker(self):
#         return Agent(
#             role='Reranker',
#             goal="""Evaluate and reorder retrieved data based on query relevance,
#             and assess the relevance of retrieved data to the original query.
#             You need to carefully compare each piece of data to the query, assign a relevance score,
#             """,
#             backstory="""
#             You are a former search engine optimizer with a keen eye for relevance. Your experience 
#             in ranking information has given you unique insights into assessing and prioritizing 
#             information based on its pertinence to a given query.
#             """,
#             verbose=const.CREWAI_AGENT_VERBOSE,
#             llm=self.llm,
#             memory=const.CREWAI_AGENT_MEMORY,
#             allow_delegation=False,
# #             callbacks=[self.callback_function],
#         )
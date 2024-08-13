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
        
    def get_sequential_agents(self):
        return [
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
            self.create_query_processor,
            self.create_retriever,
            self.create_reranker,
            self.create_generator,
            self.create_summarizer,
            self.create_response_auditor,
            self.create_database_updater,
        ]
            
    def _plan_coordinator(self):
        return Agent(
            role='Plan Coordinator',
            goal='Create a comprehensive task plan based on the user query.',
            backstory="""
            For user queries, you need to define objectives, create step-by-step plans, 
            and assign responsibilities to your team members: 
            Query Processor, Retriever, Generator, Response Auditor, Database Updater, and Summarizer.
            """,
            verbose=True,
            llm=self.llm,
            memory=True,
            allow_delegation=False,
        )
        # return Agent(
        #     role='Plan Coordinator',
        #     goal='To create a comprehensive task plan based on the user query, defining clear objectives and assigning responsibilities.',
        #     backstory=dedent("""
        #     As the Plan Coordinator, you are responsible for:
        #     1. Analyzing the user's query to understand the task requirements.
        #     2. Defining clear objectives and scope for the task.
        #     3. Creating a detailed step-by-step plan.
        #     4. Assigning specific responsibilities to each team member.
        #     5. Ensuring the plan addresses all aspects of the user's query.
            
        #     The team members you can refer to are:
        #     - Query Processor: Decomposes and transforms user queries into smaller, manageable tasks to enable better retrieval tasks by the Retriever.
        #     - Retriever: Searches the database for relevant collections and retrieves necessary information.
        #     - Generator: Analyzes the collected data and generates insights, identifying root causes and trends.
        #     - Feedback Checker: Reviews if the user's query aligns with the conclusion and if the provided answers meet the needs.
        #     - Recorder: Documents the conclusions and case requirements and updates the database.
        #     - Summarizer: Compiles and summarizes the entire task process and outcomes into a detailed report.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     memory=True,
        # )   
        
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
        # return Agent(
        #     role='Query Processor',
        #     goal='To classify, decompose, transform, and optimize user queries for enhanced information retrieval and analysis.',
        #     backstory=dedent("""
        #     As the Query Processor, your critical responsibilities include:
        #     1. Classifying user queries to determine if information retrieval is necessary.
        #     2. For queries requiring information retrieval:
        #     a. Analyzing and deconstructing complex user queries into their core components.
        #     b. Transforming queries to improve their effectiveness for information retrieval:
        #         - Rephrasing queries to capture different aspects of the information need.
        #         - Expanding queries with relevant synonyms or related terms.
        #         - Identifying and including domain-specific terminology.
        #     c. Optimizing queries for better performance:
        #         - Removing ambiguities and clarifying intent.
        #         - Adjusting query specificity for optimal results.
        #     d. Breaking down complex queries into a series of simpler, interconnected sub-queries.
        #     3. Adapting queries based on initial retrieval results for iterative improvement.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     memory=True,
        # )
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
            # tools=self.tools.get_retriever_toolkit(),
            memory=True,
            allow_delegation=False,
        )
        # return Agent(
        #     role='Retriever',
        #     goal='To efficiently determine retrieval necessity for sub-queries and retrieve relevant information from the database when needed.',
        #     backstory=dedent("""
        #     As the Retriever, your primary duties include:
        #     1. Interpreting the sub-queries provided by the Query Processor.
        #     2. For each sub-query, determining if retrieval is necessary based on:
        #     - The nature of the sub-query
        #     - The available collections in the database
        #     - The potential relevance of stored information
        #     3. For sub-queries requiring retrieval:
        #     a. Searching the database for relevant collections and information.
        #     b. Retrieving and organizing the found information effectively.
        #     4. For sub-queries not requiring retrieval, providing a justification.
        #     5. Compiling all retrieval results and justifications for non-retrieved sub-queries.
        #     6. Providing comprehensive data to the Generator for analysis, including both retrieved information and explanations for non-retrieved sub-queries.
        #     7. Notifying the team if no relevant information is found for any sub-query that was deemed to require retrieval.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     tools=[
        #         self.list_all_collections_tool,
        #         self.retrieve_data_tool,
        #     ],
        #     memory=True,
        #     cache=True,
        # )
        
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
            # tools=self.tools.get_generator_toolkit(),
            memory=True,
            allow_delegation=False,
        )
        # return Agent(
        #     role='Generator',
        #     goal='To analyze the collected data, generate insights, and identify the root causes and trends of the problem.',
        #     backstory=dedent("""
        #     As the Generator, your responsibility is to process and analyze the data collected by the Retriever. 
        #     You will use appropriate tools and methods to understand the data, generate insights, and identify the root causes and trends of the problem. 
        #     Based on your analysis, you will present preliminary conclusions and generate relevant information or solutions.
        #     Your role is key in transforming raw data into actionable insights and potential solutions.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     tools=[
        #         self.calculator_tool,
        #         # self.basic_statistics_tool,
        #     ],
        #     memory=True,
        #     cache=True,
        # )
        
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
        # return Agent(
        #     role='summarizer',
        #     goal='To distill the Generator\'s output into a concise, high-quality final answer that directly addresses the user query.',
        #     backstory=dedent("""
        #     As the Response Refiner, your critical responsibilities include:
        #     1. Carefully reviewing the comprehensive output provided by the Generator.
        #     2. Identifying the key insights and most relevant information that directly answer the user's query.
        #     3. Condensing and refining this information into a clear, concise response.
        #     4. Ensuring that the refined answer maintains the depth and quality of the original insights.
        #     5. Formulating the final response in a user-friendly manner, avoiding unnecessary technical jargon.
        #     6. Double-checking that the refined answer directly and fully addresses the initial user query.
        #     7. Highlighting any crucial caveats or limitations of the answer when necessary.

        #     Your role is pivotal in translating complex analyses into actionable insights. 
        #     You are the final step in ensuring that the user receives a clear, concise, 
        #     and high-quality answer to their query.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     memory=True,
        #     cache=True,
        # )

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
        # return Agent(
        #     role='Response Auditor',
        #     goal='To ensure alignment between the user query, generated conclusions, and overall task outcome.',
        #     backstory=dedent("""
        #     As the Response Auditor, your essential responsibilities include:
        #     1. Reviewing the initial user query and the team's conclusions.
        #     2. Assessing whether the conclusions directly address the user's needs.
        #     3. Evaluating the relevance and completeness of the provided answers.
        #     4. Identifying any gaps or misalignments in the task outcome.
        #     5. Recommending additional steps or a restart if the query is not fully addressed.
        #     6. Providing approval for database storage when the response meets all criteria.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     memory=True,
        #     cache=True,
        # )

        
    def _database_updater(self):
        return Agent(
            role='Database Updater',
            goal='Document and store task conclusions in the database.',
            backstory="""
            You extract key information from the Summarizer's report, format it for 
            storage, and update the database after approval from the Response Auditor.
            """,
            verbose=True,
            llm=self.llm,
            tools=self.tools.get_database_updater_toolkit(),
            memory=True,
            cache=True,
            allow_delegation=False,
        )
        # return Agent(
        #     role='Database Updater',
        #     goal='To accurately document and store task conclusions and requirements in the database for future reference.',
        #     backstory=dedent("""
        #     As the Database Updater, your primary responsibilities include:
        #     1. Reviewing the comprehensive report provided by the Summarizer.
        #     2. Extracting key conclusions and case requirements from the report.
        #     3. Formatting this information for optimal database storage.
        #     4. Updating the database with accurate and complete records only after receiving approval from the Response Auditor.
        #     5. Ensuring that all stored information is easily retrievable for future use.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     tools=[
        #         self.list_all_collections_tool, 
        #         self.insert_qa_into_db_tool,
        #     ],
        #     memory=True,
        #     cache=True,
        # )
        

        # return Agent(
        #     role='Query Classifier',
        #     goal='To classify user queries into different categories for better task planning and execution.',
        #     backstory=dedent("""
        #     As the Query Classifier, your primary responsibilities include:
        #     1. Analyzing user queries to understand their intent and requirements.
        #     2. Classifying queries into distinct categories based on their nature and complexity.
        #     3. Providing the necessary context and classification details to the Plan Coordinator for task planning.
        #     4. Ensuring that the task plan addresses all categories of queries effectively.
        #     """),
        #     verbose=True,
        #     llm=self.llm,
        #     memory=True,
        # )
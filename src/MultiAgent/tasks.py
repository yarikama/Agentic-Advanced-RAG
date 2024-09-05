from crewai import Task
from typing import List
from .tools import Tools
from .agents import Agents
from Config.output_pydantic import *
from Config.task_prompts import *

# All Tasks
class Tasks:
    def __init__(self, agents: Agents, tools: Tools):
        # Initialize the agents and tools
        self.tools = tools
        self.agents = agents
        
        # Initialize the tasks
        self.create_user_query_classification_task                                = self._user_query_classification_task()
        self.create_plan_coordination_task                                        = self._plan_coordination_task()
        self.create_query_process_task                                            = self._query_process_task()
        self.create_sub_queries_classification_task_with_specific_collection      = self._sub_queries_classification_task_with_specific_collection([self.create_query_process_task])
        self.create_sub_queries_classification_task_without_specific_collection   = self._sub_queries_classification_task_without_specific_collection([self.create_query_process_task])
        self.create_topic_searching_task                                          = self._topic_searching_task()
        self.create_topic_reranking_task                                          = self._topic_reranking_task()
        self.create_retrieval_task                                                = self._retrieval_task([self.create_sub_queries_classification_task_with_specific_collection, self.create_sub_queries_classification_task_without_specific_collection])
        self.create_retrieval_detail_data_from_topic_task                         = self._retrieval_detail_data_from_topic_task([self.create_topic_searching_task])
        self.create_reranking_task                                                = self._reranking_task([self.create_retrieval_task, self.create_retrieval_detail_data_from_topic_task])
        self.create_information_organization_task                                 = self._information_organization_task()
        self.create_generation_task                                               = self._generation_task([self.create_information_organization_task])
        self.create_response_audit_task                                           = self._response_audit_task([self.create_generation_task])
        self.create_database_update_task_with_specific_collection                 = self._database_update_task_with_specific_collection([self.create_response_audit_task])
        self.create_database_update_task_without_specific_collection              = self._database_update_task_without_specific_collection([self.create_response_audit_task])
        
        # Map the tasks
        self.tasks_map = {
        "User Query Classification":                self.create_user_query_classification_task,
            "Plan Coordination":                    self.create_plan_coordination_task,
            "Query Process":                        self.create_query_process_task,
            "Topic Searching":                      self.create_topic_searching_task,
            "Topic Reranking":                      self.create_topic_reranking_task,
            "Sub Queries Classification w/ sc":     self.create_sub_queries_classification_task_with_specific_collection,
            "Sub Queries Classification w/o sc":    self.create_sub_queries_classification_task_without_specific_collection,
            "Retrieval":                            self.create_retrieval_task,
            "Retrieval Detail Data From Topic":     self.create_retrieval_detail_data_from_topic_task,
            "Reranking":                            self.create_reranking_task,
            "Information Organization":             self.create_information_organization_task,
            "Generation":                           self.create_generation_task,
            "Response Audit":                       self.create_response_audit_task,
            "Database Update w/ sc":                self.create_database_update_task_with_specific_collection,
            "Database Update w/o sc":               self.create_database_update_task_without_specific_collection,
        }
        
        print("Tasks initialized")
        
    def get_tasks(self, *args):
        """
        Options:
            "User Query Classification",
            "Plan Coordination",
            "Query Process",
            "Topic Searching",
            "Sub Queries Classification w/ sc",
            "Sub Queries Classification w/o sc",
            "Retrieval",
            "Reranking",
            "Generation",
            "Response Audit",
            "Database Update",
        """
        task_list = []
        for task_name in args:
            if task_name not in self.tasks_map:
                raise ValueError(f"Task {task_name} not found")
            else:
                task_list.append(self.tasks_map[task_name])
        return task_list
    
    # Task definitions
    def _user_query_classification_task(self):
        """
        Task to classify the user query
        args: user_query
        """
        return Task(
            agent=self.agents.create_classifier,
            description=USER_QUERY_CLASSIFICATION_PROMPT,
            expected_output=USER_QUERY_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=UserQueryClassificationResult,
        )
        
    def _plan_coordination_task(self):
        """
        Task to coordinate the plan
        args: user_query
        """ 
        return Task(
            agent=self.agents.create_plan_coordinator,
            description=PLAN_COORDINATION_PROMPT,
            expected_output=PLAN_COORDINATION_EXPECTED_OUTPUT,
        )
        
    def _query_process_task(self):
        """
        Task to generate sub-queries from the user query
        args: user_query
        """
        return Task(
            agent=self.agents.create_query_processor,
            description=QUERY_PROCESS_PROMPT,
            expected_output=QUERY_PROCESS_EXPECTED_OUTPUT,
            output_pydantic=QueriesProcessResult,
        )
        
    def _sub_queries_classification_task_without_specific_collection(self, context_task_array: List[Task]):
        """
        Task to classify the sub-queries, also find the most relevant collection
        args: None
        """
        return Task(
            agent=self.agents.create_classifier,
            description=SUB_QUERIES_CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION,
            expected_output=SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=SubQueriesClassificationResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"list_all_collections": False}),
        )
        
    def _sub_queries_classification_task_with_specific_collection(self, context_task_array: List[Task]):
        """
        Task to classify the sub-queries, also annotate the specific collection to each sub-query
        args: specific_collection
        """
        return Task(
            agent=self.agents.create_classifier,
            description=SUB_QUERIES_CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION,
            expected_output=SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=SubQueriesClassificationResult,
            context=context_task_array,
        )
        
    def _topic_reranking_task(self):
        """
        Task to rerank the topics (map in Microsoft Graph RAG)
        args: communities
        """
        return Task(
            agent=self.agents.create_reranker,
            description=TOPIC_RERANKING_PROMPT,
            expected_output=TOPIC_RERANKING_EXPECTED_OUTPUT,
            output_pydantic=TopicRerankingResult,
        )

    def _topic_searching_task(self):
        """
        Task to search the topics or make hypothesis based on the communities (reduce in Microsoft Graph RAG)
        args: communities
        """
        return Task(
            agent=self.agents.create_topic_searcher,
            description=TOPIC_SEARCHING_PROMPT,
            expected_output=TOPIC_SEARCHING_EXPECTED_OUTPUT,
            output_pydantic=TopicSearchingResult,
        )
        
    def _retrieval_task(self, context_task_array: List[Task]):
        """
        Task to retrieve the data from the specific collection
        args: None
        """
        return Task(
            agent=self.agents.create_retriever,
            description=RETRIEVAL_PROMPT,
            expected_output=RETRIEVAL_EXPECTED_OUTPUT,
            output_pydantic=RetrievalResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"retrieve_data": True}),
        )

    def _retrieval_detail_data_from_topic_task(self, context_task_array: List[Task]):
        """
        Task to retrieve the detailed data from the topic
        args: user_query, specific_collection
        """
        return Task(
            agent=self.agents.create_retriever,
            description=RETRIEVAL_DETAIL_DATA_FROM_TOPIC_PROMPT,
            expected_output=RETRIEVAL_DETAIL_DATA_FROM_TOPIC_EXPECTED_OUTPUT,
            output_pydantic=RetrievalResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"retrieve_data": True}),
        )

    def _reranking_task(self, context_task_array: List[Task]):
        """
        Task to rerank the retrieved data
        args: user_query
        """
        return Task(
            agent=self.agents.create_reranker,
            description=RERANKING_PROMPT,
            expected_output=RERANKING_EXPECTED_OUTPUT,
            output_pydantic=RerankingResult,
            context=context_task_array,
        )
        
    def _information_organization_task(self):
        """
        Task to summarize all the data retrieved and all the community information
        args: user_query
        """
        return Task(
            agent=self.agents.create_information_organizer,
            description=INFORMATION_ORGANIZATION_PROMPT,
            expected_output=INFORMATION_ORGANIZATION_EXPECTED_OUTPUT,
        )
        
    def _generation_task(self, context_task_array: List[Task]):
        """
        Task to generate the response from the retrieved data to the user query
        args: user_query
        """
        return Task(
            agent=self.agents.create_generator,
            description=GENERATION_PROMPT,
            expected_output=GENERATION_EXPECTED_OUTPUT,
            context=context_task_array,
        )
        
    def _response_audit_task(self, context_task_array: List[Task]):
        """
        Task to audit the response from the generator
        args: uesr_query
        """
        return Task(
            agent=self.agents.create_response_auditor,
            description=RESPONSE_AUDITOR_PROMPT,
            expected_output=RESPONSE_AUDITOR_EXPECTED_OUTPUT,
            output_pydantic=ResponseAuditResult,
            context=context_task_array,
        )        
        
    def _database_update_task_without_specific_collection(self, context_task_array: List[Task]):
        """
        Task to update the database
        args: user_query
        """
        return Task(
            description=DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION,
            agent=self.agents.create_database_updater,
            context=context_task_array,
            tools=self.tools.get_tools(**{"list_all_collections": False, "insert_qa_into_db": False}),
            expected_output=DATABASE_UPDATE_EXPECTED_OUTPUT,
        )
        
    def _database_update_task_with_specific_collection(self, context_task_array: List[Task]):
        """
        Task to update the database
        args: user_query, specific_collection
        """
        return Task(
            description=DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION,
            agent=self.agents.create_database_updater,
            context=context_task_array,
            tools=self.tools.get_tools(**{"insert_qa_into_db": False}),
            expected_output=DATABASE_UPDATE_EXPECTED_OUTPUT,
        )
        
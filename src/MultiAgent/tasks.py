from crewai import Task
from typing import List
from .tools import Tools
from .agents import Agents
from Config.output_pydantic import *
from Config.task_prompts import *

# All Tasks
class Tasks:
    def __init__(self, agents: Agents, tools: Tools):
        self.tools = tools
        self.agents = agents
        print("Tasks initialized")
        
    # Update the task with new query and collection
    def update_tasks(self, **kwargs):
        self.user_query = kwargs.get("user_query", None)
        self.specific_collection = kwargs.get("specific_collection", None)
        
        # Initialize the tasks
        self.create_user_query_classification_task                              = self._user_query_classification_task()
        self.create_plan_coordination_task                                      = self._plan_coordination_task()
        self.create_query_process_task                                          = self._query_process_task()
        self.create_sub_queries_classification_task_with_specific_collection    = self._sub_queries_classification_task_with_specific_collection([self.create_query_process_task])
        self.create_sub_queries_classification_task_without_specific_collection = self._sub_queries_classification_task_without_specific_collection([self.create_query_process_task])
        self.create_topic_searching_task                                        = self._topic_searching_task([self.create_sub_queries_classification_task_with_specific_collection, self.create_sub_queries_classification_task_without_specific_collection])
        self.create_retrieval_task                                              = self._retrieval_task([self.create_sub_queries_classification_task_with_specific_collection, self.create_sub_queries_classification_task_without_specific_collection])
        self.create_retrieval_detail_data_from_topic_task                       = self._retrieval_detail_data_from_topic_task([self.create_topic_searching_task])
        self.create_reranking_task                                              = self._reranking_task([self.create_retrieval_task, self.create_retrieval_detail_data_from_topic_task])
        self.create_generation_task                                             = self._generation_task([self.create_reranking_task])
        self.create_response_audit_task                                         = self._response_audit_task([self.create_generation_task])
        self.create_database_update_task_with_specific_collection               = self._database_update_task_with_specific_collection([self.create_response_audit_task])
        self.create_database_update_task_without_specific_collection            = self._database_update_task_without_specific_collection([self.create_response_audit_task])
        
        self.tasks_map = {
            "User Query Classification": self.create_user_query_classification_task,
            "Plan Coordination": self.create_plan_coordination_task,
            "Query Process": self.create_query_process_task,
            "Topic Searching": self.create_topic_searching_task,
            "Sub Queries Classification w/ sc": self.create_sub_queries_classification_task_with_specific_collection,
            "Sub Queries Classification w/o sc": self.create_sub_queries_classification_task_without_specific_collection,
            "Retrieval": self.create_retrieval_task,
            "Rerank": self.create_reranking_task,
            "Generation": self.create_generation_task,
            "Response Audit": self.create_response_audit_task,
            "Database Update w/ sc": self.create_database_update_task_with_specific_collection,
            "Database Update w/o sc": self.create_database_update_task_without_specific_collection,
        }
        print("User Query and Collection updated")

    def get_tasks(self, *args):
        """
        Options:
            "User Query Classification",
            "Plan Coordination",
            "Query Process",
            "Topic Searching",
            "Sub Queries Classification",
            "Retrieval",
            "Rerank",
            "Generation",
            "Response Audit",
            "Database Update",
        """
        return [self.tasks_map[task_name] for task_name in args]
        
    # Task definitions
    def _user_query_classification_task(self):
        return Task(
            agent=self.agents.create_classifier,
            description=USER_QUERY_CLASSIFICATION_PROMPT.format(user_query=self.user_query),
            expected_output=USER_QUERY_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=UserQueryClassificationResult,
        )
        
    def _plan_coordination_task(self): 
        return Task(
            agent=self.agents.create_plan_coordinator,
            description=PLAN_COORDINATION_PROMPT.format(user_query=self.user_query),
            expected_output=PLAN_COORDINATION_EXPECTED_OUTPUT,
        )
        
    def _query_process_task(self):
        return Task(
            agent=self.agents.create_query_processor,
            description=QUERY_PROCESS_PROMPT.format(user_query=self.user_query),
            expected_output=QUERY_PROCESS_EXPECTED_OUTPUT,
            output_pydantic=QueriesProcessResult,
        )
        
    def _sub_queries_classification_task_without_specific_collection(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_classifier,
            description=SUB_QUERIES_CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION,
            expected_output=SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=SubQueriesClassificationResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"list_all_collections": False}),
        )
        
    def _sub_queries_classification_task_with_specific_collection(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_classifier,
            description=SUB_QUERIES_CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION.format(specific_collection=self.specific_collection),
            expected_output=SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=SubQueriesClassificationResult,
            context=context_task_array,
        )
        
    def _topic_searching_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_topic_searcher,
            description=TOPIC_SEARCHING_PROMPT,
            expected_output=TOPIC_SEARCHING_EXPECTED_OUTPUT,
            output_pydantic=TopicSearchingResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"global_retrieve_topic": False}),
        )

    def _retrieval_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_retriever,
            description=RETRIEVAL_PROMPT.format(),
            expected_output=RETRIEVAL_EXPECTED_OUTPUT,
            output_pydantic=RetrievalResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"retrieve_data": True}),
        )

    def _retrieval_detail_data_from_topic_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_retriever,
            description=RETRIEVAL_DETAIL_DATA_FROM_TOPIC_PROMPT.format(user_query=self.user_query, specific_collection=self.specific_collection),
            expected_output=RETRIEVAL_DETAIL_DATA_FROM_TOPIC_EXPECTED_OUTPUT,
            output_pydantic=RetrievalResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"retrieve_data": True}),
        )

    def _reranking_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_reranker,
            description=RERANK_PROMPT.format(user_query=self.user_query),
            expected_output=RERANK_EXPECTED_OUTPUT,
            output_pydantic=RerankingResult,
            context=context_task_array,
            tools=self.tools.get_tools(**{"rerank": True}),
        )
        
    def _generation_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_generator,
            description=GENERATION_PROMPT.format(user_query=self.user_query),
            expected_output=GENERATION_EXPECTED_OUTPUT,
            context=context_task_array,
        )
        
    def _response_audit_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_response_auditor,
            description=RESPONSE_AUDITOR_PROMPT.format(user_query=self.user_query),
            expected_output=RESPONSE_AUDITOR_EXPECTED_OUTPUT,
            output_pydantic=ResponseAuditResult,
            context=context_task_array,
        )        
        
    def _database_update_task_without_specific_collection(self, context_task_array: List[Task]):
        return Task(
            description=DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION.format(
                user_query=self.user_query, specific_collection=self.specific_collection
            ),
            agent=self.agents.create_database_updater,
            context=context_task_array,
            tools=self.tools.get_tools(**{"insert_qa_into_db": False}),
            expected_output=DATABASE_UPDATE_EXPECTED_OUTPUT,
        )
        
    def _database_update_task_with_specific_collection(self, context_task_array: List[Task]):
        return Task(
            description=DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION.format(user_query=self.user_query),
            agent=self.agents.create_database_updater,
            context=context_task_array,
            tools=self.tools.get_tools(**{"insert_qa_into_db": False}),
            expected_output=DATABASE_UPDATE_EXPECTED_OUTPUT,
        )
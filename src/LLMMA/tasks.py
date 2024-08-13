from .tools import Tools
from .agents import Agents
from .output_pydantic import *
from crewai import Task
from typing import List
from .prompts import *

# All Tasks
class Tasks:
    def __init__(self, agents: Agents, tools: Tools):
        self.tools = tools
        self.agents = agents
        self.user_query = ""
        self.specific_collection = ""
        print("Tasks initialized")
        
    # Update the task with new query and collection
    def update_task(self, new_query: str, new_collection: str = None):
        # Set the new query and collection
        self.user_query = new_query
        self.specific_collection = new_collection
        
        # Compile the tasks flow here!!! The order matters!!!
        self.create_user_query_classification_task = self._user_query_classification_task()
        self.create_plan_coordinator_task = self._plan_coordinator_task()
        self.create_query_processor_task = self._query_processor_task()
        self.create_classification_task = self._classification_task([self.create_query_processor_task])
        self.create_retrieval_task = self._retrieval_task([self.create_classification_task])
        self.create_rerank_task = self._rerank_task([self.create_retrieval_task])
        self.create_generation_task = self._generation_task([self.create_rerank_task])
        self.create_summarizer_task = self._summarizer_task([self.create_generation_task])
        self.create_response_auditor_task = self._response_auditor_task([self.create_summarizer_task])
        self.create_database_updater_task = self._database_updater_task([self.create_summarizer_task, self.create_response_auditor_task])
    
    # Getters for all tasks in nodes
    def get_user_query_classification_node_task(self):
        return [
            self.create_user_query_classification_task,
        ]
        
    def get_retrieval_and_generation_node_tasks(self):
        return [
            self.create_query_processor_task,
            self.create_classification_task,
            self.create_retrieval_task,
            self.create_rerank_task,
            self.create_generation_task,
            self.create_summarizer_task,
            self.create_response_auditor_task,
        ]
        
    def get_generation_node_tasks(self):
        return [
            self.create_generation_task,
            self.create_summarizer_task,
            self.create_response_auditor_task,
        ]
        
    
    # Getters for all tasks in overall process
    def get_sequential_tasks(self):
        return [
            self.create_plan_coordinator_task, # Only for Sequential Process because it is not needed in Hierarchical
            self.create_query_processor_task,
            self.create_classification_task,
            self.create_retrieval_task,
            self.create_rerank_task,
            self.create_generation_task,
            self.create_summarizer_task,
            self.create_response_auditor_task,
            self.create_database_updater_task,
        ]
        
    def get_hierarchical_tasks(self):
        return [
            self.create_query_processor_task, # Plan Coordinator is included in manager_agent in Crew
            self.create_classification_task,
            self.create_retrieval_task,
            self.create_rerank_task,
            self.create_generation_task,
            self.create_summarizer_task,
            self.create_response_auditor_task,
            self.create_database_updater_task,
        ]
    
    # Task definitions
    def _user_query_classification_task(self):
        return Task(
            agent=self.agents.create_classifier,
            description=USER_QUERY_CLASSIFICATION_PROMPT.format(user_query=self.user_query),
            expected_output=USER_QUERY_CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=UserQueryClassification,
        )
        
    def _plan_coordinator_task(self): 
        return Task(
            agent=self.agents.create_plan_coordinator,
            description=PLAN_COORDINATOR_PROMPT.format(user_query=self.user_query),
            expected_output=PLAN_COORDINATOR_EXPECTED_OUTPUT,
        )
        
    def _query_processor_task(self):
        return Task(
            agent=self.agents.create_query_processor,
            description=QUERY_PROCESSOR_PROMPT.format(user_query=self.user_query),
            expected_output=QUERY_PROCESSOR_EXPECTED_OUTPUT,
            output_pydantic=Queries,
        )
        
    def _classification_task(self, context_task_array: List[Task]):
        if self.specific_collection is None:
            desc = CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION.format(user_query=self.user_query)
            toolkit = self.tools.get_classifiction_toolkit_wo_collection()
        else:
            desc = CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION.format(user_query=self.user_query, specific_collection=self.specific_collection)
            toolkit = self.tools.get_classifiction_toolkit_w_collection()
        return Task(
            agent=self.agents.create_classifier,
            description=desc,
            expected_output=CLASSIFICATION_EXPECTED_OUTPUT,
            output_pydantic=QueriesIdentificationList,
            context=context_task_array,
            tools=toolkit,
        )

    def _retrieval_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_retriever,
            description=RETRIEVAL_PROMPT.format(user_query=self.user_query),
            expected_output=RETRIEVAL_EXPECTED_OUTPUT,
            output_pydantic=RefinedRetrievalData,
            context=context_task_array,
            tools=self.tools.get_retrieve_toolkit(),
        )

    def _rerank_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_reranker,
            description=RERANK_PROMPT.format(user_query=self.user_query),
            expected_output=RERANK_EXPECTED_OUTPUT,
            output_pydantic=RankedRetrievalData,
            context=context_task_array,
            tools=self.tools.get_reranker_toolkit(), 
        )
        
    def _generation_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_generator,
            description=GENERATION_PROMPT.format(user_query=self.user_query),
            expected_output=GENERATION_EXPECTED_OUTPUT,
            context=context_task_array,
        )
        
    def _summarizer_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_summarizer,
            description=SUMMARIZER_PROMPT.format(user_query=self.user_query),
            expected_output=SUMMARIZER_EXPECTED_OUTPUT,
            context=context_task_array
        )
        
    def _response_auditor_task(self, context_task_array: List[Task]):
        return Task(
            agent=self.agents.create_response_auditor,
            description=RESPONSE_AUDITOR_PROMPT.format(user_query=self.user_query),
            expected_output=RESPONSE_AUDITOR_EXPECTED_OUTPUT,
            output_pydantic=AuditResult,
            context=context_task_array,
        )        
        
    def _database_updater_task(self, context_task_array: List[Task]):
        if self.specific_collection is None:
            desc = DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION.format(user_query=self.user_query)
            toolkit = self.tools.get_database_updater_toolkit_wo_collection()
        else:
            desc = DATABASE_UPDATER_PROMPT_WITH_SPECIFIC_COLLECTION.format(user_query=self.user_query, specific_collection=self.specific_collection)
            toolkit = self.tools.get_database_updater_toolkit_w_collection()
        return Task(
            description=desc,
            agent=self.agents.create_database_updater,
            expected_output=DATABASE_UPDATER_EXPECTED_OUTPUT,
            context=context_task_array,
            tools=toolkit,
        )
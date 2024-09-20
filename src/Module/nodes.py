from MultiAgent import *
from SingleAgent import *
# from Frontend import *
from Utils import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *
from Config.constants import *
from pandas import DataFrame
import json
import pandas as pd
from statistics import median
from langgraph.constants import Send
from Config.task_prompts import GLOBAL_TOPIC_RERANKING_PROMPT, LOCAL_TOPIC_RERANKING_PROMPT, RERANKING_PROMPT
from langchain_openai import ChatOpenAI
import Config.constants as const
import cohere

class NodesModularRAG():
    def __init__(self):
        self.retriever = Retriever()
        self.rag_system = MultiAgent_RAG()
        self.batch_size = NODE_BATCH_SIZE
        self.global_retrieval_level = NODE_RETRIEVAL_LEVEL
        self.co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        
    # Methods for the nodes
    def cohere_rerank(self, query: str, docs: List[str]):
        response = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=len(docs),
            return_documents=True,
        )
        return [(result.document.text, int(result.relevance_score*100)) for result in response.results if result.relevance_score > 0.2]
    
    def generate_data_with_score(self, prompt: str, data: List[str]):
        """
        This function is used to generate the data with score given the prompt and data.
        
        Args:
            prompt(str): the prompt to generate the data with score
            data(List[str]): the data to generate the score
            
        Returns:
            data_with_scores(List[Tuple[str, int]]): the data with scores
        """
        llm = ChatOpenAI(
            model=const.MODEL_NAME,
            temperature=const.MODEL_TEMPERATURE,
        )
        scores = llm.with_structured_output(TopicRerankingResult).invoke(prompt).relevant_scores
        while len(scores) != len(data):
            print(f"len(scores): {len(scores)} != len(data): {len(data)}, retry...")
            scores = llm.with_structured_output(TopicRerankingResult).invoke(prompt).relevant_scores
        return list(zip(data, scores))
        
    def prepare_batch_input_for_reranking(self, 
                                        input_list: List[str], 
                                        sub_queries: List[str], 
                                        user_query: str
                                        ) -> List[Dict[str, Any]]:
        """
        This function is used to prepare the batch input for the reranking.
        args:
            input_list: List[str]: the list of data to be reranked
            sub_queries: List[str]: the list of sub queries
            user_query: str: the user query
        returns:
            batches: List[Dict[str, Any]]: the batch input for the reranking
                - user_query: str: the user query
                - sub_queries: List[str]: the list of sub queries
                - batch_data: List[str]: the list of data to be reranked
                - batch_size: int: the size of the batch
        """    
        batches = []
        for i in range(0, len(input_list), self.batch_size):
            batch_input = input_list[i:i+self.batch_size]
            batches.append({
                "user_query": user_query,
                "sub_queries": sub_queries,
                "batch_data": batch_input,
                "batch_size": len(batch_input)
            })
        return batches
    
    def sort_two_descending_list(self, list1: List[Tuple[str, int]], list2: List[Tuple[str, int]])-> List[Tuple[str, int]]:
        """
        This function is used to sort the two lists in descending order.
        args:
            list1: List[Tuple[str, int]]: the list of data to be sorted
            list2: List[Tuple[str, int]]: the list of scores
        returns:
            sorted_list: List[Tuple[str, int]]: the list of data with scores in descending order
        """
        if len(list1) == 0:
            return list2
        if len(list2) == 0:
            return list1
        result = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            if list1[i][1] > list2[j][1]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1
        result.extend(list1[i:])
        result.extend(list2[j:])
        return result
    
    def sort_tuple_desc_and_filter_0_score(self, data: List[Tuple[str, int]])-> Tuple[List[str], List[Tuple[str, int]]]:
        """
        This function is used to sort the data in descending order and filter the data by the score.
        args:
            data: List[Tuple[str, int]]: the list of data to be sorted
        returns:
            sorted_data: List[str]: the list of data sorted in descending order without 0 score
            sorted_data_with_scores: List[Tuple[str, int]]: the list of data with scores in descending order
        """
        filtered_data_with_scores = [data for data in data if data[1] > 0]
        sorted_data_with_scores = sorted(filtered_data_with_scores, key=lambda x: x[1], reverse=True)
        sorted_data = [data for data, _ in sorted_data_with_scores]
        return sorted_data, sorted_data_with_scores
    
    def sort_data_desc_and_filter_0_score(self, data: List[str], scores: List[int])-> Tuple[List[str], List[Tuple[str, int]]]:
        """
        This function is used to sort the data in descending order and filter the data by the score.
        args:
            data: List[str]: the list of data to be sorted
            scores: List[int]: the list of scores
        returns:
            sorted_data: List[str]: the list of data sorted in descending order
            sorted_data_with_scores: List[Tuple[str, int]]: the list of data with scores in descending order
        """
        assert len(data) == len(scores), "The number of data and scores must match, len(data) = %d, len(scores) = %d" % (len(data), len(scores))
        filtered_data_with_scores = [(data, score) for data, score in zip(data, scores) if score > 0]
        sorted_data_with_scores = sorted(filtered_data_with_scores, key=lambda x: x[1], reverse=True)
        sorted_data = [data for data, _ in sorted_data_with_scores]
        return sorted_data, sorted_data_with_scores
    
    def sort_data_desc_and_filter_median_score(self, data: List[str], scores: List[int])-> Tuple[List[str], List[Tuple[str, int]]]:
        """
        This function is used to sort the data in descending order and filter the data under the median score.
        args:
            data: List[str]: the list of data to be sorted
            scores: List[int]: the list of scores
        returns:
            sorted_data: List[str]: the list of data sorted in descending order
            sorted_data_with_scores: List[Tuple[str, int]]: the list of data with scores in descending order
        """
        assert len(data) == len(scores), "The number of data and scores must match, len(data) = %d, len(scores) = %d" % (len(data), len(scores))
        score_median = median(scores)
        filtered_data_with_scores = [(data, score) for data, score in zip(data, scores) if score > score_median]
        sorted_data_with_scores = sorted(filtered_data_with_scores, key=lambda x: x[1], reverse=True)
        sorted_data = [data for data, _ in sorted_data_with_scores]
        return sorted_data, sorted_data_with_scores
    
    # Nodes
    def update_next_query_node(self, state: QueriesState):
        """
        This function is used to update the next query.
        returns:
            user_query(str): the next query
        """
        index_of_current_result = len(state.all_responses)
        print("Query "+str(index_of_current_result)+"/"+str(len(state.dataset_queries)))
        return {
            "user_query": state.dataset_queries[index_of_current_result]
        }
        
    def user_query_classification_node(self, state: UnitQueryState):
        """
        This function is used to classify the user query.
        returns:
            user_query_classification_result(UserQueryClassificationResult): the result of the user query classification
            - needs_retrieval(bool): whether retrieval is needed
            - domain_range_score(int): the score of the domain range
            - justification(str): the justification for the classification decision
            - relevant_keywords(List[str]): the list of relevant keywords
        """
        return {  
            "user_query_classification_result": self.rag_system.user_query_classification_run(user_query=state.user_query)
        } 
    
    def query_process_node(self, state: UnitQueryState):
        """
        This function is used to process the user query.
        returns:
            query_process_result(QueryProcessResult): the result of the query processing
            - original_query(str): the original user query
            - transformed_queries(Optional[List[str]]): the list of transformed queries
            - decomposed_queries(Optional[List[str]]): the list of decomposed queries
        """
        return {  
            "query_process_result": self.rag_system.query_process_run(user_query=state.user_query)
        } 
    
    def sub_queries_classification_node(self, state: UnitQueryState):
        """
        This function is used to classify the sub queries.
        returns:
            sub_queries_classification_result(SubQueriesClassificationResult): the result of the sub queries classification
            - queries(List[str]): the list of sub queries
            - collection_name(List[Optional[str]]): the list of collection names for each sub query
        """
        if state.specific_collection is None:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_without_specification_run(user_query=state.user_query)
            }
        else:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_with_specification_run(user_query=state.user_query, specific_collection=state.specific_collection)
            }
        
    def information_organization_node(self, state: UnitQueryState):
        """
        This function is used to organize the information.
        returns:
            information_organization_result(InformationOrganizationResult): the result of the information organization
            - organized_information(List[str]): the list of organized information
        """
        if state.detailed_search_result:
            retrieve_data = state.detailed_search_result.all_sorted_retrieved_data
            retrieve_data = retrieve_data[:max(len(retrieve_data)//2, 1)]
        else: 
            retrieve_data = []
        return {
            "information_organization_result": self.rag_system.information_organization_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                retrieved_data=retrieve_data
            )
        }
         
    def generation_node(self, state: UnitQueryState):
        """
        This function is used to generate the response.
        returns:
            generation_result(GenerationResult): the result of the generation
            - response(str): the response
        """
        return {
            "generation_result": self.rag_system.generation_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                information=state.information_organization_result.organized_information if state.information_organization_result else [],
                retrieval_needed=state.user_query_classification_result.needs_retrieval
            )
        }
        
    def store_results_node(self, state: QueriesState):
        """
        This function is used to store the results.
        returns:
            all_generation_results(List[str]): the list of all generation results
            all_information_contexts(List[List[str]]): the list of all information contexts
            all_vectordatabase_contexts(List[List[Dict]]): the list of all vectordatabase contexts
        """
        return {
            "all_generation_results": state.generation_result,
            "all_information_contexts": state.information_organization_result.organized_information,
            "all_vectordatabase_contexts": state.detailed_search_result.all_sorted_retrieved_data
        }
        
    def repeat_count_node(self, state: UnitQueryState):
        """
        This function is used to count the repeat times.
        returns:
            repeat_times(int): the repeat times
        """
        repeat_times = state["repeat_times"]
        if repeat_times is None:
            repeat_times = 0
        return {
            "repeat_times": repeat_times + 1
        }

    def database_update_node(self, state: UnitQueryState):
        """
        This function is used to update the database.
        returns:
            state(UnitQueryState): the state of the system
        """
        return self.rag_system.database_update_run()
        
    def global_mapping_node(self, state: RerankingState):
        """
        This function is used to do the global mapping (reranking).
        returns:
            global_mapping_result(List[Tuple[str, int]]): the result of the global mapping
        """
        prompt = GLOBAL_TOPIC_RERANKING_PROMPT.format(
            user_query=state.user_query,
            sub_queries=state.sub_queries,
            batch_data=state.batch_data,
            batch_size=state.batch_size
        )
        data_with_scores = self.generate_data_with_score(prompt, state.batch_data)
        return {"global_mapping_result": data_with_scores}
    
    def local_mapping_node(self, state: RerankingState):
        """
        This function is used to do the local mapping (reranking).
        returns:
            local_mapping_result(List[Tuple[str, int]]): the result of the local mapping
        """
        prompt = LOCAL_TOPIC_RERANKING_PROMPT.format(
            user_query=state.user_query,
            sub_queries=state.sub_queries,
            batch_data=state.batch_data,
            batch_size=state.batch_size
        )
        data_with_scores = self.generate_data_with_score(prompt, state.batch_data)
        return {"local_mapping_result": data_with_scores}
    
    def local_cohere_mapping_node(self, state: UnitQueryState):
        return {"local_mapping_result": self.cohere_rerank(state.user_query, state.retrieval_data)}
        
    def detail_cohere_mapping_node(self, state: UnitQueryState):
        return {"detail_mapping_result": self.cohere_rerank(state.user_query, state.retrieval_data)}    
         
    def detail_mapping_node(self, state: RerankingState):
        """
        This function is used to do the detail mapping (reranking).
        returns:
            detail_mapping_result(List[Tuple[str, int]]): the result of the detail mapping
        """
        prompt = RERANKING_PROMPT.format(
            user_query=state.user_query,
            sub_queries=state.sub_queries,
            batch_data=state.batch_data,
            batch_size=state.batch_size
        )
        data_with_scores = self.generate_data_with_score(prompt, state.batch_data)
        return {"detail_mapping_result": data_with_scores}
    
    def global_reducing_node(self, state: UnitQueryState):
        """
        This function is used to do the global reducing (summarizing and HyDE).
        returns:
            global_reducing_result(GlobalTopicSearchingAndHyDEResult): the result of the global reducing
        """
        sorted_data, sorted_data_with_scores = self.sort_tuple_desc_and_filter_0_score(state.global_mapping_result)
        global_reducing_results = self.rag_system.global_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_data
        )
        print(f"global possible answers: {global_reducing_results.possible_answers}")
        print(f"global scores: {[score for _, score in sorted_data_with_scores]}")
        return {"global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
            possible_answers = global_reducing_results.possible_answers,
            communities_summaries = global_reducing_results.communities_summaries,
            communities_with_scores = sorted_data_with_scores,
        )}
        
    def local_reducing_node(self, state: UnitQueryState):
        """
        This function is used to do the local reducing (summarizing and HyDE).
        returns:
            local_reducing_result(LocalTopicSearchingAndHyDEResult): the result of the local reducing
        """
        if state.use_cohere_reranker:
            sorted_data_with_scores = state.local_mapping_result
            sorted_data = [data for data, _ in sorted_data_with_scores]
        else:
            sorted_data, sorted_data_with_scores = self.sort_tuple_desc_and_filter_0_score(state.local_mapping_result)
        if sorted_data == []:
            return {"local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
                information_with_scores = [],
                information_summaries = [],
                possible_answers = [],
            )}
        sorted_data = sorted_data[:len(sorted_data)//2] # only use half of the data for local reducing
        local_reducing_results = self.rag_system.local_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_data
        )
        print(f"local possible answers: {local_reducing_results.possible_answers}")
        print(f"local scores: {[score for _, score in sorted_data_with_scores]}")
        return {"local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
            possible_answers = local_reducing_results.possible_answers,
            information_summaries = local_reducing_results.information_summaries,
            information_with_scores = sorted_data_with_scores,
        )}    
            
    def detail_reducing_node(self, state: UnitQueryState):
        """
        This function is used to do the detail reducing (summarizing and HyDE).
        returns:
            detail_reducing_result(LocalTopicSearchingAndHyDEResult): the result of the detail reducing
        """
        all_topic_data = []
        if state.use_cohere_reranker:
            all_detail_data_with_scores = state.detail_mapping_result
            all_detail_data = [data for data, _ in all_detail_data_with_scores]
        else:
            all_detail_data, all_detail_data_with_scores = self.sort_tuple_desc_and_filter_0_score(state.detail_mapping_result)
        
        all_data_with_scores = all_detail_data_with_scores.copy()
        if state.local_topic_searching_and_hyde_result:
            all_topic_data.extend([data for data, _ in state.local_topic_searching_and_hyde_result.information_with_scores])
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.local_topic_searching_and_hyde_result.information_with_scores)
        if state.global_topic_searching_and_hyde_result:
            all_topic_data.extend([data for data, _ in state.global_topic_searching_and_hyde_result.communities_with_scores])
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.global_topic_searching_and_hyde_result.communities_with_scores)
        all_data = [data for data, _ in all_data_with_scores]
        print(f"detail scores: {[score for _, score in all_detail_data_with_scores]}")
        print(f"all scores: {[score for _, score in all_data_with_scores]}")
        return {"detailed_search_result": DetailedSearchResult(
            all_sorted_retrieved_data=all_data,
            all_detail_data=all_detail_data,
            all_topic_data=all_topic_data
        )}
        
    def detail_retrieval_node(self, state: UnitQueryState):
        """
        This function is used to retrieve the detail data.
        returns:
            detail_data_result(List[str]): the result of the detail data
        """
        all_queries = [state.user_query]
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)
        if state.local_topic_searching_and_hyde_result:
            all_queries.extend(state.local_topic_searching_and_hyde_result.possible_answers)
        if state.global_topic_searching_and_hyde_result:
            all_queries.extend(state.global_topic_searching_and_hyde_result.possible_answers)    
        retrieved_data = self.retriever.hybrid_retrieve(state.specific_collection, all_queries, 20)
        retrieved_data = [data["content"]+ ", Metadata: " + str(data["metadata"]) for data in retrieved_data]
        return {"retrieval_data": retrieved_data}
        
    def global_retrieval_node(self, state: UnitQueryState):
        all_communities = self.retriever.global_retrieve(self.global_retrieval_level)["community_summaries"]
        return {"retrieval_data": all_communities}
    
    def local_retrieval_node(self, state: UnitQueryState):
        scope_score = state.user_query_classification_result.domain_range_score
        if scope_score >= 61:
            top_searching_entities       = 10
            top_retrieving_entities      = 5
            top_entities                 = 5
            top_chunks                   = 2
            top_communities              = 10
            top_relationships            = 10
            top_searching_relationships  = 10
            top_retrieving_relationships = 5
            top_inside_relationships     = 5
            top_outside_relationships    = 5
        elif scope_score >= 41:
            top_searching_entities       = 10
            top_retrieving_entities      = 5
            top_entities                 = 5
            top_chunks                   = 2
            top_communities              = 10
            top_relationships            = 10
            top_searching_relationships  = 10
            top_retrieving_relationships = 5
            top_inside_relationships     = 5
            top_outside_relationships    = 5
        elif scope_score >= 21:
            top_searching_entities       = 10
            top_retrieving_entities      = 5
            top_entities                 = 5
            top_chunks                   = 2
            top_communities              = 10
            top_relationships            = 10
            top_searching_relationships  = 10
            top_retrieving_relationships = 5
            top_inside_relationships     = 5
            top_outside_relationships    = 5
        else:
            top_searching_entities       = 10
            top_retrieving_entities      = 5
            top_entities                 = 5
            top_chunks                   = 2
            top_communities              = 10
            top_relationships            = 10
            top_searching_relationships  = 10
            top_retrieving_relationships = 5
            top_inside_relationships     = 5
            top_outside_relationships    = 5
            
        # get the results from the local retriever
        relationship_vector_results = self.retriever.local_retrieve_relationship_vector_search(
            query_texts                  = [state.user_query], 
            top_entities                 = top_entities,
            top_chunks                   = top_chunks,
            top_communities              = top_communities,
            top_searching_relationships  = top_searching_relationships,
            top_retrieving_relationships = top_retrieving_relationships,
        )
        relationship_vector_chunks_texts              = relationship_vector_results["chunks_texts"]
        relationship_vector_entity_descriptions       = relationship_vector_results["entity_descriptions"]
        relationship_vector_community_summaries       = relationship_vector_results["community_summaries"]
        relationship_vector_relationship_descriptions = relationship_vector_results["relationship_descriptions"]
        
        entity_keyword_results = self.retriever.local_retrieve_entity_keyword_search(
            keywords                     =    state.user_query_classification_result.relevant_keywords,
            top_chunks                   =    top_chunks,
            top_communities              =    top_communities,
            top_inside_relationships     =    top_inside_relationships,
            top_outside_relationships    =    top_outside_relationships,
        )
        entity_keyword_chunks_texts                      = entity_keyword_results["chunks_texts"]
        entity_keyword_entity_descriptions               = entity_keyword_results["entity_descriptions"]
        entity_keyword_community_summaries               = entity_keyword_results["community_summaries"]
        entity_keyword_inside_relationship_descriptions  = entity_keyword_results["inside_relationship_descriptions"]
        entity_keyword_outside_relationship_descriptions = entity_keyword_results["outside_relationship_descriptions"]
        
        # use the results to run the topic reranking
        all_information = list(set( relationship_vector_chunks_texts
                                  + relationship_vector_community_summaries
                                  + relationship_vector_entity_descriptions
                                  + relationship_vector_relationship_descriptions
                                  + entity_keyword_chunks_texts
                                  + entity_keyword_entity_descriptions
                                  + entity_keyword_community_summaries
                                  + entity_keyword_inside_relationship_descriptions
                                  + entity_keyword_outside_relationship_descriptions))
        
        return {"retrieval_data": all_information}
    
    def local_community_retrieval_node(self, state: UnitQueryState):
        all_communities = self.retriever.local_retrieve_community_vector_search(
            query_texts = [state.user_query],
            top_communities = 20
        )["community_summaries"]
        return {"retrieval_data": all_communities}
    
    def local_hyde_node(self, state: UnitQueryState):
        """
        This function is used to generate the hypothetical document.
        returns:
            hypothetical_document_result(LocalTopicSearchingAndHyDEResult): the result of the hypothetical document
        """
        possible_answers = self.retriever.generate_hypothetical_document(state.user_query)
        return {"local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
            information_with_scores = [],
            information_summaries = [],
            possible_answers = possible_answers,
        )}
     
    # Conditional Edges
    def is_retrieval_needed_edges(self, state: UnitQueryState):
        """
        This function is used to check if retrieval is needed.
        returns:
            retrieval_needed(str): the type of retrieval needed
        """
        if not state.user_query_classification_result.needs_retrieval:
            return "retrieval_not_needed"
        if state.user_query_classification_result.domain_range_score >= 80:
            return "retrieval_needed_for_global_topic_searching"
        else:
            return "retrieval_needed_for_local_topic_searching"
        
    def is_information_organization_needed_edges(self, state: UnitQueryState):
        """
        This function is used to check if information organization is needed.
        returns:
            information_organization_needed(str): the type of information organization needed
        """
        if state.detailed_search_result and state.detailed_search_result.sorted_retrieved_data:
            return "information_organization_needed"
        else:
            return "information_organization_not_needed"
    
    def is_restart_needed_edges(self, state: UnitQueryState):
        """
        This function is used to check if restart is needed.
        returns:
            restart_needed(str): the type of restart needed
        """
        if state.response_audit_result.restart_required and state.repeat_times < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
    def is_dataset_unfinished_edges(self, state: UnitQueryState):
        """
        This function is used to check if the dataset is unfinished.
        returns:
            dataset_unfinished(str): the type of dataset unfinished
        """
        if len(state.dataset_queries) > len(state.all_results):
            return "dataset_unfinished"
        else:
            return "dataset_finished"
    
    def is_global_local_edges(self, state: UnitQueryState):
        """
        This function is used to check if the global or local retriever is needed.
        returns:
            global_local(str): the type of global or local retriever needed
        """
        if state.user_query_classification_result.domain_range_score >= 80:
            return "global retriever"
        else:
            return "local retriever"
        
    def is_local_retrieval_empty_edges(self, state: UnitQueryState):
        if state.local_topic_searching_and_hyde_result:
            return "local_retrieval_not_empty"
        else:
            return "local_retrieval_empty"
        
    # Dispatch Edges
    def local_mapping_dispatch_edges(self, state: UnitQueryState):
        if state.use_cohere_reranker:
            return Send("local_cohere_mapping_node", state)
        batches = self.prepare_batch_input_for_reranking(
            input_list=state.retrieval_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        return [Send("local_mapping_node", RerankingState(**batch)) for batch in batches]
    
    def global_mapping_dispatch_edges(self, state: UnitQueryState):
        batches = self.prepare_batch_input_for_reranking(
            input_list=state.retrieval_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        return [Send("global_mapping_node", RerankingState(**batch)) for batch in batches]
    
    def detail_mapping_dispatch_edges(self, state: UnitQueryState):
        if state.use_cohere_reranker:
            return Send("detail_cohere_mapping_node", state)
        batches = self.prepare_batch_input_for_reranking(
            input_list=state.retrieval_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        return [Send("detail_mapping_node", RerankingState(**batch)) for batch in batches]
    
class NodesMultiAgentRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.rag_system = MultiAgent_RAG(rag_config)
        self.rag_system.tasks.update_tasks(user_query, specific_collection)
    
    def overall_node(self, state: OverallState):
        return self.rag_system.overall_run()
    
class NodesSingleAgentRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.user_query = user_query
        self.specific_collection = specific_collection
        self.rag_system = SingleAgent(rag_config)
        
    def run_node(self, state: SingleState):
        return self.rag_system.run(self.user_query, self.specific_collection)
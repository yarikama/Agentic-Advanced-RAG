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

class NodesModularRAG():
    def __init__(self):
        self.retriever = Retriever()
        self.rag_system = MultiAgent_RAG()
        self.batch_size = NODE_BATCH_SIZE
        self.global_retrieval_level = NODE_RETRIEVAL_LEVEL
        
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
    
    def sort_tuple_desc_and_filter_0_score(self, data: List[Tuple[str, int]])-> List[Tuple[str, int]]:
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
    
    def update_next_query_node(self, state: OverallState):
        """
        This function is used to update the next query.
        returns:
            user_query(str): the next query
        """
        index_of_current_result = len(state.all_results)
        print("index = ", index_of_current_result)
        print("total = ", len(state.dataset_queries))
        return {
            "user_query": state.dataset_queries[index_of_current_result]
        }
        
    def user_query_classification_node(self, state: OverallState):
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
    
    def query_process_node(self, state: OverallState):
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
    
    def sub_queries_classification_node(self, state: OverallState):
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
            
    def global_topic_searching_and_hyde_node(self, state: OverallState):
        """
        This function is used to search for the global topic and do HyDE.
        returns:
            global_topic_searching_and_hyde_result(GlobalTopicSearchingAndHyDEResult): the result of the global topic searching and HyDE
            - communities_with_scores(List[Tuple[str, int]]): the list of communities with scores
            - communities_summaries(List[str]): the list of community summaries
            - possible_answers(List[str]): the list of possible answers
        """
        # Iterate all community
        all_communities = self.retriever.global_retrieve(self.global_retrieval_level)["community_summaries"]
        
        # Assign the community to the agents
        batch_inputs = self.prepare_batch_input_for_reranking(
            user_query=state.user_query,
            input_list=all_communities,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
        )
            
        all_scores = self.rag_system.global_topic_reranking_run_batch_async(node_batch_inputs=batch_inputs).relevant_scores        
        if sum(all_scores) == 0:
            return {
                "global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
                    communities_with_scores = [],
                    communities_summaries = [],
                    possible_answers = [],
                )
            }        
            
        sorted_communities, sorted_communities_with_scores = self.sort_data_desc_and_filter_0_score(all_communities, all_scores)
        # ------------------------------ Topic Searching ------------------------------
        topic_searching_results = self.rag_system.global_topic_searching_run(
            user_query=state.user_query, 
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_communities,
        )
        communities_summaries = topic_searching_results.communities_summaries
        possible_answers = topic_searching_results.possible_answers
        return {
            "global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
                communities_with_scores = sorted_communities_with_scores,
                communities_summaries = communities_summaries,
                possible_answers  = possible_answers,
            )
        } 
        
    def local_topic_searching_and_hyde_node(self, state: OverallState):
        """
        This function is used to search for the local topic and do HyDE.
        returns:
            local_topic_searching_and_hyde_result(LocalTopicSearchingAndHyDEResult): the result of the local topic searching and HyDE
            - information_with_scores(List[Tuple[str, int]]): the list of information with scores
            - information_summaries(List[str]): the list of information summaries
            - possible_answers(List[str]): the list of possible answers
        """
        # judge the top_k, top_community, top_inside_relations from the score
        scope_score = state.user_query_classification_result.domain_range_score
        if scope_score >= 61:
            top_entities              = 5
            top_chunks                = 2
            top_communities           = 10
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        elif scope_score >= 41:
            top_entities              = 7
            top_chunks                = 5
            top_communities           = 5
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        elif scope_score >= 21:
            top_entities              = 10
            top_chunks                = 8
            top_communities           = 4
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        else:
            top_entities              = 12
            top_chunks                = 12
            top_communities           = 3
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
            
        
        # get the results from the local retriever
        relationship_vector_results = self.retriever.local_retrieve_relationship_vector_search(
            query_texts             = [state.user_query], 
            top_entities            = top_entities,
            top_chunks              = top_chunks,
            top_communities         = top_communities,
            top_relationships       = top_relationships
        )
            
        relationship_vector_chunks_texts              = relationship_vector_results["chunks_texts"]
        relationship_vector_entity_descriptions       = relationship_vector_results["entity_descriptions"]
        relationship_vector_community_summaries       = relationship_vector_results["community_summaries"]
        relationship_vector_relationship_descriptions = relationship_vector_results["relationship_descriptions"]
        
        
        entity_keyword_results = self.retriever.local_retrieve_entity_keyword_search(
            keywords                     =    state.user_query_classification_result.relevant_keywords,
            top_entities                 =    top_entities,
            top_chunks                   =    top_chunks,
            top_communities              =    top_communities,
            top_inside_relationships     =    top_inside_relationships,
            top_outside_relationships    =    top_outside_relationships
        )
        entity_keyword_chunks_texts                      = entity_keyword_results["chunks_texts"]
        entity_keyword_entity_descriptions               = entity_keyword_results["entity_descriptions"]
        entity_keyword_community_summaries               = entity_keyword_results["community_summaries"]
        entity_keyword_inside_relationship_descriptions  = entity_keyword_results["inside_relationship_descriptions"]
        entity_keyword_outside_relationship_descriptions = entity_keyword_results["outside_relationship_descriptions"]
        
        # use the results to run the topic reranking
        all_information = list(set(relationship_vector_chunks_texts
                                  +relationship_vector_community_summaries
                                  +relationship_vector_entity_descriptions
                                  +relationship_vector_relationship_descriptions
                                  +entity_keyword_chunks_texts
                                  +entity_keyword_entity_descriptions
                                  +entity_keyword_community_summaries
                                  +entity_keyword_inside_relationship_descriptions
                                  +entity_keyword_outside_relationship_descriptions))

        # Batch the information
        batch_inputs = self.prepare_batch_input_for_reranking(
            input_list=all_information,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        
        all_scores = self.rag_system.local_topic_reranking_run_batch_async(node_batch_inputs=batch_inputs).relevant_scores

        # If the score is 0, use community summary as the information
        if sum(all_scores) == 0:
            community_summaries = self.retriever.local_retrieve_community_vector_search([state.user_query], 10)["community_summaries"]
            all_information = community_summaries
            batch_inputs = self.prepare_batch_input_for_reranking(
                input_list=all_information,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                user_query=state.user_query
            )
            all_scores = self.rag_system.local_topic_reranking_run_batch_async(node_batch_inputs=batch_inputs).relevant_scores
            
        # If the score is still 0, use the possible answers from the user query
        if sum(all_scores) == 0:
            possible_answers = self.retriever.generate_hypothetical_document(state.user_query)
            return {
                "local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
                    information_with_scores = [],
                    information_summaries = [],
                    possible_answers = possible_answers,
                )
            }

        sorted_information, sorted_information_with_scores = self.sort_data_desc_and_filter_0_score(all_information, all_scores)
        
        # after reranking, do topic searching for HyDE
        local_topic_searching_results = self.rag_system.local_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_information
        )
        # return the result (decide the result for generations)
        information_summaries = local_topic_searching_results.information_summaries
        possible_answers = local_topic_searching_results.possible_answers
        return {
            "local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
                information_with_scores = sorted_information_with_scores,
                information_summaries = information_summaries,
                possible_answers = possible_answers,
            )
        }
        
    def detailed_search_node(self, state: OverallState):
        """
        This function is used to do the detailed search.
        returns:
            detailed_search_result(DetailedSearchResult): the result of the detailed search
            - sorted_retrieved_data(List[str]): the list of data sorted in descending order
        """
        all_data_with_scores = []
        all_queries = [state.user_query]
        
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)

        if state.local_topic_searching_and_hyde_result:
            all_queries.extend(state.local_topic_searching_and_hyde_result.possible_answers)
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.local_topic_searching_and_hyde_result.information_with_scores)
            
        if state.global_topic_searching_and_hyde_result:
            all_queries.extend(state.global_topic_searching_and_hyde_result.possible_answers)
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.global_topic_searching_and_hyde_result.communities_with_scores)
            
        retrieved_data = self.retriever.hybrid_retrieve(state.specific_collection, all_queries, 10)

        batch_inputs = self.prepare_batch_input_for_reranking(
            input_list=retrieved_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        
        retrieved_scores = self.rag_system.reranking_run_batch_async(node_batch_inputs=batch_inputs).relevance_scores
        _, sorted_retrieved_data_with_scores = self.sort_data_desc_and_filter_0_score(retrieved_data, retrieved_scores)
        all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, sorted_retrieved_data_with_scores)
        all_data = [data for data, _ in all_data_with_scores]
        concise_all_data = all_data[:max(len(all_data)*3//4, 1)]        
        return {
            "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=concise_all_data)
        }
        
    def information_organization_node(self, state: OverallState):
        """
        This function is used to organize the information.
        returns:
            information_organization_result(InformationOrganizationResult): the result of the information organization
            - organized_information(List[str]): the list of organized information
        """
        return {
            "information_organization_result": self.rag_system.information_organization_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                retrieved_data=state.detailed_search_result.sorted_retrieved_data if state.detailed_search_result else [],
            )
        }
         
    def generation_node(self, state: OverallState):
        """
        This function is used to generate the response.
        returns:
            generation_result(GenerationResult): the result of the generation
            - response(str): the response
        """
        return {
            "generation_result": self.rag_system.generation_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else []
            )
        }
        
    def store_result_for_ragas_node(self, state: OverallState):
        """
        This function is used to store the result for RAGAS.
        returns:
            state(OverallState): the state of the system
        """
        new_answer = state.generation_result
        new_context = state.information_organization_result if state.information_organization_result else ""
        return {
            "user_query":                               None,
            "user_query_classification_result":         None,
            "query_process_result":                     None,
            "sub_queries_classification_result":        None,
            "global_topic_searching_and_hyde_result":   None,
            "local_topic_searching_and_hyde_result":    None,
            "detailed_search_result":                   None,
            "information_organization_result":          None,
            "response_audit_result":                    None,
            "generation_result":                        None,
            "repeat_times":                             0,
            "all_results":                              [new_answer],
            "all_contexts":                             [new_context],
        }
        
    def repeat_count_node(self, state: OverallState):
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

    def database_update_node(self, state: OverallState):
        """
        This function is used to update the database.
        returns:
            state(OverallState): the state of the system
        """
        return self.rag_system.database_update_run()
        
    # Conditional Nodes
    def is_retrieval_needed_cnode(self, state: OverallState):
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
        
    def is_information_organization_needed_cnode(self, state: OverallState):
        """
        This function is used to check if information organization is needed.
        returns:
            information_organization_needed(str): the type of information organization needed
        """
        if state.detailed_search_result and state.detailed_search_result.sorted_retrieved_data:
            return "information_organization_needed"
        else:
            return "information_organization_not_needed"
    
    def is_restart_needed_cnode(self, state: OverallState):
        """
        This function is used to check if restart is needed.
        returns:
            restart_needed(str): the type of restart needed
        """
        if state.response_audit_result.restart_required and state.repeat_times < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
    def is_dataset_unfinished_cnode(self, state: OverallState):
        """
        This function is used to check if the dataset is unfinished.
        returns:
            dataset_unfinished(str): the type of dataset unfinished
        """
        if len(state.dataset_queries) > len(state.all_results):
            return "dataset_unfinished"
        else:
            return "dataset_finished"
    
    def is_global_local_cnode(self, state: OverallState):
        """
        This function is used to check if the global or local retriever is needed.
        returns:
            global_local(str): the type of global or local retriever needed
        """
        if state.user_query_classification_result.domain_range_score >= 70:
            return "global retriever"
        else:
            return "local retriever"
        
    def dispatch_global_mapping_cnode(self, state: OverallState):
        all_communities = self.retriever.global_retrieve(self.global_retrieval_level)["community_summaries"]
        batches = self.prepare_batch_input_for_reranking(
            input_list=all_communities,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        return [Send("global_mapping_node", 
                     {
                        "batch_input": batch,
                        "number_ticket": i,
                     }
                    ) for i, batch in enumerate(batches)]
        
    def global_mapping_node(self, state: RerankingState):
        prompt = GLOBAL_TOPIC_RERANKING_PROMPT.format(
            user_query=state.batch_input["user_query"],
            sub_queries=state.batch_input["sub_queries"],
            batch_data=state.batch_input["input_list"]
        )
        llm = ChatOpenAI(
            model=const.MODEL_NAME,
            temperature=const.MODEL_TEMPERATURE,
        )
        scores = llm.with_structured_output(TopicRerankingResult).invoke(prompt).relevant_scores
        data_with_scores = list(zip(state.batch_input["input_list"], scores))
        return {"global_mapping_result": data_with_scores}
    
    
    def global_reducing_node(self, state: OverallState):
        sorted_data, sorted_data_with_scores = self.sort_data_desc_and_filter_0_score(state.global_mapping_result)
        global_reducing_results = self.rag_system.global_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_data
        )
        return {"global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
            communities_with_scores = sorted_data_with_scores,
            communities_summaries = global_reducing_results.communities_summaries,
            possible_answers = global_reducing_results.possible_answers,
        )}
        
    

    
    def dispatch_local_mapping_cnode(self, state: OverallState):
        scope_score = state.user_query_classification_result.domain_range_score
        if scope_score >= 61:
            top_entities              = 5
            top_chunks                = 2
            top_communities           = 10
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        elif scope_score >= 41:
            top_entities              = 7
            top_chunks                = 5
            top_communities           = 5
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        elif scope_score >= 21:
            top_entities              = 10
            top_chunks                = 8
            top_communities           = 4
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
        else:
            top_entities              = 12
            top_chunks                = 12
            top_communities           = 3
            top_relationships         = 10
            top_inside_relationships  = 5
            top_outside_relationships = 5
            
        
        # get the results from the local retriever
        relationship_vector_results = self.retriever.local_retrieve_relationship_vector_search(
            query_texts             = [state.user_query], 
            top_entities            = top_entities,
            top_chunks              = top_chunks,
            top_communities         = top_communities,
            top_relationships       = top_relationships
        )
            
        relationship_vector_chunks_texts              = relationship_vector_results["chunks_texts"]
        relationship_vector_entity_descriptions       = relationship_vector_results["entity_descriptions"]
        relationship_vector_community_summaries       = relationship_vector_results["community_summaries"]
        relationship_vector_relationship_descriptions = relationship_vector_results["relationship_descriptions"]
        
        entity_keyword_results = self.retriever.local_retrieve_entity_keyword_search(
            keywords                     =    state.user_query_classification_result.relevant_keywords,
            top_entities                 =    top_entities,
            top_chunks                   =    top_chunks,
            top_communities              =    top_communities,
            top_inside_relationships     =    top_inside_relationships,
            top_outside_relationships    =    top_outside_relationships
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
        
        batches = self.prepare_batch_input_for_reranking(
            input_list=all_information,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        
        return [Send("local_mapping_node", 
                     {
                        "batch_input": batch,
                        "number_ticket": i,
                     }
                    ) for i, batch in enumerate(batches)]
    
    def local_mapping_node(self, state: RerankingState):
        prompt = LOCAL_TOPIC_RERANKING_PROMPT.format(
            user_query=state.batch_input["user_query"],
            sub_queries=state.batch_input["sub_queries"],
            batch_data=state.batch_input["input_list"]
        )
        llm = ChatOpenAI(
            model=const.MODEL_NAME,
            temperature=const.MODEL_TEMPERATURE,
        )
        scores = llm.with_structured_output(TopicRerankingResult).invoke(prompt).relevant_scores
        data_with_scores = list(zip(state.batch_input["input_list"], scores))
        return {"local_mapping_result": data_with_scores}
    
    def local_reducing_node(self, state: OverallState):
        sorted_data, sorted_data_with_scores = self.sort_data_desc_and_filter_0_score(state.local_mapping_result)
        local_reducing_results = self.rag_system.local_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_data
        )
        return {"local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
            information_with_scores = sorted_data_with_scores,
            information_summaries = local_reducing_results.information_summaries,
            possible_answers = local_reducing_results.possible_answers,
        )}    
    
    def dispatch_detail_mapping_cnode(self, state: OverallState):
        all_queries = [state.user_query]
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)
        if state.local_topic_searching_and_hyde_result:
            all_queries.extend(state.local_topic_searching_and_hyde_result.possible_answers)
        if state.global_topic_searching_and_hyde_result:
            all_queries.extend(state.global_topic_searching_and_hyde_result.possible_answers)    
        retrieved_data = self.retriever.hybrid_retrieve(state.specific_collection, all_queries, 10)
        batches = self.prepare_batch_input_for_reranking(
            input_list=retrieved_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        return [Send("detail_mapping_node", 
                     {
                        "batch_input": batch,
                        "number_ticket": i,
                     }
                    ) for i, batch in enumerate(batches)]
        
    def detail_mapping_node(self, state: RerankingState):
        prompt = RERANKING_PROMPT.format(
            user_query=state.batch_input["user_query"],
            sub_queries=state.batch_input["sub_queries"],
            batch_data=state.batch_input["input_list"]
        )
        llm = ChatOpenAI(
            model=const.MODEL_NAME,
            temperature=const.MODEL_TEMPERATURE,
        )
        scores = llm.with_structured_output(TopicRerankingResult).invoke(prompt).relevant_scores
        data_with_scores = list(zip(state.batch_input["input_list"], scores))
        return {"detail_mapping_result": data_with_scores}
    
    def detail_reducing_node(self, state: OverallState):
        all_data_with_scores = []
        if state.local_topic_searching_and_hyde_result:
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.local_topic_searching_and_hyde_result.information_with_scores)
        if state.global_topic_searching_and_hyde_result:
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.global_topic_searching_and_hyde_result.communities_with_scores)
        if state.detailed_search_result:
            all_data_with_scores = self.sort_two_descending_list(all_data_with_scores, state.detailed_search_result.sorted_retrieved_data_with_scores)
        all_data = [data for data, _ in all_data_with_scores]
        concise_all_data = all_data[:max(len(all_data)*3//4, 1)]
        return {"detailed_search_result": DetailedSearchResult(sorted_retrieved_data=concise_all_data)}
    
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
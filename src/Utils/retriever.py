import os
from openai import OpenAI
from dotenv import load_dotenv
from .embedder import Embedder
import Config.constants as const
from pymilvus import AnnSearchRequest
from .vector_database import VectorDatabase
from .knowledge_graph_database import KnowledgeGraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from Config.task_prompts import HYDE_PROMPT
from typing import List, Dict, Any, Union, Optional, Set
from Config.output_pydantic import HyDEOutput
import json
import hashlib
load_dotenv()

class Retriever:
    _instance = None

    @classmethod
    def get_instance(
        cls, 
        vectordatabase: Optional[VectorDatabase] = None,
        graphdatabase: Optional[KnowledgeGraphDatabase] = None, 
        embedder: Optional[Embedder] = None
    ) -> 'Retriever':
        if cls._instance is None:
            cls._instance = cls(vectordatabase, graphdatabase, embedder)
        return cls._instance

    def __init__(
        self, 
        vectordatabase: Optional[VectorDatabase] = None,
        graphdatabase: Optional[KnowledgeGraphDatabase] = None, 
        embedder: Optional[Embedder] = None
    ) -> 'Retriever':
        
        self.embedder = embedder if embedder else Embedder()
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.graphdatabase = graphdatabase if graphdatabase else KnowledgeGraphDatabase()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        print("Retriever initialized")
        
    def generate_hypothetical_document(self, query: str) -> List[str]:
        prompt = HYDE_PROMPT.format(query=query)
        hyde_llm = ChatOpenAI(
            model=const.MODEL_NAME,
            temperature=0.5,
        )
        response = hyde_llm.with_structured_output(HyDEOutput).invoke(prompt)
        return response.possible_answers

    def dense_search_request(
        self, 
        dense_query_vectors: Union[List[float], List[List[float]]], 
        field_name: str,
        top_k: int = const.TOP_K
    ) -> List[Dict[str, Any]]:
        
        if const.IS_GPU_INDEX:
            dense_search_param = {
                "data": dense_query_vectors,
                "anns_field": field_name,
                "param": {
                    "metric_type": "L2",
                    "params": {
                        "itopk_size": 128,
                        "search_width": 4,
                        "min_iterations": 0,
                        "max_iterations": 0,
                        "team_size": 0
                    },
                },
                "limit": top_k
            }
        else:
            dense_search_param = {
                "data": dense_query_vectors,
                "anns_field": field_name,
                "param": {
                    "metric_type": "COSINE",
                    "params": {"ef": 100}
                },
                "limit": top_k
            }

        dense_search_request = AnnSearchRequest(**dense_search_param)
        return dense_search_request 

    def sparse_search_request(
        self, 
        sparse_query_vectors: Union[List[float], List[List[float]]], 
        field_name: str, 
        top_k: int = const.TOP_K
    ) -> List[Dict[str, Any]]:
        
        sparse_search_param = {
            "data": sparse_query_vectors,
            "anns_field": field_name,
            "param": {
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.1}                
            },
            "limit": top_k
        }

        sparse_search_request = AnnSearchRequest(**sparse_search_param)
        return sparse_search_request
    
    def hybrid_search_request(self, dense_query_vectors: Union[List[float], List[List[float]]], sparse_query_vectors: Union[List[float], List[List[float]]]):
        dense_search_request = self.dense_search_request(dense_query_vectors, "dense_vector")
        sparse_search_request = self.sparse_search_request(sparse_query_vectors, "sparse_vector")
        hybrid_search_requests = [dense_search_request, sparse_search_request]
        return hybrid_search_requests

    def hybrid_retrieve(
        self, 
        collection_name: str, 
        query_texts: List[str], 
        top_k: int = const.TOP_K, 
        alpha: float = const.ALPHA, 
        isHyDE: bool = False
    ) -> List[Dict[str, Any]]:
        """
        This is for similar multiple queries searching, the result is a deduplicated list of documents
        Args:
            collection_name: str -> The name of the collection
            query_texts: List[str] -> The list of queries
            top_k: int -> The number of the top results
            alpha: float -> The weight of the dense search
            isHyDE: bool -> Whether to use HyDE to generate hypothetical documents
        Returns:
            List[Dict[str, Any]] -> The deduplicated search results
                - content: str -> The content of the document
                - metadata: Dict[str, Any] -> The metadata of the document
        """
        dense_queries = self.embedder.embed_dense(query_texts)
        sparse_queries = self.embedder.embed_sparse(collection_name, query_texts)
        
        if isHyDE:
            hypothetical_docs = [self.generate_hypothetical_document(query) for query in query_texts]
            hyde_dense_queries = self.embedder.embed_dense(hypothetical_docs)
            hyde_sparse_queries = self.embedder.embed_sparse(collection_name, hypothetical_docs)            
            dense_queries = hyde_dense_queries + dense_queries
            sparse_queries = hyde_sparse_queries + sparse_queries
        
        # Separate dense vectors for empty sparse vectors
        dense_only_queries = []
        filtered_dense_queries = []
        filtered_sparse_queries = []
        
        for dense, sparse in zip(dense_queries, sparse_queries):
            if isinstance(sparse, tuple) and len(sparse) == 2:
                indices, values = sparse
                if len(indices) == 0 or len(values) == 0:
                    dense_only_queries.append(dense)
                else:
                    filtered_dense_queries.append(dense)
                    filtered_sparse_queries.append(sparse)
            else:
                dense_only_queries.append(dense)
        
        batch_results_from_queries = []
        if filtered_dense_queries:
            hybrid_search_requests = self.hybrid_search_request(filtered_dense_queries, filtered_sparse_queries)
            batch_results_from_queries = self.vectordatabase.hybrid_search(
                collection_name, hybrid_search_requests, "weighted", [1 - alpha, alpha], top_k
            )
        
        # Perform dense search for queries with empty sparse vectors
        if dense_only_queries:
            dense_search_request = self.dense_search_request(dense_only_queries, "dense_vector")
            dense_results = self.vectordatabase.search(collection_name, dense_search_request, top_k)
            batch_results_from_queries.extend(dense_results)
        
        return self.deduplicate_results(batch_results_from_queries)

    def deduplicate_results(self, batch_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        This function is used to deduplicate the search results
        Args:
            batch_results: List[List[Dict[str, Any]]] -> The search results from the queries
        Returns:
            unique_results: List[Dict[str, Any]] -> The deduplicated search results
        """
        seen_hashes = set()
        unique_results = []
        
        for query_results in batch_results:
            for result in query_results:
                content = result['content']
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_results.append(result)
        
        return unique_results 
    
    def dense_retrieve(self, collection_name: str, query_texts: List[str], top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        dense_queries = self.embedder.embed_dense(query_texts)
        dense_search_request = self.dense_search_request(dense_queries, "dense_vector")
        batch_results_from_queries = self.vectordatabase.search(collection_name, dense_search_request, top_k)
        return self.deduplicate_results(batch_results_from_queries)
        
    def global_retrieve(self, level: int = const.NODE_RETRIEVAL_LEVEL) -> str:
        """
        This function just return all the communities in the graph database.
        Args:
            level: the level of the community to retrieve
        Returns:
            Dict[str, List[str]]
                - community_summaries: List[str]
        """
        all_community_summaries = self.graphdatabase.dictionary_query_result(
        """
        MATCH (c:__Community__)
        WHERE c.level = $level
        RETURN collect(c.summary) AS community_summaries
        """,
            params={"level": level},
        )
        return all_community_summaries
    
    def local_retrieve_entity_vector_search(
        self, 
        query_texts: List[str],
        top_searching_entities: int = const.NEO4J_TOP_ENTITIES,
        top_retrieving_entities: int = const.NEO4J_TOP_ENTITIES,
        top_chunks: int = const.NEO4J_TOP_CHUNKS,
        top_communities: int = const.NEO4J_TOP_COMMUNITIES,
        top_outside_relationships: int = const.NEO4J_TOP_OUTSIDE_RELATIONSHIPS,
        top_inside_relationships: int = const.NEO4J_TOP_INSIDE_RELATIONSHIPS
    ) -> Dict[str, Any]:
        """
        This function is used to retrieve the local search results from the graph database.
        Args:
            query_texts: List[str] -> Input all the queries
            top_searching_entities: int -> the number of the top entities to search
            top_retrieving_entities: int -> the number of the top entities to retrieve
            top_chunks: int -> the number of the top chunks
            top_communities: int -> the number of the top communities
            top_outside_relationships: int -> the number of the top outside relationships
            top_inside_relationships: int -> the number of the top inside relationships
        Returns:
            result: Dict[str, List[str]]
                - entity_names: List[str]
                - entity_descriptions: List[str]
                - chunks_texts: List[str]
                - community_summaries: List[str]
                - inside_relationship_descriptions: List[str]
                - outside_relationship_descriptions: List[str]
        """    
        queries_vectors = self.embedder.embed_dense(query_texts)
        result = self.graphdatabase.dictionary_query_result("""
        // Using the query vectors to retrieve the entities
        UNWIND $queries_vectors AS query_vector
        CALL db.index.vector.queryNodes('entity_description_vector_index', $topSearchingEntities, query_vector) YIELD node
        WITH COLLECT(DISTINCT node) AS retrieved_entities
        
        // Chunk - Entity Mapping
        WITH COLLECT {
            UNWIND retrieved_entities as entity
            MATCH (entity)<-[:HAS_ENTITY]->(chunk:__Chunk__)
            WITH chunk, count(distinct entity) as freq
            WITH DISTINCT {chunkText: chunk.text, freq: freq} AS chunkFreqPair
            WHERE chunkFreqPair.chunkText IS NOT NULL AND chunkFreqPair.chunkText <> ''
            RETURN chunkFreqPair.chunkText AS chunkText
            ORDER BY chunkFreqPair.freq DESC
            LIMIT $topChunks
        } AS chunks_texts,
        
        // Entity - Report Mapping
        COLLECT {
            UNWIND retrieved_entities as entity
            MATCH (entity)-[:IN_COMMUNITY]->(community:__Community__)
            WITH DISTINCT {communitySummary: community.summary, rank: community.rank, weight: community.weight} AS communityRankWeightPair
            WHERE communityRankWeightPair.communitySummary IS NOT NULL AND communityRankWeightPair.communitySummary <> ''
            RETURN communityRankWeightPair.communitySummary
            ORDER BY communityRankWeightPair.rank, communityRankWeightPair.weight DESC
            LIMIT $topCommunities
        } AS community_summaries,
        
        // Inside Relationships 
        COLLECT {
            UNWIND retrieved_entities as inside_entity
            MATCH (inside_entity_1)-[inside_relationship:RELATED]-(inside_entity_2) 
            WHERE inside_entity_2 IN retrieved_entities
            WITH DISTINCT {descriptionText: inside_relationship.description, rank: inside_relationship.rank, weight: inside_relationship.weight} AS inside_relationship_rank_weight_pair
            WHERE inside_relationship_rank_weight_pair.descriptionText IS NOT NULL AND inside_relationship_rank_weight_pair.descriptionText <> ''
            RETURN inside_relationship_rank_weight_pair.descriptionText
            ORDER BY inside_relationship_rank_weight_pair.rank, inside_relationship_rank_weight_pair.weight DESC 
            LIMIT $topInsideRelationships
        } as inside_relationship_descriptions,
        
        // Outside Relationships 
        COLLECT {
            UNWIND retrieved_entities as inside_entity
            MATCH (inside_entity)-[outside_relationship:RELATED]-(outside_entity) 
            WHERE NOT outside_entity IN retrieved_entities
            WITH DISTINCT {descriptionText: outside_relationship.description, rank: outside_relationship.rank, weight: outside_relationship.weight} AS outside_relationship_rank_weight_pair
            WHERE outside_relationship_rank_weight_pair.descriptionText IS NOT NULL AND outside_relationship_rank_weight_pair.descriptionText <> ''
            RETURN outside_relationship_rank_weight_pair.descriptionText
            ORDER BY outside_relationship_rank_weight_pair.rank, outside_relationship_rank_weight_pair.weight DESC 
            LIMIT $topOutsideRelationships
        } as outside_relationship_descriptions,
        
        // Entities description
        COLLECT {
            UNWIND retrieved_entities as entity
            WITH entity
            WHERE entity.description IS NOT NULL AND entity.description <> ''
            RETURN entity.description AS descriptionText
            LIMIT $topRetrievingEntities
        } as entity_descriptions,
        
        // Entities name
        COLLECT {
            UNWIND retrieved_entities as entity
            WITH entity
            WHERE entity.name IS NOT NULL AND entity.name <> ''
            RETURN entity.name AS name
            LIMIT $topRetrievingEntities
        } as entity_names
        
        RETURN entity_names, entity_descriptions, chunks_texts, community_summaries, inside_relationship_descriptions, outside_relationship_descriptions
        """, 
        params={
            "queries_vectors": queries_vectors,
            "topSearchingEntities": top_searching_entities,
            "topRetrievingEntities": top_retrieving_entities,
            "topChunks": top_chunks,
            "topCommunities": top_communities,
            "topOutsideRelationships": top_outside_relationships,
            "topInsideRelationships": top_inside_relationships
        })
        
        return result
    
    def local_retrieve_entity_keyword_search(self, 
                    keywords: List[str],
                    top_chunks: int = const.NEO4J_TOP_CHUNKS,
                    top_communities: int = const.NEO4J_TOP_COMMUNITIES,
                    top_outside_relationships: int = const.NEO4J_TOP_OUTSIDE_RELATIONSHIPS,
                    top_inside_relationships: int = const.NEO4J_TOP_INSIDE_RELATIONSHIPS) -> Dict[str, Any]:
        """
        This function is used to retrieve the local search results from the graph database.
        Args:
            keywords: List[str] -> Input all the keywords
            top_chunks: int -> the number of the top chunks
            top_communities: int -> the number of the top communities
            top_outside_relationships: int -> the number of the top outside relationships
            top_inside_relationships: int -> the number of the top inside relationships
        Returns:
            result: Dict[
            result: Dict[str, List[str]]
                - entity_names: List[str]
                - entity_descriptions: List[str]
                - chunks_texts: List[str]
                - community_summaries: List[str]
                - inside_relationship_descriptions: List[str]
                - outside_relationship_descriptions: List[str]
        """    
        
        result = self.graphdatabase.dictionary_query_result("""
        // Chunk - Entity Mapping
        CALL db.index.fulltext.queryNodes("entity_name_index", apoc.text.join($keywords, " OR ")) YIELD node, score
        WITH COLLECT(DISTINCT node) AS retrieved_entities
        
        // Chunk - Entity Mapping
        WITH COLLECT {
            UNWIND retrieved_entities as entity
            MATCH (entity)<-[:HAS_ENTITY]->(chunk:__Chunk__)
            WITH chunk, count(distinct entity) as freq
            WITH DISTINCT {chunkText: chunk.text, freq: freq} AS chunkFreqPair
            WHERE chunkFreqPair.chunkText IS NOT NULL AND chunkFreqPair.chunkText <> ''
            RETURN chunkFreqPair.chunkText AS chunkText
            ORDER BY chunkFreqPair.freq DESC
            LIMIT $topChunks
        } AS chunks_texts,
        
        // Entity - Report Mapping
        COLLECT {
            UNWIND retrieved_entities as entity
            MATCH (entity)-[:IN_COMMUNITY]->(community:__Community__)
            WITH DISTINCT {communitySummary: community.summary, rank: community.rank, weight: community.weight} AS communityRankWeightPair
            WHERE communityRankWeightPair.communitySummary IS NOT NULL AND communityRankWeightPair.communitySummary <> ''
            RETURN communityRankWeightPair.communitySummary
            ORDER BY communityRankWeightPair.rank, communityRankWeightPair.weight DESC
            LIMIT $topCommunities
        } AS community_summaries,
        
        // Inside Relationships 
        COLLECT {
            UNWIND retrieved_entities as inside_entity
            MATCH (inside_entity_1)-[inside_relationship:RELATED]-(inside_entity_2) 
            WHERE inside_entity_2 IN retrieved_entities
            WITH DISTINCT {descriptionText: inside_relationship.description, rank: inside_relationship.rank, weight: inside_relationship.weight} AS inside_relationship_rank_weight_pair
            WHERE inside_relationship_rank_weight_pair.descriptionText IS NOT NULL AND inside_relationship_rank_weight_pair.descriptionText <> ''
            RETURN inside_relationship_rank_weight_pair.descriptionText
            ORDER BY inside_relationship_rank_weight_pair.rank, inside_relationship_rank_weight_pair.weight DESC 
            LIMIT $topInsideRelationships
        } as inside_relationship_descriptions,
        
        // Outside Relationships 
        COLLECT {
            UNWIND retrieved_entities as inside_entity
            MATCH (inside_entity)-[outside_relationship:RELATED]-(outside_entity) 
            WHERE NOT outside_entity IN retrieved_entities
            WITH DISTINCT {descriptionText: outside_relationship.description, rank: outside_relationship.rank, weight: outside_relationship.weight} AS outside_relationship_rank_weight_pair
            WHERE outside_relationship_rank_weight_pair.descriptionText IS NOT NULL AND outside_relationship_rank_weight_pair.descriptionText <> ''
            RETURN outside_relationship_rank_weight_pair.descriptionText
            ORDER BY outside_relationship_rank_weight_pair.rank, outside_relationship_rank_weight_pair.weight DESC 
            LIMIT $topOutsideRelationships
        } as outside_relationship_descriptions,
        
        // Entities description
        COLLECT {
            UNWIND retrieved_entities as entity
            WITH entity
            WHERE entity.description IS NOT NULL AND entity.description <> ''
            RETURN entity.description AS descriptionText
        } as entity_descriptions,
        
        // Entities name
        COLLECT {
            UNWIND retrieved_entities as entity
            WITH entity
            WHERE entity.name IS NOT NULL AND entity.name <> ''
            RETURN entity.name AS name
        } as entity_names
        
        RETURN entity_names, entity_descriptions, chunks_texts, community_summaries, inside_relationship_descriptions, outside_relationship_descriptions
        """, 
        params={
            "keywords": keywords,
            "topChunks": top_chunks,
            "topCommunities": top_communities,
            "topOutsideRelationships": top_outside_relationships,
            "topInsideRelationships": top_inside_relationships,
        })
        
        return result    
    
    def local_retrieve_relationship_vector_search(self, 
                                                query_texts: List[str],
                                                top_entities: int = const.NEO4J_TOP_ENTITIES,
                                                top_chunks: int = const.NEO4J_TOP_CHUNKS,
                                                top_communities: int = const.NEO4J_TOP_COMMUNITIES,
                                                top_searching_relationships: int = const.NEO4J_TOP_RELATIONSHIPS,
                                                top_retrieving_relationships: int = const.NEO4J_TOP_RELATIONSHIPS) -> Dict[str, Any]:
        """
        This function is used to retrieve the local search results from the graph database.
        Args:
            query_texts: List[str] -> Input all the queries
            top_entities: int -> the number of the top entities
            top_chunks: int -> the number of the top chunks
            top_communities: int -> the number of the top communities
            top_searching_relationships: int -> the number of the top relationships to search
            top_retrieving_relationships: int -> the number of the top relationships to retrieve
        Returns:
            result: Dict[str, List[str]]
                - entity_descriptions: List[str]
                - entity_names: List[str]
                - relationship_descriptions: List[str]
                - community_summaries: List[str]
                - chunks_texts: List[str]
        """    
        queries_vectors = self.embedder.embed_dense(query_texts)
        result = self.graphdatabase.dictionary_query_result("""
        // Using the query vectors to retrieve the relationships
        UNWIND $queries_vectors AS query_vector
        CALL db.index.vector.queryRelationships('relationship_description_vector_index', $topSearchingRelationships, query_vector) YIELD relationship
        WITH COLLECT(DISTINCT relationship) AS retrieved_relationships
        WITH COLLECT {
            UNWIND retrieved_relationships as relationship
            WITH DISTINCT {relationshipDescription: relationship.description, rank: relationship.rank, weight: relationship.weight} AS relationshipRankWeightPair
            WHERE relationshipRankWeightPair.relationshipDescription IS NOT NULL AND relationshipRankWeightPair.relationshipDescription <> ''
            RETURN relationshipRankWeightPair.relationshipDescription AS relationshipDescription
            ORDER BY relationshipRankWeightPair.rank, relationshipRankWeightPair.weight DESC
            LIMIT $topRetrievingRelationships
        } AS relationship_descriptions,
        
        // Chunk - Entity Mapping
        COLLECT {
            UNWIND retrieved_relationships as relationship
            MATCH (entity)<-[relationship:RELATED]->()
            WITH DISTINCT entity
            MATCH (entity)<-[all_relationship:RELATED]-()
            WITH entity, all_relationship
            WHERE all_relationship IN retrieved_relationships
            WITH entity, COUNT(all_relationship) AS freq
            RETURN entity
            ORDER BY freq DESC
            LIMIT $topEntities
        } AS entities, retrieved_relationships
        
        WITH *
        WITH COLLECT {
            UNWIND entities as entity
            WITH entity
            WHERE entity.description IS NOT NULL AND entity.description <> ''
            RETURN entity.description AS descriptionText
        } AS entity_descriptions,
        
        COLLECT {
            UNWIND entities as entity
            WITH entity
            WHERE entity.name IS NOT NULL AND entity.name <> ''
            RETURN entity.name AS name
        } AS entity_names, 
        
        COLLECT {
            UNWIND entities as entity
            MATCH (entity)<-[:HAS_ENTITY]->(chunk:__Chunk__)
            WITH chunk, count(distinct entity) as freq
            WITH DISTINCT {chunkText: chunk.text, freq: freq} AS chunkFreqPair
            WHERE chunkFreqPair.chunkText IS NOT NULL AND chunkFreqPair.chunkText <> ''
            RETURN chunkFreqPair.chunkText AS chunkText
            ORDER BY chunkFreqPair.freq DESC
            LIMIT $topChunks
        } AS chunks_texts,
        
        COLLECT {
            UNWIND entities as entity
            MATCH (entity)-[:IN_COMMUNITY]->(community:__Community__)
            WITH DISTINCT {communitySummary: community.summary, rank: community.rank, weight: community.weight} AS communityRankWeightPair
            WHERE communityRankWeightPair.communitySummary IS NOT NULL AND communityRankWeightPair.communitySummary <> ''
            RETURN communityRankWeightPair.communitySummary
            ORDER BY communityRankWeightPair.rank, communityRankWeightPair.weight DESC
            LIMIT $topCommunities
        } AS community_summaries, relationship_descriptions
        
        RETURN entity_descriptions, entity_names, relationship_descriptions, community_summaries, chunks_texts
        """, 
        params={
            "topRetrievingRelationships": top_retrieving_relationships,
            "topSearchingRelationships": top_searching_relationships,
            "queries_vectors": queries_vectors,
            "topEntities": top_entities,
            "topChunks": top_chunks,
            "topCommunities": top_communities,
        })
        
        return result
    
    def local_retrieve_community_vector_search(self, 
                                                query_texts: List[str],
                                                top_searching_communities: int = const.NEO4J_TOP_COMMUNITIES,
                                                top_retrieving_communities: int = const.NEO4J_TOP_COMMUNITIES,
                                              ) -> Dict[str, Any]:
        """
        This function is used to retrieve the local search results from the graph database.
        Args:
            query_texts: List[str] -> Input all the queries
            top_searching_communities: int -> the number of the top communities to search
            top_retrieving_communities: int -> the number of the top communities to retrieve
        Returns:
            result: Dict[str, List[str]]
                - community_summaries: List[str]
                
        """     
        
        queries_vectors = self.embedder.embed_dense(query_texts)
        result = self.graphdatabase.dictionary_query_result("""
        // Using the query vectors to retrieve the communities
        UNWIND $queries_vectors AS query_vector
        CALL db.index.vector.queryNodes('community_summary_vector_index', $topSearchingCommunities, query_vector) YIELD node
        WITH COLLECT(DISTINCT node) AS retrieved_communities
        WITH COLLECT {
            UNWIND retrieved_communities as community
            RETURN community.summary AS communitySummary
            ORDER BY community.rank DESC
            LIMIT $topRetrievingCommunities
        } AS community_summaries
        RETURN community_summaries
        """, 
        params={
            "queries_vectors": queries_vectors,
            "topSearchingCommunities": top_searching_communities,
            "topRetrievingCommunities": top_retrieving_communities,
        })
        return result
               

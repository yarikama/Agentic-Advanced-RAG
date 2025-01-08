import os
import time
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, Result
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
import Config.constants as const

load_dotenv()

class KnowledgeGraphDatabase:
    def __init__(self):
        self.neo4j_uri = const.NEO4J_URI
        self.neo4j_username = const.NEO4J_USERNAME
        self.neo4j_password = const.NEO4J_PASSWORD
        self.neo4j_database = const.NEO4J_DATABASE
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )       
        print("GraphDatabase initialized.")
        
    def dictionary_query_result(
        self, 
        cypher: str,
        params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query and return the result as a dictionary.
        
        Args:
            cypher (str): The Cypher query to execute.
            params (Dict[str, Any]): The parameters to pass to the query.
                - top_k (int): The number of results to return.
                - ...
        Returns:
            Dict[str, Any]: The result of the query.
        """
        result = self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_ = lambda result: [record.data() for record in result]
        )
        return result[0]
    
    def delete_all_schema(self):
        """
        Delete all schema from the database.
        Includes constraints, indexes, and everything that was created by apoc.
        """
        self.driver.execute_query("CALL apoc.schema.assert({}, {})")
        print("All schema deleted.")

    def delete_all_data(self):
        """
        Delete all data from the database.
        """
        self.driver.execute_query("MATCH (n) DETACH DELETE n")
        print("All data deleted.")
        
    def delete_all_indexes(self):
        """
        Delete all indexes from the database.
        """
        delete_indexes = ("""
        DROP INDEX entity_name_index IF EXISTS;
        DROP INDEX entity_description_vector_index IF EXISTS;
        DROP INDEX relationship_description_vector_index IF EXISTS;
        DROP INDEX community_summary_vector_index IF EXISTS;
        """).split(";")
        for delete_index in delete_indexes:
            if len((delete_index or "").strip()) > 0:
                print(delete_index)
                self.driver.execute_query(delete_index)
        print("All indexes deleted.")

    def delete_all(self):
        """
        Delete all data and schema from the database.
        """
        self.delete_all_indexes()
        self.delete_all_schema()
        self.delete_all_data()
        print("All data and schema deleted.")
    
    def batched_import(self, statement, df, batch_size=1000):
        """
        Import a dataframe into Neo4j using a batched approach.
        
        Args:
            statement (str): The Cypher statement to execute.
            df (pd.DataFrame): The dataframe to import.
            batch_size (int): The number of rows to import in each batch.
            
        Returns:
            int: The number of rows imported.
        """
        start_s = time.time()
        total = len(df)
        for start in range(0,total, batch_size):
            batch = df.iloc[start: min(start+batch_size,total)]
            result = self.driver.execute_query(
                        "UNWIND $rows AS value " + statement, 
                        rows=batch.to_dict('records'),
                        database_=self.neo4j_database
                    )
            print(result.summary.counters)
        print(f'{total} rows in { time.time() - start_s} s.')    
        return total
    
    def create_constraints(self):
        """
        Create constraints for the transformation between graph_rag and neo4j to ensure uniqueness.
        There are 7 constraints in total.
        - chunk_id
        - document_id
        - community_id
        - entity_id
        - entity_name
        - covariate_title
        - relationship_id
        """
        constraint_statements = """
        create constraint chunk_id          if not exists for (chunk:__Chunk__)                 require chunk.id            is unique;
        create constraint document_id       if not exists for (document:__Document__)           require document.id         is unique;
        create constraint community_id      if not exists for (community:__Community__)         require community.id        is unique;
        create constraint entity_id         if not exists for (entity:__Entity__)               require entity.id           is unique;
        create constraint entity_name       if not exists for (entity:__Entity__)               require entity.name         is unique;
        create constraint covariate_title   if not exists for (covariate:__Covariate__)         require covariate.title     is unique;
        create constraint relationship_id   if not exists for ()-[relationship:RELATED]->()     require relationship.id     is unique;
        """.split(";")

        for constraint_statement in constraint_statements:
            if len((constraint_statement or "").strip()) > 0:
                print(constraint_statement)
                self.driver.execute_query(constraint_statement)
                
        print("Constraints created.")
    
    def import_documents(self, data_path):
        document_df = pd.read_parquet(f'{data_path}/create_final_documents.parquet')        
        document_statement = """
        // SET DOCUMENT AND ITS PROPERTIES
        MERGE (document:__Document__ {id:value.id})
        SET document += value {.title, .raw_content}
        """
        self.batched_import(document_statement, document_df)
        print("Documents imported.")

    def import_text_units(self, data_path):
        text_df = pd.read_parquet(f'{data_path}/create_final_text_units.parquet')        
        text_statement = """
        // SET CHUNK AND ITS PROPERTIES
        MERGE (chunk:__Chunk__ {id:value.id})
        SET chunk += value {.text, .n_tokens}
        
        // ADD RELATIONSHIPS BETWEEN CHUNKS AND DOCUMENTS
        WITH chunk, value
        UNWIND value.document_ids AS document_id
        MATCH (document:__Document__ {id:document_id})
        MERGE (chunk)-[:PART_OF]->(document)
        """
        self.batched_import(text_statement, text_df)
        print("Text Units imported.")

    def import_entities(self, data_path):
        entity_df = pd.read_parquet(f'{data_path}/create_final_entities.parquet')
        entity_statement = """
        // SET ENTITY AND ITS PROPERTIES
        MERGE (entity:__Entity__ {id:value.id})
        SET entity += value {.human_readable_id, .description, name:replace(value.name,'"','')}
        
        // ADD VECTOR PROPERTY TO ENTITY
        WITH entity, value
        CALL db.create.setNodeVectorProperty(entity, "description_embedding", value.description_embedding)
        CALL apoc.create.addLabels(entity, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
        
        // ADD RELATIONSHIPS BETWEEN CHUNKS AND ENTITIES
        UNWIND value.text_unit_ids AS text_unit_id
        MATCH (chunk:__Chunk__ {id:text_unit_id})
        MERGE (chunk)-[:HAS_ENTITY]->(entity)
        """
        self.batched_import(entity_statement, entity_df)
        print("Entities imported.")
        
    def import_relationships(self, data_path):
        relationship_df = pd.read_parquet(f'{data_path}/create_final_relationships.parquet')
        relationship_statement = """
        // SET RELATIONSHIP AND ITS PROPERTIES
        MATCH (source_entity:__Entity__ {name:replace(value.source,'"','')})
        MATCH (target_entity:__Entity__ {name:replace(value.target,'"','')})
        MERGE (source_entity)-[relationship:RELATED {id: value.id}]->(target_entity)
        SET relationship += value {.rank, .weight, .human_readable_id, .description, .text_unit_ids}
        
        // ADD VECTOR PROPERTY TO RELATIONSHIP
        WITH relationship, value
        CALL db.create.setRelationshipVectorProperty(relationship, "description_embedding", value.description_embedding)
        RETURN count(*) as createdRelationships
        """
        self.batched_import(relationship_statement, relationship_df)
        print("Relationships imported.")
        
    def import_communities(self, data_path):
        community_df = pd.read_parquet(f'{data_path}/create_final_communities.parquet')
        community_statement = """
        // SET COMMUNITY AND ITS PROPERTIES
        MERGE (community:__Community__ {id:value.id})
        SET community += value {.level, .title}
        
        // ADD RELATIONSHIPS BETWEEN CHUNKS AND COMMUNITIES (Complexity too high)
        /*
        WITH *
        UNWIND value.text_unit_ids as text_unit_id
        MATCH (chunk:__Chunk__ {id:text_unit_id})
        MERGE (community)-[:HAS_CHUNK]->(chunk)
        */
        
        // SET RELATIONSHIPS BETWEEN ENTITIES AND COMMUNITIES
        WITH *
        UNWIND value.relationship_ids as relationship_id
        MATCH (source_entity:__Entity__)-[:RELATED {id:relationship_id}]->(target_entity:__Entity__)
        MERGE (source_entity)-[:IN_COMMUNITY]->(community)
        MERGE (target_entity)-[:IN_COMMUNITY]->(community)
        RETURN count(distinct community) as createdCommunities
        """
        self.batched_import(community_statement, community_df)
        print("Communities imported.")

    def import_community_reports(self, data_path):
        community_report_df = pd.read_parquet(f'{data_path}/create_final_community_reports.parquet')
        community_report_statement = """
        // SET COMMUNITY REPORT AND ITS PROPERTIES
        MERGE (community:__Community__ {id:value.community})
        SET community += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}
        
        // ADD VECTOR PROPERTY TO COMMUNITY REPORT
        WITH community, value
        CALL db.create.setNodeVectorProperty(community, "summary_embedding", value.summary_embedding)
        
        // ADD RELATIONSHIPS BETWEEN COMMUNITIES AND FINDINGS
        WITH community, value
        UNWIND range(0, size(value.findings)-1) AS finding_idx
        WITH community, value, finding_idx, value.findings[finding_idx] as value_finding
        MERGE (community)-[:HAS_FINDING]->(finding:Finding {id:finding_idx})
        SET finding += value_finding
        """
        self.batched_import(community_report_statement, community_report_df)
        print("Community Reports imported.")

    def import_covariates(self, data_path):
        covariate_df = pd.read_parquet(f'{data_path}/create_final_covariates.parquet')
        covariate_statement = """
        MERGE (covariate:__Covariate__ {id:value.id})
        SET covariate += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
        WITH covariate, value
        MATCH (chunk:__Chunk__ {id: value.text_unit_id})
        MERGE (chunk)-[:HAS_COVARIATE]->(covariate)
        """
        self.batched_import(covariate_statement, covariate_df)
        print("Covariates imported.")
        
    def import_all_data(self, data_path):
        self.import_documents(data_path)
        self.import_text_units(data_path)
        self.import_entities(data_path)
        self.import_relationships(data_path)
        self.import_communities(data_path)
        self.import_community_reports(data_path)
        # self.import_covariates(data_path)
        print("All data imported.")
        
    def create_entity_name_index(self):
        """Create an index for the entity name."""
        
        index_name = "entity_name_index"
        self.driver.execute_query(""" 
        CREATE FULLTEXT INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (entity:__Entity__) ON EACH [entity.name, entity.description]
        OPTIONS {
            indexConfig: {
                `fulltext.analyzer`: 'english',
                `fulltext.eventually_consistent`: true
            }
        }
        """
        )
        print("Entity name index created.")
        print("Index name: ", index_name)
        
    def create_entity_description_vector_index(self):
        """Create a vector index for the entity."""
        
        index_name = "entity_description_vector_index"
        self.driver.execute_query(""" 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (entity:__Entity__) ON entity.description_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        )
        print("Entity description vector index created.")
        print("Index name: ", index_name)
    
    def create_relationship_description_vector_index(self):
        """
        Create a vector index for the relationship.
        """
        index_name = "relationship_description_vector_index"
        self.driver.execute_query(""" 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR ()-[relationship:RELATED]->() ON relationship.description_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        )
        print("Relationship description vector index created.")
        print("Index name: ", index_name)
        
    def create_community_summary_vector_index(self):
        """
        Create a vector index for the community.
        """
        index_name = "community_summary_vector_index"
        self.driver.execute_query(""" 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (community:__Community__) ON community.summary_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        )
        print("Community summary vector index created.")
        print("Index name: ", index_name)
        
    def create_vector_indexes(self):
        """
        Create all vector indexes for the database.
        """
        self.create_entity_name_index()
        self.create_entity_description_vector_index()
        self.create_relationship_description_vector_index()
        self.create_community_summary_vector_index()
        print("All vector indexes created.")
        
    def create_community_weight(self):
        """
        Create a weight for the community.
        """
        self.driver.execute_query("""
        MATCH (community:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(chunk)
        WITH community, count(distinct chunk) AS chunkCount
        SET community.weight = chunkCount
        """
        )
        print("Community weight created.")
        
    def transform_graph_rag_to_neo4j(self, data_path: str = "artifacts"):
        """
        ref:    https://medium.com/towards-data-science/integrating-microsoft-graphrag-into-neo4j-e0d4fa00714c
        """
        self.create_constraints()
        self.import_all_data(data_path)
        self.create_vector_indexes()
        self.create_community_weight()
        print("Graph RAG transformed to Neo4j.")
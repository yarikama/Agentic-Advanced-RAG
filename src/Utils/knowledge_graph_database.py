import os
import time
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, Result
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
import Config.constants as const
from textwrap import dedent
from knowledge_graph_statements import (
    document_statement, 
    text_statement, 
    entity_statement, 
    relationship_statement, 
    community_statement, 
    community_report_statement, 
    covariate_statement,
    constraint_statements
)

load_dotenv()

class KnowledgeGraphDatabase:
    def __init__(self):
        self.neo4j_uri = const.NEO4J_URI
        self.neo4j_username = const.NEO4J_USERNAME
        self.neo4j_password = const.NEO4J_PASSWORD
        self.neo4j_database = const.NEO4J_DATABASE
        self.graph_rag_data_path = const.GRAPH_RAG_DATA_PATH or "artifacts"
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )       
        print("GraphDatabase initialized.")
        
    def transform_graph_rag_to_neo4j(self):
        """
        ref:    https://medium.com/towards-data-science/integrating-microsoft-graphrag-into-neo4j-e0d4fa00714c
        """
        self.create_constraints()
        self.import_all_data()
        self.create_vector_indexes()
        self.create_community_weight()
        print("Graph RAG transformed to Neo4j.")
    
    def delete_all(self):
        """Delete all data and schema from the database."""
        self.delete_all_indexes()
        self.delete_all_schema()
        self.delete_all_data()
        print("All data and schema deleted.")
        
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
        for constraint_statement in constraint_statements:
            if len((constraint_statement or "").strip()) > 0:
                print(constraint_statement)
                self.driver.execute_query(constraint_statement)
                
        print("Constraints created.")
    
    def import_all_data(self):
        self.import_documents()
        self.import_text_units()
        self.import_entities()
        self.import_relationships()
        self.import_communities()
        self.import_community_reports()
        # self.import_covariates()
        print("All data imported.")
        
    def dictionary_query_result(
        self, 
        cypher: str,
        params: Dict[str, Any] = {}
    ) -> list[dict]:
        """
        The result is a list of dictionaries.

        example:
        [
            {"property1": value1, "property2": value2, ...},
            {"property1": value1, "property2": value2, ...},
        ]
        """
        result = self.driver.execute_query(
            query_=cypher,
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
        delete_indexes = dedent("""
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
    
    def import_documents(self):
        document_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_documents.parquet')        
        self.batched_import(document_statement, document_df)
        print("Documents imported.")

    def import_text_units(self):
        text_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_text_units.parquet')        
        self.batched_import(text_statement, text_df)
        print("Text Units imported.")

    def import_entities(self):
        entity_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_entities.parquet')
        self.batched_import(entity_statement, entity_df)
        print("Entities imported.")
        
    def import_relationships(self):
        relationship_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_relationships.parquet')
        self.batched_import(relationship_statement, relationship_df)
        print("Relationships imported.")
        
    def import_communities(self):
        community_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_communities.parquet')
        self.batched_import(community_statement, community_df)
        print("Communities imported.")

    def import_community_reports(self):
        community_report_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_community_reports.parquet')
        self.batched_import(community_report_statement, community_report_df)
        print("Community Reports imported.")

    def import_covariates(self):
        covariate_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_covariates.parquet')
        self.batched_import(covariate_statement, covariate_df)
        print("Covariates imported.")
        
    def create_entity_name_index(self):
        """Create an index for the entity name."""
        
        index_name = "entity_name_index"
        self.driver.execute_query(dedent(""" 
        CREATE FULLTEXT INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (entity:__Entity__) ON EACH [entity.name, entity.description]
        OPTIONS {
            indexConfig: {
                `fulltext.analyzer`: 'english',
                `fulltext.eventually_consistent`: true
            }
        }
        """))
        print("Entity name index created.")
        print("Index name: ", index_name)
        
    def create_entity_description_vector_index(self):
        """Create a vector index for the entity."""
        
        index_name = "entity_description_vector_index"
        self.driver.execute_query(dedent(""" 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (entity:__Entity__) ON entity.description_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        ))
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
        
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
        self._create_constraints()
        self._import_all_data()
        self._create_vector_indexes()
        self._create_community_weight()
        print("Graph RAG transformed to Neo4j.")
    
    def delete_all(self):
        """Delete all data and schema from the database."""
        self._delete_all_indexes()
        self._delete_all_schema()
        self._delete_all_data()
        print("All data and schema deleted.")
    
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
    
    def _create_constraints(self):
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
    
    def _import_all_data(self):
        self._import_documents()
        self._import_text_units()
        self._import_entities()
        self._import_relationships()
        self._import_communities()
        self._import_community_reports()
        # self.import_covariates()
        print("All data imported.")
    
    def _delete_all_schema(self):
        """
        Delete all schema from the database.
        Includes constraints, indexes, and everything that was created by apoc.
        """
        self.driver.execute_query("CALL apoc.schema.assert({}, {})")
        print("All schema deleted.")

    def _delete_all_data(self):
        """
        Delete all data from the database.
        """
        self.driver.execute_query("MATCH (n) DETACH DELETE n")
        print("All data deleted.")
        
    def _delete_all_indexes(self):
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
    
    def _batched_import(self, statement, df, batch_size=1000):
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
                        query_="UNWIND $rows AS value " + statement, 
                        parameters_={"rows": batch.to_dict('records')},
                        database_=self.neo4j_database
                    )
            print(result.summary.counters)
        print(f'{total} rows in { time.time() - start_s} s.')    
        return total
    
    def _import_documents(self):
        document_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_documents.parquet')        
        self._batched_import(document_statement, document_df)
        print("Documents imported.")

    def _import_text_units(self):
        text_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_text_units.parquet')        
        self._batched_import(text_statement, text_df)
        print("Text Units imported.")

    def _import_entities(self):
        entity_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_entities.parquet')
        self._batched_import(entity_statement, entity_df)
        print("Entities imported.")
        
    def _import_relationships(self):
        relationship_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_relationships.parquet')
        self._batched_import(relationship_statement, relationship_df)
        print("Relationships imported.")
        
    def _import_communities(self):
        community_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_communities.parquet')
        self._batched_import(community_statement, community_df)
        print("Communities imported.")

    def _import_community_reports(self):
        community_report_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_community_reports.parquet')
        self._batched_import(community_report_statement, community_report_df)
        print("Community Reports imported.")

    def _import_covariates(self):
        covariate_df = pd.read_parquet(f'{self.graph_rag_data_path}/create_final_covariates.parquet')
        self._batched_import(covariate_statement, covariate_df)
        print("Covariates imported.")
        
    def _create_entity_name_index(self):
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
        
    def _create_entity_description_vector_index(self):
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
    
    def _create_relationship_description_vector_index(self):
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
        
    def _create_community_summary_vector_index(self):
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
        
    def _create_vector_indexes(self):
        self._create_entity_name_index()
        self._create_entity_description_vector_index()
        self._create_relationship_description_vector_index()
        self._create_community_summary_vector_index()
        print("All vector indexes created.")
        
    def _create_community_weight(self):
        self.driver.execute_query("""
        MATCH (community:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(chunk)
        WITH community, count(distinct chunk) AS chunkCount
        SET community.weight = chunkCount
        """
        )
        print("Community weight created.")
        
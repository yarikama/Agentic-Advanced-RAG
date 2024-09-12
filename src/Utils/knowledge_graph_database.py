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
    
    def batched_import(self, statement, df, batch_size=1000):
        """
        Import a dataframe into Neo4j using a batched approach.
        Parameters: statement is the Cypher query to execute, df is the dataframe to import, and batch_size is the number of rows to import in each batch.
        """
        total = len(df)
        start_s = time.time()
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
    
    def transform_graph_rag_to_neo4j(self, datapath: str = "artifacts"):
        """ reference
        https://medium.com/towards-data-science/integrating-microsoft-graphrag-into-neo4j-e0d4fa00714c
        """

        
        # From the output of Graph RAG
        GRAPHRAG_FOLDER = datapath
        
        # create constraints, idempotent operation
        constraint_statements = """
        create constraint chunk_id if not exists for (c:__Chunk__) require c.id is unique;
        create constraint document_id if not exists for (d:__Document__) require d.id is unique;
        create constraint entity_id if not exists for (c:__Community__) require c.community is unique;
        create constraint entity_id if not exists for (e:__Entity__) require e.id is unique;
        create constraint entity_title if not exists for (e:__Entity__) require e.name is unique;
        create constraint entity_title if not exists for (e:__Covariate__) require e.title is unique;
        create constraint related_id if not exists for ()-[rel:RELATED]->() require rel.id is unique;
        """.split(";")

        for constraint_statement in constraint_statements:
            if len((constraint_statement or "").strip()) > 0:
                print(constraint_statement)
                self.driver.execute_query(constraint_statement)
        
        # === Importing the GraphRAG data into Neo4j ===
        # Import documents
        doc_df = pd.read_parquet(
            f'{GRAPHRAG_FOLDER}/create_final_documents.parquet', 
            columns=["id", "title"])
        doc_df.head(2)
        
        doc_statement = """
        MERGE (d:__Document__ {id:value.id})
        SET d += value {.title}
        """
        self.batched_import(doc_statement, doc_df)
        
        # Import text units
        text_df = pd.read_parquet(
                f'{GRAPHRAG_FOLDER}/create_final_text_units.parquet',
                columns=["id","text","n_tokens","document_ids"])
        text_df.head(2)
        
        text_statement = """
        MERGE (c:__Chunk__ {id:value.id})
        SET c += value {.text, .n_tokens}
        WITH c, value
        UNWIND value.document_ids AS document
        MATCH (d:__Document__ {id:document})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.batched_import(text_statement, text_df)
        
        
        # Import Nodes
        entity_df = pd.read_parquet(
            f'{GRAPHRAG_FOLDER}/create_final_entities.parquet',
            columns=["name","type","description","human_readable_id","id","description_embedding","text_unit_ids"])
        entity_df.head(2)
        
        entity_statement = """
        MERGE (e:__Entity__ {id:value.id})
        SET e += value {.human_readable_id, .description, name:replace(value.name,'"','')}
        WITH e, value
        CALL db.create.setNodeVectorProperty(e, "description_embedding", value.description_embedding)
        CALL apoc.create.addLabels(e, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
        UNWIND value.text_unit_ids AS text_unit
        MATCH (c:__Chunk__ {id:text_unit})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        self.batched_import(entity_statement, entity_df)
        
        # Import Relationships
        rel_df = pd.read_parquet(
            f'{GRAPHRAG_FOLDER}/create_final_relationships.parquet',
            columns=["source","target","id","rank","weight","human_readable_id","description","text_unit_ids"])
        rel_df.head(2)
        rel_statement = """
        MATCH (source:__Entity__ {name:replace(value.source,'"','')})
        MATCH (target:__Entity__ {name:replace(value.target,'"','')})
        // not necessary to merge on id as there is only one relationship per pair
        MERGE (source)-[rel:RELATED {id: value.id}]->(target)
        SET rel += value {.rank, .weight, .human_readable_id, .description, .text_unit_ids}
        RETURN count(*) as createdRels
        """
        self.batched_import(rel_statement, rel_df)
        
        # import Communities
        community_df = pd.read_parquet(
            f'{GRAPHRAG_FOLDER}/create_final_communities.parquet', 
            columns=["id","level","title","text_unit_ids","relationship_ids"])
        community_df.head(2) 
        community_statement = """
        MERGE (c:__Community__ {community:value.id})
        SET c += value {.level, .title}
        /*
        UNWIND value.text_unit_ids as text_unit_id
        MATCH (t:__Chunk__ {id:text_unit_id})
        MERGE (c)-[:HAS_CHUNK]->(t)
        WITH distinct c, value
        */
        WITH *
        UNWIND value.relationship_ids as rel_id
        MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
        MERGE (start)-[:IN_COMMUNITY]->(c)
        MERGE (end)-[:IN_COMMUNITY]->(c)
        RETURN count(distinct c) as createdCommunities
        """
        self.batched_import(community_statement, community_df)
        
        # Import Community Reports
        community_report_df = pd.read_parquet(
            f'{GRAPHRAG_FOLDER}/create_final_community_reports.parquet',
            columns=["id","community","level","title","summary", "findings","rank","rank_explanation","full_content"])
        community_report_df.head(2)
        community_report_statement = """
        MERGE (c:__Community__ {community:value.community})
        SET c += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}
        WITH c, value
        UNWIND range(0, size(value.findings)-1) AS finding_idx
        WITH c, value, finding_idx, value.findings[finding_idx] as finding
        MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})
        SET f += finding
        """
        self.batched_import(community_report_statement, community_report_df)
        
        # Import Covariates
        # cov_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_covariates.parquet')
        # cov_df.head(2)
        
        # cov_statement = """
        # MERGE (c:__Covariate__ {id:value.id})
        # SET c += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
        # WITH c, value
        # MATCH (ch:__Chunk__ {id: value.text_unit_id})
        # MERGE (ch)-[:HAS_COVARIATE]->(c)
        # """
        # self.batched_import(cov_statement, cov_df)
        
    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        result = self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_ = lambda result: [record.data() for record in result]  # Converts the result to a list of dicts
        )
        return result[0]
        
    
    def create_entity_vector_index(self):
        """
        Create a vector index for the entity.
        """
        index_name = "entity"
        self.db_query(
        """ 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        )
    
    def create_community_vector_index(self):
        """
        Create a vector index for the community.
        """
        index_name = "community"
        self.db_query(
        """ 
        CREATE VECTOR INDEX """ + index_name + """ 
        IF NOT EXISTS FOR (c:__Community__) ON c.summary_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: """ + str(const.EMBEDDING_DENSE_DIM) + """,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        )
        
    def create_community_weight(self):
        """
        Create a weight for the community.
        """
        self.db_query(
        """
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """
        )
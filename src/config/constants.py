import os
from dotenv import load_dotenv

load_dotenv()

# For Milvus Settings:
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_DATABASE_NAME = "default"

# For Embedding Settings:
# EMBEDDING_MODEL_NAME = 'infgrad/stella_en_1.5B_v5'
EMBEDDING_MODEL_NAME = 'text-embedding-3-small'
EMBEDDING_DEVICE = 'cuda'
EMBEDDING_LANGUAGE = 'en'
if EMBEDDING_MODEL_NAME == 'infgrad/stella_en_1.5B_v5':
    EMBEDDING_DENSE_DIM = 1024
elif EMBEDDING_MODEL_NAME == 'text-embedding-3-small':
    EMBEDDING_DENSE_DIM = 1536
elif EMBEDDING_MODEL_NAME == 'text-embedding-3-large':
    EMBEDDING_DENSE_DIM = 3072
EMBEDDING_SPARSE_CORPUS = "corpus.json"

# For Search Settings:
TOP_K = 8
ALPHA = 0.3
USE_HYDE = False
RERANK_TYPE = "weighted"
RERANK_WEIGHTS = [0.7, 0.3]

# For Chunking Settings:
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# For Create Collection When Processing Data Settings:
IS_CREATE_COLLECTION = False
IS_USING_NEW_CORPUS = True
IS_GPU_INDEX = True
# IS_GPU_INDEX = False

# For Huggingface Datasets Settings:
HF_NAME = None
HF_SPLITS = "train"
HF_CONTENT_COLUMN = "content"   
HF_METADATA_COLUMN = "metadata"

# For Streaming Settings:
BATCH_SIZE = 2000

# For HyDE Settings:
MODEL_NAME = "gpt-4o-mini"
MODEL_TEMPERATURE = 0.1

# Node Settings:
NODE_BATCH_SIZE = 5
NODE_RETRIEVAL_LEVEL = 1

# For Neo4j Settings:
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

NEO4J_TOP_ENTITIES = 5
NEO4J_TOP_CHUNKS = 5
NEO4J_TOP_COMMUNITIES = 5
NEO4J_TOP_OUTSIDE_RELATIONSHIPS = 5
NEO4J_TOP_INSIDE_RELATIONSHIPS = 5
NEO4J_TOP_RELATIONSHIPS = 5

# For MultiAgent Settings:
CREWAI_AGENT_VERBOSE = False
CREWAI_AGENT_MEMORY = False
CREWAI_PROCESS_VERBOSE = False


# For Pydantic Reducer Settings:
RESET_LIST = [("RESET",-1)]

# For Vector Database Settings:
IS_DOCUMENT_MAPPING = False
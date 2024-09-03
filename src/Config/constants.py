# For Milvus Settings:
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_DATABASE_NAME = "default"

# For Embedding Settings:
EMBEDDING_MODEL_NAME = 'infgrad/stella_en_1.5B_v5'
EMBEDDING_DEVICE = 'cuda'
EMBEDDING_LANGUAGE = 'en'
EMBEDDING_DENSE_DIM = 1024
EMBEDDOMG_SPARSE_CORPUS = "corpus.json"

# For Search Settings:
TOP_K = 5
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
IS_GPU_INDEX = False
# IS_GPU_INDEX = True

# For Huggingface Datasets Settings:
HF_NAME = None
HF_SPLITS = "train"
HF_CONTENT_COLUMN = "content"   
HF_METADATA_COLUMN = "metadata"

# For Streaming Settings:
BATCH_SIZE = 10000

# For HyDE Settings:
MODEL_NAME = "gpt-4o-mini"
MODEL_TEMPERATURE = 0.1

# For Milvus Settings:
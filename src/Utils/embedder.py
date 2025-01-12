import random
import string
import os
import numpy as np
from pymilvus import model
from dotenv import load_dotenv
from config import constants as const
from typing import Union, List, Dict
from sentence_transformers import SentenceTransformer
from pymilvus.model.sparse import BM25EmbeddingFunction # type: ignore
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer # type: ignore
import openai

load_dotenv()

class OpenAIEmbeddingFunction:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._dim = self._get_model_dimension()

    def _get_model_dimension(self):
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimensions.get(self.model_name, 1536)

    def encode_queries(self, texts: Union[str, List[str]]):
        return self._encode(texts)

    def encode_documents(self, texts: Union[str, List[str]]):
        return self._encode(texts)

    def _encode(self, texts: Union[str, List[str]]):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
        
    @property
    def dim(self):
        return self._dim

class Embedder:
    _dense_embedder_instance = None

    @classmethod
    def get_dense_embedder(cls, model_name, device):
        if cls._dense_embedder_instance is None:
            if model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
                cls._dense_embedder_instance = OpenAIEmbeddingFunction(model_name)
            else:
                cls._dense_embedder_instance = model.dense.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                    device=device
                )
        return cls._dense_embedder_instance
    
    def __init__(self, dense_model_name: str = const.EMBEDDING_MODEL_NAME, device: str = const.EMBEDDING_DEVICE, language: str = const.EMBEDDING_LANGUAGE):
        self.dense_model_name = dense_model_name
        self.device = device
        self.language = language
        self.analyzer = build_default_analyzer(language=self.language)
        self.dense_embedder = self.get_dense_embedder(self.dense_model_name, self.device)
        self.sparse_embedder = self.initialize_sparse_embedder()
        print("Embedder initialized")

    def initialize_dense_embedder(self):
        return self.get_dense_embedder(self.dense_model_name, self.device)

    def initialize_sparse_embedder(self) -> BM25EmbeddingFunction:
        print("Initializing sparse embedder...")
        return BM25EmbeddingFunction(self.analyzer)

    def fit_sparse_embedder(self, new_documents: List[str]):
        self.sparse_embedder.fit(new_documents)

    def save_sparse_embedder(self, path: str = const.EMBEDDING_SPARSE_CORPUS):
        if not os.path.exists("./.corpus"):
            os.makedirs("./.corpus")
        self.sparse_embedder.save(path)

    def load_sparse_embedder(self, path: str = const.EMBEDDING_SPARSE_CORPUS):
        self.sparse_embedder.load(path)

    def embed_dense(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            dense_vector = self.dense_embedder.encode_queries([text])
        else:
            dense_vector = self.dense_embedder.encode_documents(text)
        return dense_vector

    def embed_sparse(self, collection_name: str, text: Union[str, List[str]]):
        self.load_sparse_embedder("./.corpus/" + collection_name)
        if isinstance(text, str):
            sparse_vector = self.sparse_embedder.encode_queries([text])
        else:
            sparse_vector = self.sparse_embedder.encode_documents(text)
        return sparse_vector        
    
    def embed_text(self, collection_name: str, text: Union[str, List[str]]) -> Union[Dict[str, Union[List[float], Dict[int, float]]], List[Dict[str, Union[List[float], Dict[int, float]]]]]:
        dense = self.embed_dense(text)
        sparse = self.embed_sparse(collection_name, text)
        return {"dense": dense, "sparse": sparse}
    
    @property
    def dense_dim(self) -> int:
        return self.dense_embedder.dim
    
    

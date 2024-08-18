import random
import string
import os
import numpy as np
from pymilvus import model
from dotenv import load_dotenv
from Config import constants as const
from typing import Union, List, Dict
from sentence_transformers import SentenceTransformer
from pymilvus.model.sparse import BM25EmbeddingFunction # type: ignore
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer # type: ignore

load_dotenv()
        
class Embedder:
    def __init__(self, dense_model_name: str = const.EMBEDDING_MODEL_NAME, device: str = const.EMBEDDING_DEVICE, language: str = const.EMBEDDING_LANGUAGE):
        self.dense_model_name = dense_model_name
        self.device = device
        self.language = language
        self.analyzer = build_default_analyzer(language=self.language)
        self.dense_embedder = self.initialize_dense_embedder()
        self.sparse_embedder = self.initialize_sparse_embedder()
        self._dense_dim = self.dense_embedder.dim
        print("Embedder initialized")

    def initialize_dense_embedder(self) -> SentenceTransformer:
        return model.dense.SentenceTransformerEmbeddingFunction(
            model_name=self.dense_model_name,
            device=self.device
        )

    def initialize_sparse_embedder(self) -> BM25EmbeddingFunction:
        print("Initializing sparse embedder...")
        return BM25EmbeddingFunction(self.analyzer)

    def fit_sparse_embedder(self, new_documents: List[str]):
        self.sparse_embedder.fit(new_documents)

    def save_sparse_embedder(self, path: str = const.EMBEDDOMG_SPARSE_CORPUS):
        if not os.path.exists("./.corpus"):
            os.makedirs("./.corpus")
        self.sparse_embedder.save(path)

    def load_sparse_embedder(self, path: str = const.EMBEDDOMG_SPARSE_CORPUS):
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
        return self._dense_dim


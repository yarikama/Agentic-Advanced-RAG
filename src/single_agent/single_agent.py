import torch
from utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from langchain_openai import ChatOpenAI
from config import constants as c
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from config.rag_config import RAGConfig

load_dotenv()

class SingleAgent:
    def __init__(self, rag_config: RAGConfig):
        # Utils
        self.vector_database = rag_config.vector_database if rag_config.vector_database else VectorDatabase()
        self.embedder = rag_config.embedder if rag_config.embedder else Embedder()
        self.retriever = Retriever(self.vector_database, self.embedder)
        self.data_processor = DataProcessor(self.vector_database, self.embedder)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
        self.llm = ChatOpenAI(
            model=rag_config.model_name if rag_config.model_name else c.MODEL_NAME,
            temperature=rag_config.model_temperature if rag_config.model_temperature else c.MODEL_TEMPERATURE,
        )
        self.user_query = ""
        self.specific_collection = ""
        
    def update_task(self, user_query: str, specific_collection: str):
        self.user_query = user_query
        self.specific_collection = specific_collection
        
    def rerank(self, retrieved_data: list):
        all_pairs = []
        all_items = []
        
        for group in retrieved_data:
            for item in group:
                all_pairs.append([self.user_query, item['content']])
                all_items.append(item)
        
        inputs = self.rerank_tokenizer(all_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            scores = self.reranker(**inputs).logits.squeeze(-1)
        
        scored_data = list(zip(all_items, scores))
        ranked_results = sorted(scored_data, key=lambda x: x[1], reverse=True)
        
        reranked_data = []
        for item, score in ranked_results:
            reranked_item = {
                'content': item['content'],
                'metadata': item['metadata'],
                'score': score.item()
            }
            reranked_data.append(reranked_item)
        
        return reranked_data
    

    def generation(self, reranked_data: List[dict]):
        # Filter data with score > 0
        filtered_data = [item for item in reranked_data if item['score'] > -2]
        
        # Build context
        context = ""
        for item in filtered_data:
            context += item['content']
            context += '\n\n'
                
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "question": self.user_query})
        
        return response
    
    def run(self):
        # Retrive the data
        retrieved_data = self.retriever.hybrid_retrieve(self.specific_collection, self.user_query, 5)
        reranked_data = self.rerank(retrieved_data)
        # Generate the answer
        response = self.generation(reranked_data)
        return {
            "retreived_data": retrieved_data,
            "reranked_data": reranked_data,
            "result": response
        }        
        
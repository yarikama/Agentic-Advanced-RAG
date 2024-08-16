import torch
from Utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from langchain_openai import ChatOpenAI
from Utils import constants as c
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class SingleAgent:
    def __init__(self, vectordatabase: VectorDatabase = None, embedder: Embedder = None):
        # Utils
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.embedder = embedder if embedder else Embedder()
        self.retriever = Retriever(self.vectordatabase, self.embedder)
        self.data_processor = DataProcessor(self.vectordatabase, self.embedder)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
        self.llm = ChatOpenAI(
            model=c.MODEL_NAME,
            temperature=c.MODEL_TEMPERATURE,
        )
        
    def rerank(self, user_query: str, retrieved_data: list):
        all_pairs = []
        all_items = []
        
        for group in retrieved_data:
            for item in group:
                all_pairs.append([user_query, item['content']])
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
    

    def generation(self, user_query: str, reranked_data: List[dict]):
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
        
        response = chain.invoke({"context": context, "question": user_query})
        
        return response
    
    def run(self, user_query: str, specific_collection: str):
        # Retrive the data
        retrieved_data = self.retriever.hybrid_retrieve(specific_collection, user_query, 5)
        reranked_data = self.rerank(user_query, retrieved_data)
        # Generate the answer
        response = self.generation(user_query, reranked_data)
        return {
            "retreived_data": retrieved_data,
            "reranked_data": reranked_data,
            "result": response
        }        
        
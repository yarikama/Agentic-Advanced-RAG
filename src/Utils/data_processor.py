import os
import re
import nltk
import json
import random
import tempfile
import itertools
import unicodedata
import numpy as np
import scipy.sparse
from tqdm import tqdm
import concurrent.futures
from nltk.corpus import words
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from .embedder import Embedder
from datasets import load_dataset
from Config import constants as const
from .vector_database import VectorDatabase
from langchain.schema.document import Document
from typing import List, Union, Dict, Generator, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    JSONLoader, CSVLoader,PyPDFLoader,TextLoader, UnstructuredExcelLoader,
    HuggingFaceDatasetLoader, UnstructuredEmailLoader, UnstructuredEPubLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredODTLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, UnstructuredImageLoader,
)

load_dotenv()

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
        

# def chunked_iterable(iterable, size):
#     it = iter(iterable)
#     previous_last_item = None
#     while True:
#         chunk = []
#         for _ in range(size):
#             try:
#                 item = next(it)
#                 if item != previous_last_item:
#                     chunk.append(item)
#                     previous_last_item = item
#             except StopIteration:
#                 break
        
#         if not chunk:
#             break
        
#         yield tuple(chunk)

class DataProcessor:
    def __init__(self, vectordatabase: Optional[VectorDatabase] = None, embedder: Optional[Embedder] = None):
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.embedder = embedder if embedder else Embedder()
        nltk.download('words')
        self.ntlk_word = words.words()
        print("Data Processor initialized")
    
    def single_file_process(self, collection_name: str, document_path: str, is_create: bool = const.IS_CREATE_COLLECTION, use_new_corpus: bool = const.IS_USING_NEW_CORPUS):
        print(f"Processing document {document_path}...")
        content = self.load_document(document_path)
        if content:
            self.insert_document(collection_name, content, is_create, use_new_corpus)
            print(f"Document {document_path} processed successfully.")
        else:
            print(f"Error processing document {document_path}.")
        
    def directory_files_process(self, collection_name: str, directory_path: str, is_create: bool = const.IS_CREATE_COLLECTION, use_new_corpus: bool = const.IS_USING_NEW_CORPUS):
        all_documents = []
        for root, _, files in os.walk(directory_path):
            for file in tqdm(files, desc="aggregating documents"):
                document_path = os.path.join(root, file)
                content = self.load_document(document_path)
                if content:
                    all_documents.extend(content)
        self.insert_document(collection_name, all_documents, is_create, use_new_corpus)
        print(f"Directory [\"{directory_path}\"] processed successfully.\n")        
    
    def uploaded_file_process(self, 
                            collection_name: str, 
                            uploaded_file, 
                            is_create: bool = const.IS_CREATE_COLLECTION, 
                            use_new_corpus: bool = const.IS_USING_NEW_CORPUS
                            ):
        print(f"Processing uploaded file {uploaded_file.name}...")
        content = self.load_uploaded_file(uploaded_file)
        if content:
            self.insert_document(collection_name, content, is_create, use_new_corpus)
            print(f"Uploaded file {uploaded_file.name} processed successfully.")
            return content
        else:
            print(f"Error processing uploaded file {uploaded_file.name}.")
            return None

    
    def load_uploaded_file(self, uploaded_file) -> Union[List[Dict[str, Union[str, Dict]]], None]:
        tmp_dir = ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        base_filename = os.path.basename(uploaded_file.name)
        tmp_file_path = os.path.join(tmp_dir, base_filename)
        
        counter = 1
        while os.path.exists(tmp_file_path):
            name, ext = os.path.splitext(base_filename)
            tmp_file_path = os.path.join(tmp_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        try:
            with open(tmp_file_path, 'wb') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
            
            result = self.load_document(tmp_file_path)
            return result
        
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def load_document(self, file_path: str) -> Union[List[Dict[str, Union[str, Dict]]], None]:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf8')
            elif file_extension == '.epub':
                loader = UnstructuredEPubLoader(file_path)
            elif file_extension == '.html' or file_extension == '.htm':
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == '.odt':
                loader = UnstructuredODTLoader(file_path)
            elif file_extension in ['.ppt', '.pptx']:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                loader = UnstructuredImageLoader(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension in ['.eml', '.msg']:
                loader = UnstructuredEmailLoader(file_path)
            elif file_extension == '.json':
                loader = JSONLoader(file_path, jq_schema='.', text_content=False)
            else:
                print(f"Unsupported file type: {file_extension}")
                return None

            documents = loader.load()
            processed_documents = []
            for doc in documents:
                processed_content = self.preprocess_content(doc.page_content)
                processed_documents.append({
                    "content": processed_content,
                    "metadata": doc.metadata
                })
            print("Data preprocessing done.")
            return processed_documents

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None

    def preprocess_content(self, content: str) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()

        # 統一為 NFKC 正規化形式
        text = unicodedata.normalize('NFKC', text)

        # 移除 URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # 移除多餘的空白字符
        text = re.sub(r'[ \t]+', ' ', text).strip()

        # 移除特殊字符，但保留某些標點符號
        text = re.sub(r'[^\w\s.,!?;:()"-]', '', text)

        # 統一引號
        text = text.replace('"', '"').replace('"', '"')

        # 移除連續的標點符號
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)

        # 確保句子之間有適當的空格
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)

        return text.strip()

    def split_document(self, 
                       documents: List[Dict[str, Union[str, Dict]]], 
                       chunk_size: int = const.CHUNK_SIZE, 
                       chunk_overlap: int = const.CHUNK_OVERLAP
                       ) -> List[Document]:
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        all_content = [doc["content"] for doc in documents]
        
        chunks = []
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]           
            split_texts = text_splitter.split_text(content)
            doc_chunks = [Document(page_content=chunk, metadata=metadata) for chunk in split_texts]
            chunks.extend(doc_chunks)

        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        return chunks, all_content

    def insert_document(self, 
                    collection_name: str, 
                    document: Dict[str, Union[str, Dict]], 
                    is_create: bool = const.IS_CREATE_COLLECTION, 
                    use_new_corpus: bool = const.IS_USING_NEW_CORPUS
                    ):
        """
        Use this if you don't need to load or split your document.
        It means that you already have a pure preprocessed document.
        """
        print("Now splitting document...")
        chunks, all_contents_for_corpus = self.split_document(document)
        print("Document split successfully.\n")
        
        if use_new_corpus:
            def generate_semi_random_text() -> str:
                return ' '.join(random.choice(self.ntlk_word) for _ in range(50))
            additional_docs = [generate_semi_random_text() for _ in range(1000)]
            all_contents_add_fake = all_contents_for_corpus + additional_docs
            print("Now fitting sparse embedder with new documents...")
            self.embedder.fit_sparse_embedder(all_contents_add_fake)
            self.embedder.save_sparse_embedder("./.corpus/" + collection_name)
            print("Sparse embedder fitted and saved successfully.\n")
        
        print("Now generate embeddings and storing them in Milvus...")
        self.store_embeddings_in_milvus(collection_name, chunks, is_create)
        print("Document processed successfully and stored in Milvus.\n")
    
    def store_embeddings_in_milvus(self, 
                                collection_name: str, 
                                chunks: List[Document], 
                                is_create: bool = const.IS_CREATE_COLLECTION, 
                                batch_size: int = const.BATCH_SIZE
                                ):  
        def get_window_content(doc_contents, index, window_size=1):
            start = max(0, index - window_size)
            end = min(len(doc_contents), index + window_size + 1)
            return " ".join(doc_contents[start:end])
        
        def entity_generator():
            for batch in chunked_iterable(chunks, batch_size):
                doc_contents = [chunk.page_content for chunk in batch]
                embeddings = self.embedder.embed_text(collection_name, doc_contents)
                
                sparse_matrix = embeddings['sparse']
                dense_matrix = np.array(embeddings['dense']) 
                
                # Check for empty sparse vectors, if any, skip the entire batch
                valid_indices = []
                for i in range(sparse_matrix.shape[0]):
                    if sparse_matrix.indptr[i] < sparse_matrix.indptr[i+1]:
                        valid_indices.append(i)
                    else:
                        print(f"Warning: Empty sparse vector detected in batch for document {i}")
                        print(f"Content: {doc_contents[i][:100]}...")
                
                if valid_indices:
                    valid_indices = np.array(valid_indices)
                    valid_dense = dense_matrix[valid_indices].tolist()
                    valid_sparse = sparse_matrix[valid_indices]
                    
                    valid_contents = [
                        get_window_content(doc_contents, i)
                        for i in valid_indices
                    ]
                    valid_metadata = [json.dumps(batch[i].metadata) for i in valid_indices]
                    
                    yield [
                        valid_dense,
                        valid_sparse,
                        valid_contents,
                        valid_metadata,
                    ]
                else:
                    print("Warning: All documents in this batch have empty sparse vectors. Skipping batch.")
                    
        def insert_batch(batch):
            self.vectordatabase.insert_data(collection_name, batch)
            return len(batch[2])

        if is_create:
            dense_dim = self.embedder.dense_dim
            self.vectordatabase.create_collection(collection_name, dense_dim)
            
        max_workers = os.cpu_count()*2
        total_inserted = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(insert_batch, batch): batch for batch in entity_generator()}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    inserted_count = future.result()
                    total_inserted += inserted_count
                    print(f"Inserted batch of {inserted_count} entities. Total: {total_inserted}")
                except Exception as exc:
                    print(f"Error inserting batch: {exc}")
        print(f"Successfully inserted {total_inserted} chunks into Milvus.\n")
        
if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    embedder = Embedder()
    dataprocessor = DataProcessor(vectordatabase, embedder)


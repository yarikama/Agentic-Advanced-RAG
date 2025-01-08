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
import pandas as pd
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
    yield from iter(lambda: tuple(itertools.islice(it, size)), ())

class DataProcessor:
    def __init__(self, vectordatabase: Optional[VectorDatabase] = None, embedder: Optional[Embedder] = None):
        nltk.download('words')
        self.ntlk_word = words.words()
        self.embedder = embedder if embedder else Embedder()
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        print("Data Processor initialized")
        
    def dataframe_process(
        self, 
        collection_name: str, 
        df: pd.DataFrame, 
        is_create: bool = const.IS_CREATE_COLLECTION, 
        use_new_corpus: bool = const.IS_USING_NEW_CORPUS,
        is_document_mapping: bool = const.IS_DOCUMENT_MAPPING
    ) -> None:
        """This function processes the dataframe and inserts the documents into the vector database"""
        
        print(f"Processing dataframe with {df.shape[0]} rows...")
        documents = df.to_dict(orient='records')
        self.insert_document(collection_name, documents, is_create, use_new_corpus, is_document_mapping)
        print(f"Dataframe with {df.shape[0]} rows processed successfully.")
    
    def single_file_process(
        self, 
        collection_name: str, 
        document_path: str, 
        is_create: bool = const.IS_CREATE_COLLECTION, 
        use_new_corpus: bool = const.IS_USING_NEW_CORPUS
    ) -> None:
        """This function processes a single file and inserts the documents into the vector database"""

        documents = self.load_document(document_path)
        print(f"Processing document {document_path} with {len(documents)} documents...")
        self.insert_document(collection_name, documents, is_create, use_new_corpus, False)
        print(f"Document {document_path} processed successfully.")
        
    def directory_files_process(
        self, 
        collection_name: str, 
        directory_path: str, 
        is_create: bool = const.IS_CREATE_COLLECTION, 
        use_new_corpus: bool = const.IS_USING_NEW_CORPUS
    ) -> None:
        """
        This function processes all files in a directory and inserts the documents into the vector database.
        """
        aggregated_directory_documents = []
        for root, _, files in os.walk(directory_path):
            for file in tqdm(files, desc="aggregating documents"):
                document_path = os.path.join(root, file)
                documents = self.load_document(document_path)
                aggregated_directory_documents.extend(documents)
        self.insert_document(collection_name, aggregated_directory_documents, is_create, use_new_corpus, False)
        print(f"Directory [\"{directory_path}\"] processed successfully.\n")        
    
    def uploaded_file_process(
        self, 
        collection_name: str, 
        uploaded_file, 
        is_create: bool = const.IS_CREATE_COLLECTION, 
        use_new_corpus: bool = const.IS_USING_NEW_CORPUS
    ) -> None:
        """This function processes a single uploaded file and inserts the documents into the vector database"""
        
        print(f"Processing uploaded file {uploaded_file.name}...")
        documents = self.load_uploaded_file(uploaded_file)
        print(f"Uploaded file {uploaded_file.name} loaded with {len(documents)} documents.")
        self.insert_document(collection_name, documents, is_create, use_new_corpus, False)
        print(f"Uploaded file {uploaded_file.name} processed successfully.")
        return documents
    
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

    def load_document(self, 
                      file_path: str
                      ) -> Union[List[Dict[str, Union[str, Dict]]], None]:
        """
        This function loads the document from the file path.
        
        Args:
            file_path: str
        Returns:
            documents: List[Dict[str, Union[str, Dict]]]
        """
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
        """
        This function preprocesses the content.
        Including:
        - removing html tags
        - normalizing the text
        - removing urls
        - removing extra spaces
        - removing special characters
        
        Args:
            content: str
        Returns:
            processed_content: str
        """
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?;:()"-]', '', text)
        text = text.replace('"', '"').replace('"', '"')
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        return text.strip()

    def split_document(self, 
                       documents: List[Dict[str, Union[str, Dict]]], 
                       chunk_size: int = const.CHUNK_SIZE, 
                       chunk_overlap: int = const.CHUNK_OVERLAP,
                       is_document_mapping: bool = const.IS_DOCUMENT_MAPPING
                       ) -> List[Document]:
        """
        This function splits the document into chunks.
        
        Args:
            documents: List[Dict[str, Union[str, Dict]]]
            - Required: "content" (str): The content of the document.
            - Required: "metadata" (Dict): The metadata of the document.
            - Optional: "document_id" (str): The id of the document.
            chunk_size: int
            chunk_overlap: int
        Returns:
            List[Document]
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        
        all_chunks = []
        all_document_ids = []        
        all_content = [doc["content"] for doc in documents]
        
        for document in documents:
            content = document["content"]
            metadata = document["metadata"]           
            split_texts = text_splitter.split_text(content)
            document_chunks = [Document(page_content=chunk, metadata=metadata) for chunk in split_texts]
            all_chunks.extend(document_chunks)
            if is_document_mapping:
                all_document_ids.extend([document["document_id"] for document in document_chunks])

        print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks, all_content, all_document_ids

    def insert_document(self, 
                        collection_name: str, 
                        documents: List[Dict[str, Union[str, Dict]]], 
                        is_create: bool = const.IS_CREATE_COLLECTION, 
                        use_new_corpus: bool = const.IS_USING_NEW_CORPUS,
                        is_document_mapping: bool = const.IS_DOCUMENT_MAPPING):
        """
        This function inserts the document into the vector database.
        Use this function to insert a single document.
        You can choose to use new corpus to fit the sparse embedder or not.
        
        Args:
            collection_name: str
            document: Dict[str, Union[str, Dict]]
            is_create: bool
            use_new_corpus: bool
            is_document_mapping: bool
        """
        def generate_semi_random_text() -> str:
                return ' '.join(random.choice(self.ntlk_word) for _ in range(50))
            
        print("Now splitting document...")
        chunks, all_contents_for_corpus, doc_ids = self.split_document(documents=documents, is_document_mapping=is_document_mapping)
        print("Document split successfully.\n")
        
        if use_new_corpus:
            additional_ntlk_words = [generate_semi_random_text() for _ in range(1000)]
            all_contents_with_ntlk_words = all_contents_for_corpus + additional_ntlk_words
            self.embedder.fit_sparse_embedder(all_contents_with_ntlk_words)
            print("Now fitting sparse embedder with new documents...")
            self.embedder.save_sparse_embedder("./.corpus/" + collection_name)
            print("Sparse embedder fitted and saved successfully.\n")
        
        print("Now generate embeddings and storing them in Milvus...")
        self.store_entities_in_milvus(collection_name=collection_name, 
                                      chunks=chunks, 
                                      is_create=is_create, 
                                      batch_size=const.BATCH_SIZE,
                                      is_document_mapping=is_document_mapping,
                                      doc_ids=doc_ids)
        print("Document processed successfully and stored in Milvus.\n")
    
    def store_entities_in_milvus(self, 
                                collection_name: str, 
                                chunks: List[Document], 
                                is_create: bool = const.IS_CREATE_COLLECTION, 
                                batch_size: int = const.BATCH_SIZE,
                                is_document_mapping: bool = const.IS_DOCUMENT_MAPPING,
                                doc_ids: List[str] = None,):
        """
        This function stores the chunks in the vector database.
        
        Args:
            collection_name: str
            chunks: List[Document]
            is_create: bool
            batch_size: int
            is_document_mapping: bool
            doc_ids: List[str]
        """
          
        def get_window_content(doc_contents, index, window_size=1):
            """
            This function returns the window content of the document.
            It is used to get the window content of the document.
            
            Args:
                doc_contents: List[str]
                index: int
                window_size: int
            Returns:
                aggregated_content: str
            """
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
                    valid_contents = [get_window_content(doc_contents, i) for i in valid_indices]
                    valid_metadata = [json.dumps(batch[i].metadata) for i in valid_indices]
                    if is_document_mapping:
                        valid_doc_ids = [doc_ids[i] for i in valid_indices] 
                        yield [
                            valid_dense,
                            valid_sparse,
                            valid_contents,
                            valid_metadata,
                            valid_doc_ids
                        ]
                    else:
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
            self.vectordatabase.create_collection(collection_name, dense_dim, is_document_mapping)
            
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


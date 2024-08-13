import os
import nltk
import json
import random
import itertools
import numpy as np
import scipy.sparse
from tqdm import tqdm
from nltk.corpus import words
from dotenv import load_dotenv
from .embedder import Embedder
from . import constants as const
from datasets import load_dataset
from .vector_database import VectorDatabase
from langchain.schema.document import Document
from typing import List, Union, Dict, Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    JSONLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    HuggingFaceDatasetLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
    UnstructuredExcelLoader
)

load_dotenv()

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
        

class DataProcessor:
    def __init__(self, vectordatabase: VectorDatabase, embedder: Embedder):
        self.vectordatabase = vectordatabase
        self.embedder = embedder
        nltk.download('words')
        self.ntlk_word = words.words()
        print("Data Processor initialized")

    def generate_semi_random_text(self) -> str:
        return ' '.join(random.choice(self.ntlk_word) for _ in range(50))

    def load_document(self, file_path: str) -> Union[List[Dict[str, Union[str, Dict]]], None]:
        """
        根據文件副檔名載入文件並返回其內容和元數據。
        
        :param file_path: 文件路徑
        :return: 文件內容和元數據的列表，每個元素是一個字典。如果文件類型不支持，返回 None。
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
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None

    def load_multiple_datasets(self, dataset_configs: Dict[str, Dict]) -> Dict[str, List[Dict[str, Union[str, Dict]]]]:
        loaded_datasets = {}
        for dataset_name, config in dataset_configs.items():
            print(f"Loading dataset: {dataset_name}")
            loaded_datasets[dataset_name] = self.load_huggingface_dataset(
                dataset_name=config['name'],
                split=config.get('split', 'train'),
                content_column=config.get('content_column', 'context'),
                metadata_column=config.get('metadata_column', 'title'),
                name=config.get('config')
            )
        return loaded_datasets

    def prepare_rag_evaluation_datasets(self) -> Dict[str, List[Dict[str, Union[str, Dict]]]]:
    
        dataset_configs = {
            "SQuAD": {
                "name": "squad",
                "split": "train",
                "content_column": "context",
                "metadata_column": "title"
            },
            # "NaturalQuestions": {
            #     "name": "natural_questions",
            #     "split": "train",
            #     "content_column": "question",
            #     "metadata_column": "title"
            # },
            # "TriviaQA": {
            #     "name": "trivia_qa",
            #     "config": "unfiltered",
            #     "split": "train",
            #     "content_column": "question",
            #     "metadata_column": "entity_pages.title"
            # },
            # "HotpotQA": {
            #     "name": "hotpot_qa",
            #     "config": "distractor",
            #     "split": "train",
            #     "content_column": "context",
            #     "metadata_column": "title"
            # }
        }
        return self.load_multiple_datasets(dataset_configs)

    def load_huggingface_dataset(self, 
                                dataset_name: str, 
                                split: str = "train", 
                                content_column: str = 'context', 
                                metadata_column: str = 'title', 
                                name: str = None
                                ) -> List[Dict[str, Union[str, Dict]]]:

        dataset = load_dataset(dataset_name, name=name, split=split)
        
        documents = []
        for item in dataset:
            content = item[content_column]
            metadata = {'title': f"{dataset_name} {item.get(metadata_column, '')}"}
            documents.append({"content": content, "metadata": metadata})
        
        print(f"Loaded {len(documents)} documents from {dataset_name} dataset.")
        return documents

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
            
            # 只對 content 進行分割
            split_texts = text_splitter.split_text(content)
            
            # 為每個分割後的文本塊創建一個新的 Document，並保留原始的 metadata
            doc_chunks = [Document(page_content=chunk, metadata=metadata) for chunk in split_texts]
            chunks.extend(doc_chunks)

        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        return chunks, all_content

    
    def store_embeddings_in_milvus(self, 
                                collection_name: str, 
                                chunks: List[Document], 
                                is_create: bool = const.IS_CREATE_COLLECTION, 
                                batch_size: int = const.BATCH_SIZE
                                ):  
        
        if is_create:
                dense_dim = self.embedder._dense_dim
                self.vectordatabase.create_collection(collection_name, dense_dim)

        def entity_generator():
            for batch in chunked_iterable(chunks, batch_size):
                doc_contents = [chunk.page_content for chunk in batch]
                embeddings = self.embedder.embed_text(collection_name, doc_contents)
                
                sparse_matrix = embeddings['sparse']
                dense_matrix = np.array(embeddings['dense']) 
                
                # 檢查每個文檔的稀疏向量是否為空,如果是則跳過
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
                    # sentence window
                    
                    def get_window_content(doc_contents, index, window_size=1):
                        start = max(0, index - window_size)
                        end = min(len(doc_contents), index + window_size + 1)
                        return " ".join(doc_contents[start:end])
                    
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
        
        # 使用 streaming 插入
        total_inserted = 0
        for batch in entity_generator():
            self.vectordatabase.insert_data(collection_name, batch)
            total_inserted += len(batch[2])  
            print(f"Inserted batch of {len(batch[2])} entities. Total: {total_inserted}")
        
        print(f"Successfully inserted {total_inserted} chunks into Milvus.\n")

        
    def document_process(self, 
                        collection_name: str, 
                        document: Dict[str, Union[str, Dict]], 
                        is_create: bool = const.IS_CREATE_COLLECTION, 
                        use_new_corpus: bool = const.IS_USING_NEW_CORPUS
                        ):
        
        print("Now splitting document...")
        chunks, all_contents = self.split_document(document)
        print("Document split successfully.\n")
        
        # print("all_contents", all_contents)
        all_contents_add_fake = all_contents.copy()
        additional_docs = [self.generate_semi_random_text() for _ in range(1000)]
        all_contents_add_fake = all_contents + additional_docs
        
        if use_new_corpus:
            print("Now fitting sparse embedder with new documents...")
            self.embedder.fit_sparse_embedder(all_contents_add_fake)
            self.embedder.save_sparse_embedder("./.corpus/" + collection_name)
            print("Sparse embedder fitted and saved successfully.\n")
        
        print("Now generate embeddings and storing them in Milvus...")
        self.store_embeddings_in_milvus(collection_name, chunks, is_create)
        print("Document processed successfully and stored in Milvus.\n")
        
    def document_file_process(self, collection_name: str, document_path: str, is_create: bool = const.IS_CREATE_COLLECTION):
        print(f"Processing document {document_path}...")
        content = self.load_document(document_path)
        if content:
            self.document_process(collection_name, content, is_create)
            print(f"Document {document_path} processed successfully.")
        else:
            print(f"Error processing document {document_path}.")
        
    def directory_process(self, collection_name: str, directory_path: str, is_create: bool = const.IS_CREATE_COLLECTION):
        all_documents = []
        for root, _, files in os.walk(directory_path):
            for file in tqdm(files, desc="aggregating documents"):
                document_path = os.path.join(root, file)
                content = self.load_document(document_path)
                if content:
                    all_documents.extend(content)
        
        
        self.document_process(collection_name, all_documents, is_create)
        print(f"Directory [\"{directory_path}\"] processed successfully.\n")        
        # return chunks

        
if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    embedder = Embedder()
    dataprocessor = DataProcessor(vectordatabase, embedder)
    dataprocessor.directory_process("testing123", "../.data", is_create=True)    


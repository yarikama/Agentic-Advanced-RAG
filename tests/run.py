from Utils import *
from LLMMA import *

def run():
    # embedder = Embedder()
    # vectorDatabase = VectorDatabase()
    # dataset_loader = DatasetLoader()
    # retriever = Retriever(vectorDatabase, embedder)
    # dataProcessor = DataProcessor(vectorDatabase, embedder)
    # rag_system = LLMMA_Sequential_RAG_System("Hello world", vectorDatabase, embedder, "alice")
    
    # dataset_names = ["squad",]
    # for dataset_name in dataset_names:
    #     df = dataset_loader.load_dataset(dataset_name)
    #     docs_from_dataset, df = dataset_loader.process_dataset(df)
    #     dataProcessor.document_process(docs_from_dataset)
    
    # result, context = rag_system.run("What is the importance of the character alice?", "alice")
    # print(result, context)
    embedder = Embedder()
    vectorDatabase = VectorDatabase()
    rag_system = LLMMA_RAG_System(vectorDatabase, embedder)
    rag_system.run("sequential", "What is the importance of the character alice?", "alice")
    
if __name__ == "__main__":
    run()
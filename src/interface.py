import pandas as pd
from Utils import *
from Module import *
import streamlit as st
from crewai.tasks.task_output import TaskOutput
from Frontend import *
from Config.rag_config import RAGConfig
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import threading
import time
import asyncio
from pydantic import BaseModel
from typing import TypedDict


if 'vector_database' not in st.session_state:
    st.session_state.vector_database = VectorDatabase()
if 'embedder' not in st.session_state:
    st.session_state.embedder = Embedder()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor(st.session_state.vector_database, st.session_state.embedder)
if 'collections' not in st.session_state:
    st.session_state.collections = st.session_state.vector_database.list_collections()
    st.session_state.collections.append("None")
if 'dataset_loader' not in st.session_state:
    st.session_state.dataset_loader = DatasetLoader()

modular_rag_tab, build_tab, evaluation_tab = st.tabs(["Modular RAG Chatbot", "Build From Data", "Evaluation"])

st.sidebar.title("RAG Configuration")
model_select = st.sidebar.selectbox("Select Model", 
                                    ["gpt-4o", "gpt-4o-mini", "llama3-8b", "llama3-80b"],
                                    index=1,
                                    key="model_select")

model_temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.1, 0.1)

with modular_rag_tab:
    st.header("Modular RAG Chatbot")
    st.markdown("We will be using the Modular RAG Chatbot for this demo.")
    st.markdown(f"You are now using model: `{model_select}` with temperature: `{model_temperature}`, change the model and temperature in the sidebar.")
    st.subheader("User Input")
    choose_collection = st.selectbox("Select Your Collection", st.session_state.collections, index=0, key="choose_collection")
    user_query = st.text_input("User Query", "Who helps alice the most in the book?")
    if st.button("RAG Run!"):
        output_log = st.empty()
        with output_log.container():                             
            def run_rag_workflow():
                callback = ImprovedCustomStreamlitCallbackHandler(output_log.container())
                # callback = StreamlitCallbackHandler(output_log.container())
                # callback = CustomStreamlitCallbackHandler()
                rag_config = RAGConfig(
                    model_name=model_select,
                    model_temperature=model_temperature,
                    vector_database=st.session_state.vector_database,
                    callback_function=callback
                )
                rag_system = WorkFlowModularRAG(user_query, choose_collection, rag_config)
                initState = OverallState(user_query=user_query, collection=choose_collection)
                for result in rag_system.graph.stream(initState):
                    if result:
                        st.write(result)   
                                             
            st.write("Running RAG Workflow...")
            run_rag_workflow()
    
# with build_tab:
#     st.header("Build Your Own RAG Database")
#     input_method = st.selectbox("Select Import Method", ["Upload File", "User Directorie", "Use Hugging Face Datasets", ], index=1)
#     if input_method == "Upload File":
#         uploaded_file = st.file_uploader("Choose a file")
#         if uploaded_file:
#             is_create = st.checkbox("You Want to Create New Collection")
#             if is_create:
#                 specific_collection = st.text_input("Collection Name", "Must_be_Alphanumeric_and_no_space")
#             else:
#                 specific_collection = st.selectbox("Select Collection to Insert", st.session_state.collections, index=0, key="specific_collection")
#                 if specific_collection == "None":
#                     specific_collection = "User_data"
#             if st.button("Process File"):
#                 file =  st.session_state.data_processor.uploaded_file_process(specific_collection, uploaded_file, is_create, True)
#                 if file:
#                     st.expander("File Preview", expanded=False).write(file)
#                     st.write("File Uploaded Successfully!")
        

with evaluation_tab:
    st.header("Datasets We Use for Evaluation")
    st.markdown("We import the datasets from the Hugging Face datasets library.")
    st.markdown("We will be using the SQuAD and HotpotQA datasets for this demo.")

    if 'df_squad' not in st.session_state:
        st.session_state.df_squad = pd.read_parquet(".parquet/squad.parquet")
    if 'df_hotpotqa' not in st.session_state:
        st.session_state.df_hotpotqa = pd.read_parquet(".parquet/hotpotqa.parquet")
    # if 'df_triviaqa' not in st.session_state:
    #     st.session_state.df_triviaqa = st.session_state.dataset_loader.load_dataset("trivia_qa")
    # if 'df_naturalqa' not in st.session_state:
    #     st.session_state.df_naturalqa = st.session_state.dataset_loader.load_dataset("natural_qa")
        
    st.subheader("SQuAD Dataset")
    st.dataframe(st.session_state.df_squad[:100])

    st.subheader("HotpotQA Dataset")
    st.dataframe(st.session_state.df_hotpotqa[:100])
    
    # st.subheader("TriviaQA Dataset")
    # st.dataframe(st.session_state.df_triviaqa[:5])
    
    # st.subheader("NaturalQA Dataset")
    # st.dataframe(st.session_state.df_naturalqa[:5])

    # doc_squad_processed, df_squad = data_set_loader.process_dataset(df_squad, "squad")
    # doc_hotpotqa_processed, df_hotpotqa = data_set_loader.process_dataset(df_hotpotqa, "hotpot_qa")

    # st.subheader("SQuAD Dataset Processed")
    # st.dataframe(df_squad[:5])
    # st.subheader("SQuAD Documents")
    # st.write(doc_squad_processed[:2])

    # st.subheader("HotpotQA Dataset Processed")
    # st.dataframe(df_hotpotqa[:5])
    # st.subheader("HotpotQA Documents")
    # st.write(doc_hotpotqa_processed[:2])
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

modular_rag_tab, build_tab, evaluation_tab = st.tabs(["Modular RAG Chatbot", "Build From Data", "Evaluation"])

st.sidebar.title("RAG Configuration")
model_select = st.sidebar.selectbox("Select Model", 
                                    ["gpt-4o", "gpt-4o-mini", "llama3-8b", "llama3-80b"],
                                    index=1,
                                    key="model_select")

model_temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.1, 0.1)
vector_database = VectorDatabase()
collections = vector_database.list_collections()
collections.append("None")
with modular_rag_tab:
    st.header("Modular RAG Chatbot")
    st.markdown("We will be using the Modular RAG Chatbot for this demo.")
    st.markdown(f"You are now using model: `{model_select}` with temperature: `{model_temperature}`, change the model and temperature in the sidebar.")
    st.subheader("User Input")
    choose_collection = st.selectbox("Select Your Collection", collections, index=len(collections)-1, key="choose_collection")
    user_query = st.text_input("User Query", "What is the capital of France?")
    output_container = st.container()
    callback = CustomStreamlitCallbackHandler()
    # callback = StreamlitCallbackHandler(st.container())
    with output_container:
        rag_config = RAGConfig(
            model_name=model_select,
            model_temperature=model_temperature,
            vector_database=vector_database,
            callback_function=callback
        )
        if st.button("RAG Run!"):
            with st.spinner('Processing...'):
                def run_rag_workflow():
                    ragSystem = WorkFlowModularRAG(user_query, choose_collection, rag_config)
                    initState = OverallState(user_query=user_query, collection=choose_collection)
                    app = ragSystem.app
                    app.invoke(initState)
                
                with st.expander("Log"):
                    st.write(run_rag_workflow())
                # thread = threading.Thread(target=run_rag_workflow)
                # thread.start()
                
                # while thread.is_alive():
                #     time.sleep(0.1)
                    
                

    
    
with build_tab:
    st.header("Build Your Own RAG Database")
    

with evaluation_tab:
    st.header("Datasets We Use for Evaluation")
    st.markdown("We import the datasets from the Hugging Face datasets library.")
    st.markdown("We will be using the SQuAD and HotpotQA datasets for this demo.")
    data_set_loader = DatasetLoader()

    # df_squad = data_set_loader.load_dataset("squad")
    # df_hotpotqa = data_set_loader.load_dataset("hotpot_qa")

    # st.subheader("SQuAD Dataset")
    # st.dataframe(df_squad[:5])

    # st.subheader("HotpotQA Dataset")
    # st.dataframe(df_hotpotqa[:5])

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
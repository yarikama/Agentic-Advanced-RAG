import pandas as pd
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from Config.rag_config import RAGConfig
from MultiAgent import LLMMA_RAG_System
from Frontend import *
# 創建 Streamlit 容器
parent_container = st.container()

# 配置回調處理器
callback_handler = CustomStreamlitCallbackHandler()

# 配置 RAG 系統
rag_config = RAGConfig(
    model_name="gpt-4o",
    model_temperature=0.1,
    callback_function=callback_handler
)

# 初始化 RAG 系統
rag_system = LLMMA_RAG_System(rag_config=rag_config)

# 更新任務並運行
with st.spinner('Running the RAG system...'):
    with st.expander("log"):
        rag_system.tasks.update_task("What is the capital of France?", "None")
        rag_system.overall_run("sequencial", "What is the capital of France?", "None")

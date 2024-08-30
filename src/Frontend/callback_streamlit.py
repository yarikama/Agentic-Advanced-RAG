"""Callback Handler that prints to std out."""
from __future__ import annotations

from threading import Lock
import queue


from typing import TYPE_CHECKING, Any, Dict, Optional, List

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish

import streamlit as st


import threading
from streamlit.runtime.scriptrunner.script_run_context import (
    SCRIPT_RUN_CONTEXT_ATTR_NAME,
    get_script_run_ctx,
)
from streamlit.errors import NoSessionContext
from typing import Any, TypeVar, cast
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

T = TypeVar("T")

def with_streamlit_context(fn: T) -> T:
    """確保回調函數在正確的 Streamlit 上下文中執行。"""
    ctx = get_script_run_ctx()

    if ctx is None:
        raise NoSessionContext(
            "with_streamlit_context 必須在有上下文的情況下調用。"
        )

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        thread = threading.current_thread()
        has_context = hasattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME) and (
            getattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME) == ctx
        )

        if not has_context:
            setattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, ctx)

        try:
            return fn(*args, **kwargs)
        finally:
            if not has_context:
                delattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME)

    return cast(T, wrapper)

class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    @with_streamlit_context
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        st.markdown(f"Entering new {class_name} chain...")

    @with_streamlit_context
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        st.write("Finished chain.")

    @with_streamlit_context
    def on_agent_action(self, action: "AgentAction", color: Optional[str] = None, **kwargs: Any) -> Any:
        """Run on agent action."""
        for line in action.log.split("\n"):
            st.markdown(line)

    @with_streamlit_context
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        st.write(serialized)
        st.write(input_str)

    @with_streamlit_context
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        with st.expander("Tool Ended:", expanded=True):
            with st.expander("Observation:", expanded=True):
                if observation_prefix is not None:
                    st.markdown(f"\n{observation_prefix}")
            st.markdown(f"\n{output}")
            if llm_prefix is not None:
                with st.expander("LLM Prefix:", expanded=True):
                    st.markdown(f"\n{llm_prefix}")

    @with_streamlit_context
    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        st.write("Agent ending")

    @with_streamlit_context
    def on_agent_finish(
        self, finish: "AgentFinish", color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        st.write("Agent ended")


class ImprovedCustomStreamlitCallbackHandler(BaseCallbackHandler):
    """Improved Callback Handler that safely updates Streamlit UI."""
    def __init__(self, container):
        self.container = container
    
    @with_streamlit_context
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self.container.markdown(f"Starting: {class_name}")

    @with_streamlit_context
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.container.markdown("Chain completed")
        
    @with_streamlit_context
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        self.container.markdown(f"Starting tool: {serialized.get('name', 'Unknown tool')}")

    @with_streamlit_context
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.container.markdown(f"Tool output: {output}")


def render_streamlit_messages(messages: List[tuple]):
    for message_type, content in messages:
        if message_type == "chain_start":
            with st.expander("Starting a new Agent Chain:", expanded=True):
                st.markdown(content)
        elif message_type == "chain_end":
            with st.expander("Finished chain.", expanded=False):
                st.write(content)
        elif message_type == "agent_action":
            with st.expander("AI Thought Bubble - Next Action:", expanded=True):
                for line in content.split("\n"):
                    st.markdown(line)
        elif message_type == "tool_start":
            serialized, input_str = content
            with st.expander("Tool Started", expanded=True):
                st.write(serialized)
                st.write(input_str)
        elif message_type == "tool_end":
            with st.expander("Tool Ended:", expanded=True):
                st.markdown(f"\n{content}")
        elif message_type == "text":
            with st.expander("Agent message:", expanded=False):
                st.write(content)
        elif message_type == "agent_finish":
            with st.expander("Agent Ended.", expanded=True):
                st.write(content)
                
class PrintCallback(BaseCallbackHandler):
    def printMsg(self, msg):
        print(msg)
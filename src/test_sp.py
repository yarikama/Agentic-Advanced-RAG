import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel


class ExampleModel(BaseModel):
    some_text: str
    some_number: int
    some_boolean: bool

data = sp.pydantic_form(key="my_sample_form", model=ExampleModel)
if data:
    st.json(data.model_dump())
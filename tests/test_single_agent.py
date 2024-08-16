from src.Module import *

user_query = "what is the importance of the character alice?"
specific_collection = "alice"

workflow = WorkFlowSingleAgentRAG(user_query , specific_collection)

init_state = SingleState(
    user_query=user_query,
    specific_collection=specific_collection,
)

app = workflow.app
app.invoke(init_state)
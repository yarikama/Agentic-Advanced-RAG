from LangGraph import WorkFlow
from LangGraph.state import OverallState

user_query = "where is the France?"
specific_collection = "alice"

workflow = WorkFlow(user_query , specific_collection)

init_state = OverallState(
    user_query=user_query,
    specific_collection=specific_collection,
    repeat_times=0,
)

app = workflow.app
app.invoke(init_state)

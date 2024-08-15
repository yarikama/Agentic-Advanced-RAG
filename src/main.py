from Module import WorkFlowModularRAG
from Module.state import OverallState

user_query = "what is the importance of the character alice?"
specific_collection = None

workflow = WorkFlowModularRAG(user_query , specific_collection)

init_state = OverallState(
    user_query=user_query,
    specific_collection=specific_collection,
    repeat_times=0,
)

app = workflow.app
app.invoke(init_state)

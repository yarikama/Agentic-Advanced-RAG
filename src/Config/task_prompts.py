# prompts.py
from textwrap import dedent


# User Query Classification Task
USER_QUERY_CLASSIFICATION_PROMPT = dedent("""
Analyze the following user query and determine if it requires information retrieval to be answered accurately:

User Query: "{user_query}"

Your task is to classify this query as either requiring retrieval or not. Consider the following guidelines:

1. Queries that typically require retrieval:
   - Specific factual information (e.g., historical dates, statistics, current events)
   - Detailed explanations of complex topics
   - Information about recent or rapidly changing subjects

2. Queries that typically don't require retrieval:
   - General knowledge questions
   - Simple calculations or conversions
   - Language translations
   - Requests for creative content generation
   - Basic concept explanations

Provide your classification as a boolean value:
- True if the query requires retrieval
- False if the query can be answered without retrieval

Justify your decision briefly.
""")

USER_QUERY_CLASSIFICATION_EXPECTED_OUTPUT = dedent("""
A pydantic object with the following structure:
class UserQueryClassificationResult(BaseModel):
    needs_retrieval: bool
    justification: str
""")

# Plan Coordinator Task
PLAN_COORDINATION_PROMPT = dedent("""
As the Plan Coordinator, create a high-level plan for your teammates to answer this user query: {user_query}

Your task:
1. Create a step-by-step plan that outlines the major stages to answer the query.
2. Each step should be a clear, concise action.
3. Ensure the plan covers all aspects of addressing the query.
4. Consider the roles of each team member and assign responsibilities accordingly
    Your team members: Query Processor, Retriever, Reranker, Generator, Response Auditor, Database Updater, and Summarizer.            
""")

PLAN_COORDINATION_EXPECTED_OUTPUT = dedent("""
A plan outlining the major stages for your teammates to answer the user query.
""")


# Query Processor Task
QUERY_PROCESS_PROMPT = dedent("""
User query: {user_query}

Analyze the following user query and prepare it for retrieval:
a. Transform or rephrase the query to improve its effectiveness for retrieval.
b. Identify any complex or ambiguous terms that may require further decomposition.
c. Decompose the transformed query into simpler sub-queries if necessary.
""")

QUERY_PROCESS_EXPECTED_OUTPUT = dedent("""
Your output should be a Pydantic object of type QueriesProcessResult with the following structure:
class QueriesProcessResult(BaseModel):
    original_query: str
    transformed_queries: Optional[List[str]]
    decomposed_queries: Optional[List[str]]

Ensure that:
- If needs_retrieval is True, provide transformed_query and optionally decomposed_queries.
- decomposed_queries and transformed_queries, if provided, should be a list of strings, each representing a sub-query.
""")                            


# First Query Processor Task
SUB_QUERIES_CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification and identify relevant collections:

Expected context: You will receive a QueriesProcessResult object with the following structure:

class QueriesProcessResult(BaseModel):
    original_query: str
    transformed_queries: Optional[List[str]]
    decomposed_queries: Optional[List[str]]

Use the list_all_collections_tool to get a list of all available collections.

For every query(in original query, transformed query, decomposed_queries) in QueriesProcessResult:
a. Classify the query as needing retrieval or not (by judging whether the query requires external data or time-sensitive information).
- If retrieval is needed, store the query with the collection name.
    - Determine the most relevant collection name for the query from the list of collections.
    - If no collection is relevant, use None.
- If retrieval is not needed, skip the query.
    - Elaborate on the reason for skipping in the justification.

Compile a SubQueriesClassificationResult only for queries that need retrieval.
""")

SUB_QUERIES_CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification:

Expected context: You will receive a Queries object with the following structure:

class QueriesProcessResult(BaseModel):
    original_query: str
    transformed_queries: Optional[List[str]]
    decomposed_queries: Optional[List[str]]

Specific Collection: {specific_collection}

For every query(in original query, transformed query, decomposed_queries) in Queries:
a. Classify the query as needing retrieval or not (by judging whether the query requires external data or time-sensitive information).
- If retrieval is needed, store the query with the collection name.
    - the collection name is the input of specific collection.
- If retrieval is not needed, skip the query.
    - Elaborate on the reason for skipping in the justification.

Compile a QueriesIdentification only for queries that need retrieval.
""")                                                            


SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT = dedent("""
Your output should be a pydantc object with the following structure:
class SubQueriesClassificationResult(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]
""")

TOPIC_SEARCHING_PROMPT = dedent("""
1. Analyze the user's question and the retrieve the topic using the global_retrieve_topic tool.
    - The global_retrieve_topic tool requires the user's question and a level of community data to retrieve (0-3).
    - It will return all the community data.
2. For each key point:
    - Assign an importance score (0-100).
""")

TOPIC_SEARCHING_EXPECTED_OUTPUT = dedent("""
The response should be formatted as a Pydantic object with the following structure:

class TopicSearchingEntity(BaseModel):
    description: str
    score: int
    example_sentence: List[str]

class TopicSearchingResult(BaseModel):
    topics: List[TopicSearchingEntity]
""")

RETRIEVAL_PROMPT = dedent("""
Using a SubQueriesClassificationResult object from the context, perform the retrieval process:

class SubQueriesClassificationResult(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]

1. Extract the list of collection names and list of queries from the SubQueriesClassificationResult object.

2. Use the _retrieve tools with these two lists:
- The first argument should be the list of collection names from SubQueriesClassificationResult.
- The second argument should be the list of queries from SubQueriesClassificationResult.
- Decide the top_k value based on the expected number of relevant results. (e.g., top_k=5)

3. The _retrieve tool will return a dictionary of 2 lists:
- content: List[str]  # Retrieved content for each query
- metadata: List[Dict[str, Any]]  # Retrieved metadata for each query
There is no duplicate content entries in the retrieved data.
""")

# 4. Remove irrelevant data:
# a. For each content entry, evaluate its relevance to the original user query: {user_query}.
# b. Remove content entries that are deemed irrelevant, along with their corresponding metadata.
# c. If any concerns about relevance arise, don't remove the entire entry.

RETRIEVAL_EXPECTED_OUTPUT = dedent("""
A RetrievalResult pydantic object containing consolidated metadata and content lists.
""")

RETRIEVAL_DETAIL_DATA_FROM_TOPIC_PROMPT = dedent("""
specific_collection = {specific_collection}

You will be given a list of TopicSearchingEntity objects. Each object has the following structure:
class TopicSearchingEntity:
description: str
score: float
example_sentences: List[str]
Select the topics or example sentences with high scores from the TopicSearchingEntity objects. Prioritize those with higher scores as they are likely to be more relevant.
For each selected high-scoring topic or example sentence:
a. Use it as a query for the _retrieve tool.
b. When using the _retrieve tool, include the specific_collection as a parameter.
c. The _retrieve tool will return a dictionary with two lists:

content: List[str]  # Retrieved content for each query
metadata: List[Dict[str, Any]]  # Retrieved metadata for each query


After retrieving data for all selected topics/sentences:
a. Combine all the retrieved content and metadata.
b. Remove any duplicate content entries from the combined results.
Organize the final set of unique, relevant content and metadata.
Present the retrieved information in a clear, structured format, linking each piece of content to its corresponding metadata where applicable.

Remember:

Focus on using the most relevant topics or sentences for retrieval.
Always use the specific_collection when calling the _retrieve tool.
Ensure there are no duplicates in the final set of retrieved data.
The goal is to provide comprehensive, non-redundant information related to the high-scoring topics.
""")

RETRIEVAL_DETAIL_DATA_FROM_TOPIC_EXPECTED_OUTPUT = dedent("""
A RetrievalResult pydantic object containing consolidated metadata and content lists.
""")



RERANK_PROMPT = dedent("""
Perform a reranking process on the retrieved content based on its relevance to the user query.

User Query: "{user_query}"

You have to score each content based on its relevance to the user query.
A relevance score should be a float value between 0 and 1, where 1 indicates high relevance and 0 indicates low relevance.
Retrun the list of scores for each content based on their relevance to the user query.
""")

RERANK_EXPECTED_OUTPUT = dedent("""
Your output should be a RerankingResult object.
class RerankingResult(BaseModel):
    ranked_content: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[float]
""")

GENERATION_PROMPT = dedent("""
Analyze the reranked data and formulate a comprehensive answer to the user's query.

Original user query: {user_query}

Your task:
1. Review the original user query.
2. Carefully examine the ranked data provided by the Reranker for each sub-query.
3. Synthesize all the information to form a coherent and comprehensive answer that addresses the original user query.
4. Ensure that your response covers all aspects of the user's query and the derived sub-queries.
5. Identify key findings, insights, and connections across all the data.
6. Provide a detailed analysis, breaking down the answer into relevant topics or aspects.
7. Assess the confidence level of your analysis based on the available data.
8. Identify any limitations in the data or analysis.
9. If applicable, suggest potential follow-up questions or areas for further investigation.

Your output should be a comprehensive analysis that ties together all the information from the sub-queries to directly address the original user query.
""")

GENERATION_EXPECTED_OUTPUT = dedent("""
"A comprehensive analysis answering the user's original query based on all the provided data from sub-queries."
""")

RESPONSE_AUDITOR_PROMPT = dedent("""
Review the summary provided in the context and evaluate if it adequately addresses the user's query and meets the RAGAS evaluation criteria.

User query: {user_query}

Your task:
1. Carefully review the original user query to understand the user's intent and requirements.
2. Examine the summary provided by the Summarizer, focusing on these key metrics aligned with RAGAS:
a. Context Relevance: How well the summary uses relevant information from the retrieved context
b. Answer Relevance: How directly and completely the summary addresses the original query
c. Faithfulness: How truthful the summary is to the source information without adding unsupported claims
d. Conciseness: How concise and to-the-point the summary is while maintaining completeness

3. For each metric, provide a score between 0 and 1, where 0 is the lowest and 1 is the highest.
4. Calculate an overall(average) score based on these individual metrics.
5. If the overall score is below 0.7, flag the response for a restart from the query processing stage.
6. Provide brief comments for each metric and additional general comments if necessary.


Ensure that:
- Each metric (Context Relevance, Answer Relevance, Faithfulness, Conciseness) is represented in the metrics list.
- All scores (including overall_score) are between 0 and 1.
- restart_required is set to True if the overall_score is below 0.7, False otherwise.
- Provide concise and constructive comments for each metric and in additional_comments if needed.

If restart_required is True, include in additional_comments specific suggestions for improvement in the query processing or other stages.
""")

RESPONSE_AUDITOR_EXPECTED_OUTPUT = dedent("""
Your output should be a Pydantic object of type ResponseAuditResult with the following structure:
ResponseAuditResult(
    metrics: List[AuditMetric],
    overall_score: float,
    restart_required: bool,
    additional_comments: Optional[str]
)

Where AuditMetric is structured as:
AuditMetric(
    name: str,
    score: float,
    comment: Optional[str]
)
""")

DATABASE_UPDATER_PROMPT_WITHOUT_SPECIFIC_COLLECTION = dedent("""
Store the user query and summary response in the database if approved by the Response Auditor.

User query: {user_query}

Steps:
1. Review the Response Auditor's evaluation and Classification results.

2. If the Response Auditor approves (overall_score >= 0.7 and restart_required is False) and the Classification result indicates retrieval is needed:
   a. Use _list_all_collections() to get a list of available collections.
   b. Analyze the user query and choose the most relevant collection from the list.
   c. Use _dense_retrieve_data([chosen_collection], [user_query], top_k=1) to check for similar existing queries.
   d. If no similar query exists or the existing answer is significantly different:
      i. Prepare the question-answer pair:
         question = user_query
         answer = summarizer's complete response without any modification
      ii. Use _insert_qa_into_db(chosen_collection, question, answer) to store the information.
   e. If a similar query exists with a very similar answer, skip the insertion to avoid duplication.
   
3. Output whether the insertion operation was successful or skipped and explain the reason in pydanctic object.
""")

DATABASE_UPDATER_PROMPT_WITH_SPECIFIC_COLLECTION = dedent("""
Store the user query and summary response in the specified collection if approved by the Response Auditor.

User query: {user_query}
Specific collection: {specific_collection}

Steps:
1. Review the Response Auditor's evaluation and Classification results.

2. If the Response Auditor approves (overall_score >= 0.7 and restart_required is False) and the Classification result indicates retrieval is needed:
   a. Use _dense_retrieve_data([specific_collection], [user_query], top_k=1) to check for similar existing queries.
   b. If no similar query exists or the existing answer is significantly different:
      i. Prepare the question-answer pair:
         question = user_query
         answer = summarizer's complete response without any modification
      ii. Use _insert_qa_into_db(specific_collection, question, answer) to store the information.
   c. If a similar data is retrieved, skip the insertion process (don't use the insert_qa_too) and end the task.
   
3. If the Response Auditor does not approve or the Classification result indicates no retrieval is needed, skip the insertion process.

3. Output whether the insertion operation was successful or skipped and explain the reason in pydanctic object.
""")

DATABASE_UPDATE_EXPECTED_OUTPUT= dedent("""
A pydantic object with the following structure:
class DatabaseUpdateResult(BaseModel):
    success: bool
    reason: str
""")
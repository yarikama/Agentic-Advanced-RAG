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
UserQueryClassification:
    needs_retrieval: bool,
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
"A plan outlining the major stages for your teammates to answer the user query."
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
Your output should be a Pydantic object of type Queries with the following structure:
Queries(
    original_query: str,
    transformed_queries: Optional[List[str]],
    decomposed_queries: Optional[List[str]]
)

Ensure that:
- If needs_retrieval is True, provide transformed_query and optionally decomposed_queries.
- decomposed_queries and transformed_queries, if provided, should be a list of strings, each representing a sub-query.
""")                            

# First Query Processor Task
SUB_QUERIES_CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification and identify relevant collections:

Expected context: You will receive a Queries object with the following structure:

Queries(
    original_query: str,
    transformed_query: Optional[str],
    decomposed_queries: Optional[List[str]]
)

Use the list_all_collections_tool to get a list of all available collections.

For every query(in original query, transformed query, decomposed_queries) in Queries:
a. Classify the query as needing retrieval or not (by judging whether the query requires external data or time-sensitive information).
- If retrieval is needed, store the query with the collection name.
    - Determine the most relevant collection name for the query from the list of collections.
    - If no collection is relevant, use None.
- If retrieval is not needed, skip the query.
    - Elaborate on the reason for skipping in the justification.

Compile a QueriesIdentification only for queries that need retrieval.
A QueriesIdentification object should contain the queries and their corresponding collection names.
""")

SUB_QUERIES_CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification:

Expected context: You will receive a Queries object with the following structure:

Queries(
    original_query: str,
    transformed_query: Optional[str],
    decomposed_queries: Optional[List[str]]
)

Specific Collection: {specific_collection}

For every query(in original query, transformed query, decomposed_queries) in Queries:
a. Classify the query as needing retrieval or not (by judging whether the query requires external data or time-sensitive information).
- If retrieval is needed, store the query with the collection name.
    - the collection name is the input of specific collection.
- If retrieval is not needed, skip the query.
    - Elaborate on the reason for skipping in the justification.

Compile a QueriesIdentification only for queries that need retrieval.
A QueriesIdentification object should contain the queries and their corresponding collection names.
""")                                                            


SUB_QUERIES_CLASSIFICATION_EXPECTED_OUTPUT = dedent("""
Your output should be a pydantc object with the following structure:
class QueriesIdentification(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]
""")

RETRIEVAL_PROMPT = dedent("""
Using a QueriesIdentification object from the context, perform the retrieval process:
QueriesIdentification object:
- queries: List[str]
- collection_name: List[Optional[str]]

1. Extract the list of collection names and list of queries from the QueriesIdentification object.

2. Use the _retrieve tools with these two lists:
- The first argument should be the list of collection names from QueriesIdentification.
- The second argument should be the list of queries from QueriesIdentification.
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
A RefinedRetrievalData pydantic object containing consolidated metadata and content lists.
""")

RETRIEVAL_GRAPH_TOPIC_PROMPT = dedent("""
User Query: "{user_query}"

Task:
1. Use the QueriesIdentification object from the context:
   - Extract the list of queries from QueriesIdentification.queries
   - Ignore the collection_name field

2. Retrieve relevant topics:
   - Use the _retrieve_graph_topic tool with the extracted list of queries
   - Find the top_k most relevant topics related to these queries

3. Generate hypothetical answers:
   - Based on the retrieved topics and the original user query, generate a list of possible answers or responses
   - These answers should be hypothetical and aimed at addressing the user's query in light of the retrieved topics

4. Store results using _retrieval_details tool:
   - Use the _retrieval_details tool to store the generated information
   - Pass the following to the _retrieval_details tool:
     a. The list of retrieved relevant topics from step 2
     b. The list of generated hypothetical answers from step 3

Ensure that:
1. The _retrieve_graph_topic tool is used correctly to obtain the initial topics
2. The hypothetical answers are generated based on both the retrieved topics and the user query
3. The _retrieval_details tool is used to store both the topics and the generated answers
4. The process follows the exact order: retrieve topics, generate answers, store both in _retrieval_details

Note: The final step of using the _retrieval_details tool is crucial for storing the information for subsequent retrieval steps. There's no need to create a separate RetrievalGlobalTopics object in this process.
""")


RERANK_PROMPT = dedent("""
Perform a reranking process on the retrieved content based on its relevance to the user query.

User Query: "{user_query}"

Retrieved Data: "{retrieved_data}"

You have to score each content based on its relevance to the user query.
A relevance score should be a float value between 0 and 1, where 1 indicates high relevance and 0 indicates low relevance.
Retrun the list of scores for each content based on their relevance to the user query.
""")

RERANK_EXPECTED_OUTPUT = dedent("""
Your output should be a RankedRetrievalData object.
RankedRetrievalData:
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

SUMMARIZER_PROMPT = dedent("""
Summarize the comprehensive analysis provided by the Generator into a concise, accurate, and highly relevant response to the user's original query. Your summary will be evaluated based on context relevance, answer relevance, faithfulness, and conciseness.

Original user query: {user_query}

Your task:
1. Carefully review the original user query and the Generator's comprehensive analysis.
2. Create a summary that excels in the following areas:

a. Context Relevance:
    - Ensure every piece of information in your summary is directly related to the query.
    - Avoid including any irrelevant or tangential information.

b. Answer Relevance:
    - Provide a clear, direct answer to the main question(s) in the original query.
    - Ensure your answer is complete and addresses all aspects of the query.

c. Faithfulness (Truthfulness):
    - Stick strictly to the facts and insights provided in the Generator's analysis.
    - Do not introduce any new information or make assumptions beyond what's in the source material.
    - If there are uncertainties or limitations in the data, clearly state them.

d. Conciseness:
    - Make your summary as brief as possible while still fully answering the query.
    - Use clear, straightforward language.
    - Avoid repetition and unnecessary elaboration.

3. Structure your summary as follows:
- Start with a direct answer to the main query.
- Follow with key supporting facts and insights, prioritized by relevance.
- Include a brief statement on data limitations or confidence level, if relevant.
- End with a concise conclusion that ties back to the original query.

4. Double-check your summary to ensure:
- It doesn't contain any information not present in the Generator's analysis.
- Every sentence directly contributes to answering the user's query.
- The language is clear and accessible, avoiding unnecessary jargon.

Your output should be a highly relevant, faithful, and concise summary that directly and fully answers the user's original query, optimized for high performance in RAGAS evaluation.
""")

SUMMARIZER_EXPECTED_OUTPUT = dedent("""
A concise, highly relevant summary including:
1. Direct and complete answer to the main query
2. Key supporting facts and insights, strictly from the source material
3. Brief mention of limitations or confidence level (if applicable)
4. Concise conclusion tying back to the original query

The summary should be optimized for high RAGAS scores in context relevance, answer relevance, faithfulness, and conciseness.
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
Your output should be a Pydantic object of type AuditResult with the following structure:
AuditResult(
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

DATABASE_UPDATER_EXPECTED_OUTPUT = dedent("""
A Pydantic object of type UpdateCondition with the following structure:
UpdateCondition(
    is_database_updated: bool,
    reason: str
)
""")
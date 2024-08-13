# prompts.py
from textwrap import dedent

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

PLAN_COORDINATOR_PROMPT = dedent("""
As the Plan Coordinator, create a high-level plan for your teammates to answer this user query: {user_query}

Your task:
1. Create a step-by-step plan that outlines the major stages to answer the query.
2. Each step should be a clear, concise action.
3. Ensure the plan covers all aspects of addressing the query.
4. Consider the roles of each team member and assign responsibilities accordingly
    Your team members: Query Processor, Retriever, Reranker, Generator, Response Auditor, Database Updater, and Summarizer.            
""")

PLAN_COORDINATOR_EXPECTED_OUTPUT = dedent("""
"A plan outlining the major stages for your teammates to answer the user query."
""")

QUERY_PROCESSOR_PROMPT = dedent("""
User query: {user_query}

Analyze the following user query and prepare it for retrieval:
a. Transform or rephrase the query to improve its effectiveness for retrieval.
b. Identify any complex or ambiguous terms that may require further decomposition.
c. Decompose the transformed query into simpler sub-queries if necessary.
""")

QUERY_PROCESSOR_EXPECTED_OUTPUT = dedent("""
Your output should be a Pydantic object of type Queries with the following structure:
Queries(
    original_query: str,
    transformed_query: Optional[str],
    decomposed_queries: Optional[List[str]]
)

Ensure that:
- If needs_retrieval is True, provide transformed_query and optionally decomposed_queries.
- If needs_retrieval is False, only set this boolean field to False.
- decomposed_queries, if provided, should be a list of strings, each representing a sub-query.
- All fields should be properly filled according to your analysis.
""")                            

CLASSIFICATION_PROMPT_WITHOUT_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification and identify relevant collections:

Expected context: You will receive a Queries object with the following structure:

Queries(
    original_query: str,
    transformed_query: Optional[str],
    decomposed_queries: Optional[List[str]]
)

1. Use the list_all_collections_tool to get a list of all available collections.

2. For each Queries object in the context:
a. Analyze the query (either transformed_query or each query in decomposed_queries if present).
b. Determine the most relevant collection name for the query based on the list from step 1.
    If no collection is relevant, use None.
c. Keep the needs_retrieval flag from the Queries object.

3. Compile a list of QueryClassification objects only for queries that need retrieval (needs_retrieval is True).
Each QueryClassification object should contain the query and its corresponding collection name.

4. Output the final list of QueryClassification objects.
""")

CLASSIFICATION_PROMPT_WITH_SPECIFIC_COLLECTION = dedent("""
Using the query decomposition and transformation results from the context, perform classification,
and set the specific collection provided: {specific_collection} to collection name for all queries.

Expected context: You will receive a list of Queries objects with the following structure:
Queries(
    original_query: str,
    transformed_query: Optional[str],
    decomposed_queries: Optional[List[str]]
)
1. For each query in the context:
    a. Analyze the query (either transformed_query or each query in decomposed_queries if present).
    b. Set the collection name to the specific collection provided.
    c. Keep the needs_retrieval flag from the Queries object.
2. Compile a list of QueryClassification objects only for queries that need retrieval (needs_retrieval is True).
""")

CLASSIFICATION_EXPECTED_OUTPUT = dedent("""
Your output should be a list of Pydantic Objects with the following structure:
class QueryClassification(BaseModel):
    query: str
    needs_retrieval: bool
    collection_name: Optional[str] = None
If retrieval is not needed for a query, set its "collection_name" to None.
""")

RETRIEVAL_PROMPT = dedent("""
User Query: {user_query}

Using the list of QueryClassification objects from the context, perform the retrieval process:

1. Process the QueryClassification objects:
a. Extract the collection names and queries from each QueryClassification object.
b. Create two separate lists: one for collection names and one for queries.
c. Only include items where needs_retrieval is True and collection_name is not None.
d. Ensure the lists are in the same order.

2. Use the retrieve_data_tool with these two lists:
- The first argument should be the list of collection names.
- The second argument should be the list of queries.
- Use the top_k value: 5 for retrieving data.

3. The retrieve_data_tool will return a list of dictionaries, each containing:
- query: str
- collection_name: str
- retrieved_data: List[Dict[str, Any]]

4. Compile these results into a RefinedRetrievalData object:
RefinedRetrievalData:
- metadata: List[Dict[str, Any]]  # Extracted from retrieved_data
- content: List[str]  # Extracted from retrieved_data

5. Remove duplicates:
a. Identify and remove any duplicate content entries.
b. For each removed duplicate content, also remove its corresponding metadata entry.
c. Ensure that the metadata and content lists remain synchronized after removal.

6. Remove irrelevant data:
a. For each content entry, evaluate its relevance to the original user query.
b. Remove content entries that are deemed irrelevant, along with their corresponding metadata.
c. If any concerns about relevance arise, don't remove the entire entry.

7. Return the final RefinedRetrievalData object with unique, relevant content and corresponding metadata.
""")

RETRIEVAL_EXPECTED_OUTPUT = dedent("""
A RefinedRetrievalData object containing consolidated metadata and content lists.
""")

RERANK_PROMPT = dedent("""
Perform a reranking process on the retrieved content based on its relevance to the user query.

User Query: "{user_query}"

Follow these steps:
1. The input RefinedRetrievalData contains:
    - metadata: List[Dict[str, Any]]
    - content: List[str]
2. Your tasks are:
    a) Evaluate the relevance of each piece of content to the user query.
    b) Assign a relevance score (float between 0 and 1) to each piece of content.
    c) Compile these scores into a list of floats (relevance_scores).
    d) Use the rerank tool with the original metadata, content, and your relevance_scores.
3. The rerank tool will return ranked data, metadata, and relevance scores which are same as RankedRetrievalData.
4. Just put the result return by your tool into RankedRetrievalData.
RankedRetrievalData:
    ranked_data: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[float]
""")

RERANK_EXPECTED_OUTPUT = dedent("""
Your output should be a RankedRetrievalData object.
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

DATABASE_UPDATER_PROMPT = dedent("""
Store the user query and summary response in the database if approved by the Response Auditor.

User query: {user_query}
Specific collection: {specific_collection}

Steps:
1. Review the Response Auditor's evaluation.

2. If the Response Auditor approves (overall_score >= 0.7 and restart_required is False):
a. If a specific collection is not provided, use list_all_collections_tool to identify the most suitable collection.
otherwise, use the provided specific collection.
b. Use insert_qa_into_db_tool to store the information in the selected collection.
c. The stored information should only include:
    - Question: The original user query
    - Answer: The summarizer's complete response

3. If not approved (overall_score < 0.7 or restart_required is True), take no action.

Do not modify the summarizer's response in any way. Store and output it exactly as provided.
Do not store or output any additional information beyond the user query and summarizer's response.
""")


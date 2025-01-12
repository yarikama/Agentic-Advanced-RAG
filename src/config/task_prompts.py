# prompts.py
from textwrap import dedent

# HyDE Prompt
HYDE_PROMPT = dedent("""
Your task is to generate 3 - 5 hypothetical answer that could answer the user query.
User Query: "{query}"
""")


# User Query Classification Task
USER_QUERY_CLASSIFICATION_PROMPT = dedent("""
Analyze the following user query and determine if it requires information retrieval to be answered accurately, while also evaluating its global or local scope:

User Query: "{user_query}"

Your task is to:
1. Classify this query as either requiring retrieval or not (Boolean value)
2. Evaluate the query's domain range score (Integer value from 0 to 100)
3. Provide a brief justification for your decision (String value)
4. Pick up the most relevant keywords or entities from the user query (List[str])

Consider the following guidelines:

1. Queries that typically require retrieval:
   - Specific factual information (e.g., historical dates, statistics, temporal events)
   - Detailed explanations of complex topics
   - Information about recent or rapidly changing subjects
   - Query on a specific document, article, object, dataset, person, organization, event, character, etc.
   - Need to retrieve external data to answer the query
   
2. Queries that typically don't require retrieval:
   - General knowledge questions
   - Simple calculations
   - Language translations
   - Requests for creative content generation

3. Domain Range Score (0-100):
    0-20 points: Extremely Specific
        Definition: Very precise questions involving a single operation or concept.
        Example: "How do I adjust the focus on a microscope to view a cell sample?", "In what time, M.C. Escher was born?"
        
    21-40 points: Technical Expertise
        Definition: Questions requiring specific domain knowledge or skills, usually with clear solutions.
        Example: "How do I change the oil filter in a car?"
        
    41-60 points: Concrete Issues
        Definition: Questions about specific problems or situations, potentially involving multiple steps or considerations.
        Example: "How do I grow and care for tomato plants at home?"
        
    61-80 points: Broad Local
        Definition: Questions about details from larger local areas, requiring consideration of multiple subsystems or factors.
        Example: "How can we improve the transportation system of a city?", "Whose stock price is higher, Apple or Samsung in 2021?"
        
    81-100 points: Global
        Definition: Questions about wide-ranging systemic issues, requiring consideration of multiple domains and long-term impacts. To summarize, or to introduce something with complete ingestion.
        Example: "How can we solve the global climate change crisis?", "What is the main topic of this dataset?", "What is the main idea of this book?", "What is this movie about?"
        
4. Relevant Keywords:
    - Pick up the most relevant keywords or entities from the user query.
    - Example: "What is the capital of the United States?" -> ["capital", "United States"]
""")

USER_QUERY_CLASSIFICATION_EXPECTED_OUTPUT = dedent("""
A pydantic object with the following structure:
class UserQueryClassificationResult(BaseModel):
    needs_retrieval: bool
    domain_range_score: int
    justification: str
    relevant_keywords: List[str]
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
class QueryProcessResult(BaseModel):
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

class QueryProcessResult(BaseModel):
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

class QueryProcessResult(BaseModel):
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
Your output should be a pydantic object with the following structure:
class SubQueriesClassificationResult(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]
""")

# Retrieval Task
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

RETRIEVAL_EXPECTED_OUTPUT = dedent("""
A RetrievalResult pydantic object containing consolidated metadata and content lists.
""")

GLOBAL_MAPPING_PROMPT = dedent("""
Your task is to analyze all provided community information and generate two types of outputs related to the user's query.
User Query: "{user_query}"
Sub-queries: "{sub_queries}"

Your specific responsibilities are:

1. Analyze all community information:
   ----------------batch_communities----------------
   {batch_data}

2. Generate two lists:
   a) Key Points List:
      - Create a list of unique and relevant key points derived from all community information.
      - Each key point should either be directly related to the user query or capable of answering it.
      - If all community information does not provide any information related to the user query or sub-queries, return an empty list.
      - Aim for comprehensive coverage while avoiding redundancy.
      - Do not include any other information not related to the user query or sub-queries.
      - Do not tell me that the community information does not provide any information related to the user query or sub-queries.

   b) Imagined Answers List:
      - Based on the community information, imagine possible answers to the user query.
      - These answers should be plausible extensions or interpretations of the available information.
      - Ensure diversity in the imagined answers to cover various possibilities.
      - You don't have to create too many possible answers, 2-3 is enough, focus on the key points.
      - If you don't have enough information to answer the query, just randomly aggregate information from the retrieved data.

Important notes:
- Focus on quality and relevance rather than quantity for both lists.
- Ensure that the key points and imagined answers are distinct from each other.
- Aim for clarity and conciseness in both key points and imagined answers.
- The key points and imagined answers should be related to the user query.
Your output should be a pydantic object with the following structure, do not include datatype like dict in your output:
class GlobalMappingResult(BaseModel):
    communities_summaries: List[str]
    possible_answers: List[str]
""")

GLOBAL_MAPPING_EXPECTED_OUTPUT = dedent("""
class GlobalMappingResult(BaseModel):
    communities_summaries: List[str]
    possible_answers: List[str]
""")

LOCAL_TOPIC_RERANKING_PROMPT = dedent("""
Your task is to evaluate each data's relevance to the user's query or sub-queries relevant to the user's query.
User Query: "{user_query}"
And the sub-queries: "{sub_queries}"

Your specific responsibilities are:
1. Compare each data to the user's query and sub-queries.
2. Assign a relevance score to each data based on how well it matches the user's query and sub-queries from 0 to 100.
   - Higher scores indicate better relevance.
3. Create a list of these relevance scores, don't include any other information.
4. CRITICAL: Ensure the number of scores in your output list EXACTLY matches the number of data :{batch_size} in the input.

You will receive a list of {batch_size} data. 
----------------batch_data----------------
{batch_data}

Important notes:
- The order of your relevance scores must match the exact order of the data in the input.
- Maintain consistency in your scoring method across all communities.
- Do not include any explanations or additional text outside the Pydantic object.
- If you find that your score list does not match the number of data : {batch_size}, you MUST redo the entire process until it does.

FINAL CHECK: Before submitting your response, be sure that the scores list contains exactly {batch_size} relevance scores.
""")

LOCAL_TOPIC_RERANKING_EXPECTED_OUTPUT = dedent("""
class TopicRerankingResult(BaseModel):
    relevant_scores: List[int]                                                                     
""")

# Topic Searching Task
GLOBAL_TOPIC_SEARCHING_PROMPT = dedent("""
You have received multiple pieces of community information related to a user query or sub-queries decomposed from the user query by descending relevance scores.

----------------User Query----------------
{user_query}

----------------Sub-queries----------------
{sub_queries}

----------------Community Information----------------
{data}

Your task is to analyze this information and help prepare for a vector database search to answer the user's query.

Follow these steps:
1. If no community information is provided, return 2 empty lists.
2. Carefully read and analyze all provided community information.
3. Summarize the key points from this information and from the user query into concise community summaries.
4. For generating hypothetical possible answers:
    - First, construct 3-5 sentences that answer the user's query.
    - For example:
        "How does Alex keep Shine away from his castle?" -> "Alex uses a xxx to keep Shine away from his castle." or "Alex took advantage of the xxx to prevent Shine from entering his castle."
        "Where does the 2nd king live?" -> "The 2nd king lives in the xxx." or "At the xxx, the king had a beautiful life."
    - Second, use the retrieved data or summaries to replace the xxx in the sentences.
    - If all the retrieved data is not helpful or you don't have enough information to answer the query, just randomly choose one object mentioned in the retrieved data to replace the xxx.
    - the data picked should be the same type as the xxx in the sentences.(If xxx is probably a location, choose a location; if xxx is probably an object, choose an object; if xxx is probably an action, choose an action)
    - Don't say you don't have enough information to answer the query, just make up a plausible answer.
Return the community summaries and possible answers in a pydantic object:
communities_summaries: List[str]
possible_answers: List[str]
""")

GLOBAL_TOPIC_SEARCHING_EXPECTED_OUTPUT = dedent("""
class GlobalTopicSearchingAndHyDEResult(BaseModel):
    communities_summaries: List[str]
    possible_answers: List[str]
""")


LOCAL_TOPIC_SEARCHING_PROMPT = dedent("""
You have received multiple pieces of data related to a user query or sub-queries decomposed from the user query by descending relevance scores.

----------------User Query----------------
{user_query}

----------------Sub-queries----------------
{sub_queries}

----------------Data----------------
{data}   
Your task is to analyze this batch of data and help prepare for a vector database search to answer the user's query.

The batch of data may include:
- Communities extracted from the article
- Chunks extracted from the article
- Entities extracted from the article
- Relations extracted from the articles

Follow these steps:
1. If no data is provided, return 2 empty list.
2. Carefully read and analyze all provided data.
3. Summarize the key points from this information and from the user query into concise information summaries.
4. For generating hypothetical possible answers:
    - First, construct 3-5 sentences that answer the user's query.
    - For example:
        "How does Alex keep Shine away from his castle?" -> "Alex uses a xxx to keep Shine away from his castle." or "Alex took advantage of the xxx to prevent Shine from entering his castle."
        "Where does the 2nd king live?" -> "The 2nd king lives in the xxx." or "At the xxx, the king had a beautiful life."
    - Second, use the retrieved data or summaries to replace the xxx in the sentences.
    - If all the retrieved data is not helpful or you don't have enough information to answer the query, just randomly choose one object mentioned in the retrieved data to replace the xxx.
    - Don't say you don't have enough information to answer the query, just make up a plausible answer.
    - the data picked should be the same type as the xxx in the sentences.(If xxx is probably a location, choose a location; if xxx is probably an object, choose an object; if xxx is probably an action, choose an action)
Return the community summaries and possible answers in a pydantic object:
information_summaries: List[str]
possible_answers: List[str]
""")

LOCAL_TOPIC_SEARCHING_EXPECTED_OUTPUT = dedent("""
class LocalTopicSearchingAndHyDEResult(BaseModel):
    information_summaries: List[str]
    possible_answers: List[str]
""")

# Retrieve Detail Data from Topic Task
RETRIEVAL_DETAIL_DATA_FROM_TOPIC_PROMPT = dedent("""
specific_collection = {specific_collection}

You will be given a list of TopicSearchingEntity objects. Each object has the following structure:
class TopicSearchingEntity:
    description: str
    score: int
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

# Information Organization Task
INFORMATION_ORGANIZATION_PROMPT = dedent("""
Input:
1. User Query: {user_query}
2. Sub-queries relevant to the user's query: {sub_queries}
3. Retrieved Data: {retrieved_data}

Your tasks:
1. Carefully review the retrieved data.
2. Pick up helpful information from the retrieved data to answer the user's query and remove the irrelevant information.
3. Organize and aggregate the information into strings and preserve the original language, especially modal verbs like "shall", "may", "must", etc., that indicate levels of certainty or possibility.
4. Identify and highlight any connections or contradictions between different pieces of information.
5. Structure the data in a way that facilitates easy understanding, without losing important details or nuances.
6. Include relevant metadata, timestamps, organizing it alongside the corresponding information.
7. Return the organized information in a pydantic object:
8. If the user query requires temporal information, generate an additional string that describes the chronological order of helpful information.

class InformationOrganizationResult(BaseModel):
    organized_information: List[str]
    
Guidelines:
- Maintain the integrity of the original information. Do not add, infer, or fabricate any information not present in the original data.
- Be very careful with temporal information especially for user queries that ask about time.
- If information seems contradictory or inconsistent, note this without attempting to resolve the contradiction.
""")

INFORMATION_ORGANIZATION_EXPECTED_OUTPUT = dedent("""
Your output should be a pydantic object of type InformationOrganizationResult with the following structure:
class InformationOrganizationResult(BaseModel):
    organized_information: List[str]
""")                    

# Response Generation Task
GENERATION_PROMPT = dedent("""
Original user query: {user_query} 
Relevant sub-queries: {sub_queries}
Retrieval Needed: {retrieval_needed}

---Information---
{information}

Your task:
1. If the query contains negations or complex logic, explicitly define your understanding of key terms (e.g., "inconsistent").
2. If Retrieval_Needed is True, and you don't have enough information to generate a response, state that Insufficient information.
3. Otherwise, generate a response based on general knowledge or the information provided.
4. Review the original user query and relevant sub-queries.
5. Carefully examine the data provided for each sub-query, especially temporal, comparative, and conditional information.
6. Structure your response as follows:
   a) Short Answer: At the very beginning of your response, provide a direct, concise answer 
   (e.g., Yes/No for yes/no, did/didn't, have/haven't questions, or specific information for other types of queries.)
   b) Detailed Explanation: Explain your answer comprehensively.
   c) For example: [Yes/No/Specific information] [Comprehensive explanation]
8. Perform a logic check: Ensure your short answer logically aligns with your detailed explanation and the original query.
8. Self-verify: Before finalizing, double-check that your answer directly addresses the original query: {user_query}
""")

GENERATION_EXPECTED_OUTPUT = dedent("""
"A direct answer and a comprehensive analysis answering the user's original query based on all the provided data from sub-queries."
""")


# Response Auditor Task
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
    overall_score: int,
    restart_required: bool,
    additional_comments: Optional[str]
)

Where AuditMetric is structured as:
AuditMetric(
    name: str,
    score: int,
    comment: Optional[str]
)
""")

# Database Updater Task
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
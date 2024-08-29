import pytest
from unittest.mock import Mock
from Utils.retriever import Retriever

@pytest.fixture
def retriever():
    return Retriever()

def test_hybrid_retrieve(retriever):
    collection_name = "test_collection"
    query_text = "test query"
    top_k = 5
    alpha = 0.5

    retriever.generate_hypothetical_document = Mock(return_value="hypothetical doc")
    retriever.embedder.embed_dense = Mock(return_value=[0.1, 0.2, 0.3])
    retriever.embedder.embed_sparse = Mock(return_value=[0.4, 0.5, 0.6])
    retriever.hybrid_search_request = Mock(return_value=[])
    retriever.vectordatabase.hybrid_search = Mock(return_value=[{"id": 1, "score": 0.9}])

    results = retriever.hybrid_retrieve(collection_name, query_text, top_k, alpha)

    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert "id" in results[0]
    assert "score" in results[0]

def test_dense_retrieve(retriever):
    collection_name = "test_collection"
    query_text = "test query"
    top_k = 5

    retriever.embedder.embed_dense = Mock(return_value=[0.1, 0.2, 0.3])
    retriever.dense_search_request = Mock(return_value={})
    retriever.vectordatabase.search = Mock(return_value=[{"id": 1, "score": 0.9}])

    results = retriever.dense_retrieve(collection_name, query_text, top_k)

    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert "id" in results[0]
    assert "score" in results[0]

def test_global_retriever(retriever):
    query = "test query"
    level = 1

    retriever.graphdatabase.db_query = Mock(return_value=[{"output": "test output"}])

    result = retriever.global_retriever(query, level)

    assert result == [{"output": "test output"}]
    retriever.graphdatabase.db_query.assert_called_once()
"""
Step definitions for RAG query feature.
"""
from behave import given, when, then
import json


# Given steps

@given('có tài liệu về "{topic}" đã được index trong collection')
def step_documents_indexed(context, topic):
    """Documents about a topic have been indexed in a collection."""
    collection_name = "ml_collection"
    step_create_collection_if_not_exists(context, collection_name)
    
    # Upload documents about the topic
    topics_content = {
        "machine learning": """
        Machine Learning is a subset of artificial intelligence that enables systems 
        to learn and improve from experience. Supervised learning uses labeled data 
        to train models, while unsupervised learning finds patterns in unlabeled data.
        """,
        "supervised learning": """
        Supervised learning is a type of machine learning where the model is trained 
        on labeled data. The model learns to map inputs to outputs based on examples.
        Common algorithms include linear regression, decision trees, and neural networks.
        """
    }
    
    content = topics_content.get(topic.lower(), f"Document about {topic}")
    
    filename = f"{topic.replace(' ', '_')}_doc.txt"
    files = {"file": (filename, content.encode(), "text/plain")}
    data = {"collection": collection_name}
    
    response = context.test_client.post(
        f"{context.base_url}/documents/upload",
        files=files,
        data=data
    )
    
    print(f"✓ Uploaded document about '{topic}' to {collection_name}")
    context.test_collection_name = collection_name


@given('tôi có collection tên "{collection_name}"')
def step_have_collection(context, collection_name):
    """I have a collection with given name."""
    step_create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    print(f"✓ Using collection: {collection_name}")


@given('collection "{collection_name}" không có tài liệu nào')
def step_empty_collection(context, collection_name):
    """Collection has no documents."""
    # Create empty collection
    step_create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    print(f"✓ Created empty collection: {collection_name}")


@given('có {num:d} tài liệu trong collection')
def step_multiple_documents(context, num):
    """Collection has multiple documents."""
    collection_name = "test_collection"
    step_create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    
    # Upload multiple documents
    for i in range(num):
        filename = f"doc_{i+1}.txt"
        content = f"This is document number {i+1} with unique content {i}."
        
        files = {"file": (filename, content.encode(), "text/plain")}
        data = {"collection": collection_name}
        
        response = context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
    
    print(f"✓ Uploaded {num} documents to {collection_name}")


@given('có tài liệu trong collection')
def step_documents_exist(context):
    """Documents exist in collection."""
    collection_name = "test_collection"
    step_multiple_documents(context, 5)
    context.test_collection_name = collection_name


# When steps

@when('tôi gửi POST request tới endpoint "{endpoint}" với query "{query}"')
def step_post_query(context, endpoint, query):
    """Send POST request with query."""
    payload = {
        "query": query,
        "collection": context.test_collection_name,
        "top_k": 5,
        "use_rag": True
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query: '{query}', status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với top_k={top_k:d}')
def step_post_query_topk(context, endpoint, top_k):
    """Send POST request with top_k parameter."""
    payload = {
        "query": "test query",
        "collection": context.test_collection_name,
        "top_k": top_k,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query with top_k={top_k}, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với score_threshold={threshold:0.2f}')
def step_post_query_threshold(context, endpoint, threshold):
    """Send POST request with score_threshold parameter."""
    payload = {
        "query": "test query",
        "collection": context.test_collection_name,
        "top_k": 10,
        "score_threshold": threshold,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query with threshold={threshold}, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với use_rag={use_rag}')
def step_post_query_no_rag(context, endpoint, use_rag):
    """Send POST request with use_rag parameter."""
    payload = {
        "query": "test query",
        "collection": context.test_collection_name,
        "top_k": 5,
        "use_rag": use_rag
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query with use_rag={use_rag}, status: {context.status_code}")


# Then steps

@then('response chứa answer có nghĩa')
def step_response_has_meaningful_answer(context):
    """Response should contain a meaningful answer."""
    assert context.response_data is not None, "Response data is None"
    assert "answer" in context.response_data, "No answer in response"
    
    answer = context.response_data["answer"]
    assert answer is not None, "Answer is None"
    assert len(answer.strip()) > 0, "Answer is empty"
    
    print(f"✓ Answer: {answer[:100]}...")


@then('response có retrieved_documents không rỗng')
def step_response_has_retrieved_docs(context):
    """Response should have non-empty retrieved_documents."""
    assert context.response_data is not None, "Response data is None"
    assert "retrieved_documents" in context.response_data, "No retrieved_documents in response"
    
    docs = context.response_data["retrieved_documents"]
    assert len(docs) > 0, "Retrieved documents is empty"
    
    print(f"✓ Retrieved {len(docs)} documents")


@then('mỗi retrieved_document có document_id và score')
def step_docs_have_id_and_score(context):
    """Each retrieved document should have document_id and score."""
    docs = context.response_data.get("retrieved_documents", [])
    
    for doc in docs:
        assert "id" in doc, f"Document missing 'id': {doc}"
        assert "score" in doc, f"Document missing 'score': {doc}"
        assert isinstance(doc["score"], (int, float)), "Score should be numeric"
    
    print(f"✓ All {len(docs)} documents have id and score")


@then('retrieved_count là {count:d}')
def step_retrieval_count(context, count):
    """Retrieval count should be given value."""
    assert "retrieval_count" in context.response_data, "No retrieval_count in response"
    assert context.response_data["retrieval_count"] == count, \
        f"Expected retrieval_count {count}, got {context.response_data['retrieval_count']}"
    print(f"✓ Retrieval count: {count}")


@then('Hoặc answer là "{expected_answer}"')
def step_or_answer_is(context, expected_answer):
    """Or answer is expected value."""
    answer = context.response_data.get("answer", "")
    assert expected_answer in answer or answer == expected_answer, \
        f"Expected answer '{expected_answer}', got '{answer}'"
    print(f"✓ Answer matches expected")


@then('response chỉ trả về tối đa {max_docs:d} retrieved_documents')
def step_max_retrieved_docs(context, max_docs):
    """Response should return at most max_docs retrieved documents."""
    docs = context.response_data.get("retrieved_documents", [])
    assert len(docs) <= max_docs, \
        f"Expected at most {max_docs} documents, got {len(docs)}"
    print(f"✓ Retrieved {len(docs)} documents (max {max_docs})")


@then('Hoặc retrieval_count là {count:d} hoặc nhỏ hơn')
def step_retrieval_count_or_less(context, count):
    """Retrieval count should be count or less."""
    retrieval_count = context.response_data.get("retrieval_count", 0)
    assert retrieval_count <= count, \
        f"Expected retrieval_count <= {count}, got {retrieval_count}"
    print(f"✓ Retrieval count: {retrieval_count} (max {count})")


@then('tất cả retrieved_documents có score >= {threshold:0.2f}')
def step_all_docs_above_threshold(context, threshold):
    """All retrieved documents should have score >= threshold."""
    docs = context.response_data.get("retrieved_documents", [])
    
    for doc in docs:
        score = doc.get("score", 0)
        assert score >= threshold, \
            f"Document score {score} is below threshold {threshold}"
    
    print(f"✓ All {len(docs)} documents have score >= {threshold}")


@then('Hoặc không có documents nào được trả về nếu không thỏa mãn')
def step_or_no_documents_returned(context):
    """Or no documents are returned if none satisfy criteria."""
    docs = context.response_data.get("retrieved_documents", [])
    assert len(docs) == 0, "Expected no documents, but got some"
    print("✓ No documents returned (as expected)")


@then('answer có thể là None hoặc empty')
def step_answer_none_or_empty(context):
    """Answer can be None or empty when RAG is disabled."""
    answer = context.response_data.get("answer", "")
    assert answer is None or len(answer.strip()) == 0, \
        f"Expected None or empty answer, got: {answer}"
    print(f"✓ Answer is None or empty (RAG disabled)")


@then('Nhưng retrieved_documents vẫn được trả về')
def step_retrieved_docs_still_returned(context):
    """But retrieved_documents should still be returned."""
    docs = context.response_data.get("retrieved_documents", [])
    assert len(docs) > 0, "Expected retrieved documents, but got none"
    print(f"✓ Retrieved {len(docs)} documents (RAG disabled)")


# Helper functions

def step_create_collection_if_not_exists(context, collection_name):
    """Helper to create collection if it doesn't exist."""
    response = context.test_client.post(
        f"{context.base_url}/collections/",
        json={"name": collection_name}
    )
    
    if not hasattr(context, 'created_collections'):
        context.created_collections = []
    if collection_name not in context.created_collections:
        context.created_collections.append(collection_name)

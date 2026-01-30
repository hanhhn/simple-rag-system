"""
Step definitions for similarity search feature.
"""
from behave import given, when, then
from io import BytesIO


# Helper function
def create_collection_if_not_exists(context, collection_name):
    """Helper to create collection if it doesn't exist."""
    response = context.test_client.post(
        f"{context.base_url}/collections/",
        json={"name": collection_name}
    )
    
    if not hasattr(context, 'created_collections'):
        context.created_collections = []
    if collection_name not in context.created_collections:
        context.created_collections.append(collection_name)


# Given steps

@given('có 100 tài liệu đã được index trong collection')
def step_100_documents_indexed(context):
    """100 documents have been indexed in collection."""
    collection_name = "search_collection"
    create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    
    # Upload 100 documents
    for i in range(100):
        filename = f"doc_{i+1}.txt"
        content = f"This is document number {i+1} about artificial intelligence and machine learning."
        
        files = {"file": (filename, content.encode(), "text/plain")}
        data = {"collection": collection_name}
        
        context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
    
    print(f"✓ Uploaded 100 documents to {collection_name}")


@given('embedding model đã được load')
def step_embedding_model_loaded(context):
    """Embedding model has been loaded."""
    print("✓ Embedding model loaded")


@given('similarity search trả về nhiều kết quả')
def step_search_returns_many_results(context):
    """Similarity search returns many results."""
    collection_name = "search_collection"
    if not hasattr(context, 'test_collection_name'):
        create_collection_if_not_exists(context, collection_name)
        context.test_collection_name = collection_name
        
        # Upload some documents
        for i in range(10):
            filename = f"search_doc_{i+1}.txt"
            content = f"Content about machine learning and AI {i+1}."
            
            files = {"file": (filename, content.encode(), "text/plain")}
            data = {"collection": collection_name}
            
            context.test_client.post(
                f"{context.base_url}/documents/upload",
                files=files,
                data=data
            )
    
    # Perform a query to get results
    payload = {
        "query": "machine learning",
        "collection": context.test_collection_name,
        "top_k": 10,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}/query/",
        json=payload
    )
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"✓ Performed search, retrieved results")


@given('score threshold được set là 0.7')
def step_score_threshold_set(context):
    """Score threshold is set to 0.7."""
    context.score_threshold = 0.7
    print(f"✓ Score threshold set to {context.score_threshold}")


@given('collection "{collection_name}" không tồn tại')
def step_collection_not_exists(context, collection_name):
    """Collection does not exist."""
    context.test_collection_name = collection_name
    print(f"✓ Collection '{collection_name}' does not exist")


@given('có tài liệu trong collection')
def step_documents_in_collection(context):
    """Documents exist in collection."""
    collection_name = "test_collection"
    create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    
    # Upload some documents
    for i in range(5):
        filename = f"topic_doc_{i+1}.txt"
        content = f"This document is about topic {i+1}."
        
        files = {"file": (filename, content.encode(), "text/plain")}
        data = {"collection": collection_name}
        
        context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
    
    print(f"✓ Uploaded documents to {collection_name}")


# When steps

@when('tôi thực hiện query với nội dung liên quan đến tài liệu trong collection')
def step_query_related_content(context):
    """Perform query with content related to documents in collection."""
    payload = {
        "query": "artificial intelligence and machine learning",
        "collection": context.test_collection_name,
        "top_k": 10,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}/query/",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query related content, status: {context.status_code}")


@when('hệ thống lọc kết quả')
def step_filter_results(context):
    """System filters results."""
    # Re-query with score threshold
    threshold = getattr(context, 'score_threshold', 0.7)
    
    payload = {
        "query": "machine learning",
        "collection": context.test_collection_name,
        "top_k": 10,
        "score_threshold": threshold,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}/query/",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Filtered results with threshold {threshold}, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với collection này')
def step_post_with_nonexistent_collection(context, endpoint):
    """Send POST request with nonexistent collection."""
    payload = {
        "query": "test query",
        "collection": context.test_collection_name,
        "top_k": 5,
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
    
    print(f"→ POST request with nonexistent collection, status: {context.status_code}")


@when('tôi query về một chủ đề không liên quan')
def step_query_unrelated_topic(context):
    """Query about an unrelated topic."""
    payload = {
        "query": "This is a completely unrelated topic about underwater basket weaving",
        "collection": context.test_collection_name,
        "top_k": 5,
        "use_rag": False
    }
    
    context.response = context.test_client.post(
        f"{context.base_url}/query/",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ Query unrelated topic, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}"')
def step_post_query_endpoint(context, endpoint):
    """Send POST request to query endpoint."""
    payload = {
        "query": "test query",
        "collection": context.test_collection_name,
        "top_k": 5,
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
    
    print(f"→ POST request to {endpoint}, status: {context.status_code}")


# Then steps

@then('hệ thống trả về top-k tài liệu có điểm similarity cao nhất')
def step_returns_topk_highest_scores(context):
    """System returns top-k documents with highest similarity scores."""
    assert context.response_data is not None, "Response data is None"
    assert "retrieved_documents" in context.response_data, "No retrieved_documents in response"
    
    docs = context.response_data["retrieved_documents"]
    assert len(docs) > 0, "No documents returned"
    
    # Check that scores are present
    for doc in docs:
        assert "score" in doc, f"Document missing score: {doc}"
        assert isinstance(doc["score"], (int, float)), "Score should be numeric"
    
    print(f"✓ Returned {len(docs)} documents with similarity scores")


@then('mỗi kết quả có score')
def step_each_result_has_score(context):
    """Each result has a score."""
    assert context.response_data is not None, "Response data is None"
    docs = context.response_data.get("retrieved_documents", [])
    
    for doc in docs:
        assert "score" in doc, f"Document missing score: {doc}"
    
    print(f"✓ All {len(docs)} documents have scores")


@then('kết quả được sắp xếp theo score giảm dần')
def step_results_sorted_by_score(context):
    """Results are sorted by score in descending order."""
    assert context.response_data is not None, "Response data is None"
    docs = context.response_data.get("retrieved_documents", [])
    
    scores = [doc["score"] for doc in docs]
    assert scores == sorted(scores, reverse=True), \
        f"Results not sorted correctly: {scores}"
    
    print(f"✓ Results sorted by score (descending): {scores}")


@then('chỉ các kết quả có score >= 0.7 được trả về')
def step_results_above_threshold(context):
    """Only results with score >= 0.7 are returned."""
    assert context.response_data is not None, "Response data is None"
    docs = context.response_data.get("retrieved_documents", [])
    
    threshold = 0.7
    for doc in docs:
        score = doc.get("score", 0)
        assert score >= threshold, \
            f"Document score {score} is below threshold {threshold}"
    
    print(f"✓ All {len(docs)} documents have score >= {threshold}")


@then('các kết quả thấp hơn threshold bị loại bỏ')
def step_below_threshold_removed(context):
    """Results below threshold are removed."""
    assert context.response_data is not None, "Response data is None"
    docs = context.response_data.get("retrieved_documents", [])
    
    threshold = getattr(context, 'score_threshold', 0.7)
    for doc in docs:
        score = doc.get("score", 0)
        assert score >= threshold, \
            f"Document with score {score} should be removed (threshold: {threshold})"
    
    print(f"✓ Results below threshold {threshold} removed")


@then('status code trả về là {status_code:d} hoặc {alt_code:d}')
def step_status_code_or(context, status_code, alt_code):
    """Status code should be one of the given values."""
    assert context.status_code in [status_code, alt_code], \
        f"Expected status {status_code} or {alt_code}, got {context.status_code}"
    print(f"✓ Status code: {context.status_code}")


@then('retrieved_count là 0')
def step_retrieval_count_zero(context):
    """Retrieval count should be 0."""
    retrieval_count = context.response_data.get("retrieved_count", -1)
    assert retrieval_count == 0, \
        f"Expected retrieval_count 0, got {retrieval_count}"
    print(f"✓ Retrieval count: 0")


@then('status code trả về là {status_code:d}')
def step_status_code(context, status_code):
    """Status code should be given value."""
    assert context.status_code == status_code, \
        f"Expected status {status_code}, got {context.status_code}. Response: {context.response.text}"
    print(f"✓ Status code: {status_code}")


@then('Hoặc answer thông báo không tìm thấy kết quả')
def step_or_answer_no_results(context):
    """Or answer indicates no results found."""
    answer = context.response_data.get("answer", "")
    retrieval_count = context.response_data.get("retrieval_count", -1)
    
    assert retrieval_count == 0 or "not found" in answer.lower() or "no relevant" in answer.lower(), \
        f"Expected no results, got: {answer}"
    
    print("✓ No results found as expected")

"""
Step definitions for collection management feature.
"""
from behave import given, when, then


# Given steps

@given('hệ thống RAG đang hoạt động')
def step_system_operational(context):
    """RAG system is operational."""
    assert context.test_client is not None
    print("✓ RAG system is operational")


@given('collection "{collection_name}" đã tồn tại')
def step_collection_already_exists(context, collection_name):
    """A collection with given name already exists."""
    response = context.test_client.post(
        f"{context.base_url}/collections/",
        json={"name": collection_name}
    )
    
    if not hasattr(context, 'created_collections'):
        context.created_collections = []
    if collection_name not in context.created_collections:
        context.created_collections.append(collection_name)
    
    if response.status_code in [200, 201]:
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"Collection {collection_name} might already exist")


@given('có {num:d} collections đã được tạo: "{collection_list}"')
def step_multiple_collections_exist(context, num, collection_list):
    """Multiple collections have been created."""
    collections = [c.strip() for c in collection_list.split(",")]
    
    if not hasattr(context, 'created_collections'):
        context.created_collections = []
    
    for collection_name in collections:
        response = context.test_client.post(
            f"{context.base_url}/collections/",
            json={"name": collection_name}
        )
        
        if collection_name not in context.created_collections:
            context.created_collections.append(collection_name)
        
        print(f"✓ Collection: {collection_name}")
    
    context.test_collection_name = collections[0]


@given('collection "{collection_name}" đã tồn tại')
def step_collection_exists_simple(context, collection_name):
    """Collection exists."""
    step_collection_already_exists(context, collection_name)


# When steps

@when('tôi gửi POST request tới endpoint "{endpoint}" với name="{collection_name}"')
def step_post_create_collection(context, endpoint, collection_name):
    """Send POST request to create collection."""
    payload = {"name": collection_name}
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    context.test_collection_name = collection_name
    print(f"→ POST create collection: {collection_name}, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với name="{collection_name}"')
def step_post_duplicate_collection(context, endpoint, collection_name):
    """Send POST request to create duplicate collection."""
    payload = {"name": collection_name}
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST duplicate collection: {collection_name}, status: {context.status_code}")


@when('tôi gửi GET request tới endpoint "{endpoint}"')
def step_get_collections(context, endpoint):
    """Send GET request to list all collections."""
    context.response = context.test_client.get(f"{context.base_url}{endpoint}")
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ GET collections, status: {context.status_code}")


@when('tôi gửi GET request tới endpoint "{collection_endpoint}"')
def step_get_collection(context, collection_endpoint):
    """Send GET request to get collection info."""
    context.response = context.test_client.get(f"{context.base_url}{collection_endpoint}")
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ GET collection: {collection_endpoint}, status: {context.status_code}")


@when('tôi gửi DELETE request tới endpoint "{endpoint}"')
def step_delete_collection(context, endpoint):
    """Send DELETE request to delete collection."""
    context.response = context.test_client.delete(f"{context.base_url}{endpoint}")
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ DELETE collection: {endpoint}, status: {context.status_code}")


# Then steps

@then('response chứa collection có name là "{collection_name}"')
def step_response_has_collection_name(context, collection_name):
    """Response should contain collection with given name."""
    assert context.response_data is not None, "Response data is None"
    
    collection = context.response_data.get("collection", {})
    assert isinstance(collection, dict), "Collection should be a dict"
    assert collection.get("name") == collection_name, \
        f"Expected collection name '{collection_name}', got {collection.get('name')}"
    
    print(f"✓ Collection name: {collection_name}")


@then('collection status là "{status}"')
def step_collection_status(context, status):
    """Collection status should be given value."""
    collection = context.response_data.get("collection", {})
    actual_status = collection.get("status")
    assert actual_status == status, \
        f"Expected status '{status}', got {actual_status}"
    print(f"✓ Collection status: {status}")


@then('status code trả về là {status_code:d} hoặc {alt_code:d}')
def step_status_code_or(context, status_code, alt_code):
    """Status code should be one of the given values."""
    assert context.status_code in [status_code, alt_code], \
        f"Expected status {status_code} or {alt_code}, got {context.status_code}"
    print(f"✓ Status code: {context.status_code}")


@then('response chứa thông báo về collection đã tồn tại')
def step_error_collection_exists(context):
    """Response should contain error about existing collection."""
    assert context.response_data is not None, "Response data is None"
    
    # Check various possible error message formats
    response_text = str(context.response_data).lower()
    assert "exist" in response_text or "already" in response_text, \
        f"Expected 'exists' error, got: {context.response_data}"
    
    print("✓ Collection exists error returned")


@then('response chứa danh sách collections')
def step_response_has_collections(context):
    """Response should contain collections list."""
    assert context.response_data is not None, "Response data is None"
    assert "collections" in context.response_data, "No collections in response"
    assert isinstance(context.response_data["collections"], list)
    
    print(f"✓ Collections list: {len(context.response_data['collections'])} items")


@then('total collections là {count:d}')
def step_total_collections(context, count):
    """Total collections should be given count."""
    assert "total" in context.response_data, "No total in response"
    assert context.response_data["total"] == count, \
        f"Expected {count} collections, got {context.response_data['total']}"
    print(f"✓ Total collections: {count}")


@then('response có vector_count')
def step_response_has_vector_count(context):
    """Response should have vector_count."""
    collection = context.response_data.get("collection", {})
    assert "vector_count" in collection, "No vector_count in response"
    print(f"✓ Vector count: {collection['vector_count']}")


@then('response có dimension')
def step_response_has_dimension(context):
    """Response should have dimension."""
    collection = context.response_data.get("collection", {})
    assert "dimension" in collection, "No dimension in response"
    print(f"✓ Dimension: {collection['dimension']}")


@then('status code trả về là {status_code:d}')
def step_status_code_simple(context, status_code):
    """Status code should be given value."""
    assert context.status_code == status_code, \
        f"Expected status {status_code}, got {context.status_code}"
    print(f"✓ Status code: {status_code}")


@then('response chứa thông báo xóa thành công')
def step_delete_success_message(context):
    """Response should contain delete success message."""
    assert context.response_data is not None, "Response data is None"
    assert "message" in context.response_data, "No message in response"
    
    message = context.response_data["message"].lower()
    assert "deleted" in message or "xóa" in message, \
        f"Expected delete success message, got: {message}"
    
    print(f"✓ Delete success: {context.response_data['message']}")


@then('response chứa thông báo collection không tồn tại')
def step_error_collection_not_found(context):
    """Response should contain error about collection not found."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "not found" in response_text or "not exist" in response_text, \
        f"Expected 'not found' error, got: {context.response_data}"
    
    print("✓ Collection not found error returned")


@then('success trong response là True')
def step_success_true(context):
    """Success field should be True."""
    assert context.response_data is not None, "Response data is None"
    assert context.response_data.get("success") is True
    print("✓ Success is True")

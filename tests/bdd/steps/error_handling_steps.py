"""
Step definitions for error handling and validation feature.
"""
from behave import given, when, then
from io import BytesIO


# Given steps

@given('hệ thống đang hoạt động')
def step_system_operational(context):
    """System is operational."""
    assert context.test_client is not None
    print("✓ System is operational")


@given('hệ thống có giới hạn file size là 10MB')
def step_system_file_size_limit(context):
    """System has file size limit of 10MB."""
    # This is already configured in config
    print("✓ File size limit configured (10MB)")


@given('tôi có một file lớn hơn 10MB')
def step_have_large_file(context):
    """I have a file larger than 10MB."""
    context.test_filename = "large_file.pdf"
    context.test_file_content = b"X" * (10 * 1024 * 1024 + 1)  # 10MB + 1 byte
    context.test_content_type = "application/pdf"
    print(f"✓ Created large file ({len(context.test_file_content)} bytes)")


@given('hệ thống chỉ hỗ trợ PDF, TXT, MD, DOCX')
def step_system_supported_formats(context):
    """System supports only specific file formats."""
    print("✓ Supported formats: PDF, TXT, MD, DOCX")


@given('tôi có một file .xyz không được hỗ trợ')
def step_have_unsupported_file(context):
    """I have an unsupported .xyz file."""
    context.test_filename = "unsupported_file.xyz"
    context.test_file_content = b"Unsupported file content."
    context.test_content_type = "application/octet-stream"
    print("✓ Created unsupported file: .xyz")


# When steps

@when('tôi gửi POST request tới endpoint "{endpoint}" với query rỗng')
def step_post_empty_query(context, endpoint):
    """Send POST request with empty query."""
    payload = {
        "query": "",
        "collection": "test_collection",
        "top_k": 5
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
    
    print(f"→ POST request with empty query, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với query dài hơn 1000 ký tự')
def step_post_long_query(context, endpoint):
    """Send POST request with query longer than 1000 characters."""
    long_query = "a" * 1001
    payload = {
        "query": long_query,
        "collection": "test_collection",
        "top_k": 5
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
    
    print(f"→ POST request with long query ({len(long_query)} chars), status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với top_k=0')
def step_post_topk_zero(context, endpoint):
    """Send POST request with top_k=0."""
    payload = {
        "query": "test query",
        "collection": "test_collection",
        "top_k": 0
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
    
    print(f"→ POST request with top_k=0, status: {context.status_code}")


@when('tôi gửi POST request với top_k=10000')
def step_post_topk_large(context):
    """Send POST request with top_k=10000."""
    payload = {
        "query": "test query",
        "collection": "test_collection",
        "top_k": 10000
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
    
    print(f"→ POST request with top_k=10000, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với name rỗng')
def step_post_empty_name(context, endpoint):
    """Send POST request with empty name."""
    payload = {"name": ""}
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST request with empty name, status: {context.status_code}")


@when('tôi gửi POST request với name có ký tự đặc biệt không hợp lệ')
def step_post_invalid_name(context):
    """Send POST request with invalid special characters in name."""
    payload = {"name": "invalid@name#"}
    
    context.response = context.test_client.post(
        f"{context.base_url}/collections/",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST request with invalid name, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với file này')
def step_post_large_file(context, endpoint):
    """Send POST request with large file."""
    files = {"file": (context.test_filename, BytesIO(context.test_file_content), context.test_content_type)}
    data = {"collection": "test_collection"}
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        files=files,
        data=data
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST request with large file, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với dimension=0')
def step_post_dimension_zero(context, endpoint):
    """Send POST request with dimension=0."""
    payload = {"name": "test_collection", "dimension": 0}
    
    context.response = context.test_client.post(
        f"{context.base_url}{endpoint}",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST request with dimension=0, status: {context.status_code}")


@when('tôi gửi POST request với dimension=10000')
def step_post_dimension_large(context):
    """Send POST request with dimension=10000."""
    payload = {"name": "test_collection", "dimension": 10000}
    
    context.response = context.test_client.post(
        f"{context.base_url}/collections/",
        json=payload
    )
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ POST request with dimension=10000, status: {context.status_code}")


# Then steps

@then('status code trả về là {status_code:d}')
def step_status_code(context, status_code):
    """Status code should be given value."""
    assert context.status_code == status_code, \
        f"Expected status {status_code}, got {context.status_code}. Response: {context.response.text}"
    print(f"✓ Status code is {status_code}")


@then('response chứa thông báo validation error')
def step_error_validation(context):
    """Response should contain validation error."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "validation" in response_text or "invalid" in response_text or "error" in response_text, \
        f"Expected validation error, got: {context.response_data}"
    print("✓ Validation error returned")


@then('response chứa thông báo về query quá dài')
def step_error_query_too_long(context):
    """Response should contain error about query being too long."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "too long" in response_text or "maximum" in response_text or "exceeds" in response_text, \
        f"Expected 'too long' error, got: {context.response_data}"
    print("✓ Query too long error returned")


@then('response chứa thông báo về top_k quá lớn')
def step_error_topk_large(context):
    """Response should contain error about top_k being too large."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "exceed" in response_text or "maximum" in response_text or "cannot" in response_text, \
        f"Expected 'top_k too large' error, got: {context.response_data}"
    print("✓ Top_k too large error returned")


@then('response chứa thông báo về name không hợp lệ')
def step_error_invalid_name(context):
    """Response should contain error about invalid name."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "invalid" in response_text or "special" in response_text or "character" in response_text, \
        f"Expected 'invalid name' error, got: {context.response_data}"
    print("✓ Invalid name error returned")


@then('response chứa thông báo về file size quá lớn')
def step_error_file_too_large(context):
    """Response should contain error about file size being too large."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "size" in response_text or "exceed" in response_text or "maximum" in response_text, \
        f"Expected 'file too large' error, got: {context.response_data}"
    print("✓ File too large error returned")


@then('response chứa thông báo về file type không được hỗ trợ')
def step_error_unsupported_type(context):
    """Response should contain error about unsupported file type."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "unsupport" in response_text or "format" in response_text or "not support" in response_text, \
        f"Expected 'unsupported format' error, got: {context.response_data}"
    print("✓ Unsupported format error returned")


@then('response chứa thông báo về dimension quá lớn')
def step_error_dimension_large(context):
    """Response should contain error about dimension being too large."""
    assert context.response_data is not None, "Response data is None"
    
    response_text = str(context.response_data).lower()
    assert "too large" in response_text or "exceed" in response_text or "maximum" in response_text, \
        f"Expected 'dimension too large' error, got: {context.response_data}"
    print("✓ Dimension too large error returned")

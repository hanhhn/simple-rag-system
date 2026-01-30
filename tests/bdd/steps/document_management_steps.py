"""
Step definitions for document management feature.
"""
from behave import given, when, then
from io import BytesIO
import json
from pathlib import Path


# Given steps

@given('hệ thống RAG đã được khởi động và hoạt động bình thường')
def step_system_running(context):
    """RAG system is running and operational."""
    # TestClient is already initialized in environment.py
    assert context.test_client is not None
    print("✓ RAG system is running")


@given('tôi có một tài liệu PDF hợp lệ')
def step_have_pdf_document(context):
    """I have a valid PDF document."""
    # Create a simple PDF-like content
    context.test_file_content = b"This is a test PDF document content about machine learning and AI research."
    context.test_filename = "test_document.pdf"
    context.test_content_type = "application/pdf"
    print(f"✓ Created test document: {context.test_filename}")


@given('tôi có một collection tên "{collection_name}" đã tồn tại')
def step_collection_exists(context, collection_name):
    """A collection with given name already exists."""
    # Create the collection if it doesn't exist
    response = context.test_client.post(
        f"{context.base_url}/collections/",
        json={"name": collection_name}
    )
    
    # Track for cleanup
    if not hasattr(context, 'created_collections'):
        context.created_collections = []
    if collection_name not in context.created_collections:
        context.created_collections.append(collection_name)
    
    # Collection might already exist (200 or 201)
    if response.status_code in [200, 201]:
        print(f"✓ Collection '{collection_name}' created")
    elif response.status_code == 500 and "already exists" in response.text:
        print(f"✓ Collection '{collection_name}' already exists")
    else:
        print(f"Response: {response.status_code}, {response.text}")


@given('một tài liệu đã được tải lên và xử lý xong')
def step_document_uploaded_and_processed(context):
    """A document has been uploaded and processed."""
    # First create collection
    step_collection_exists(context, "test_collection")
    
    # Upload a test document
    context.test_filename = "processed_doc.pdf"
    context.test_file_content = b"This is a document that has been fully processed and indexed."
    
    files = {"file": (context.test_filename, BytesIO(context.test_file_content), "application/pdf")}
    data = {"collection": "test_collection"}
    
    response = context.test_client.post(
        f"{context.base_url}/documents/upload",
        files=files,
        data=data
    )
    
    context.response = response
    context.status_code = response.status_code
    print(f"✓ Document uploaded with status: {context.status_code}")


@given('tài liệu có filename là "{filename}" trong collection "{collection_name}"')
def step_document_with_filename_in_collection(context, filename, collection_name):
    """Document with filename exists in collection."""
    step_collection_exists(context, collection_name)
    context.test_filename = filename
    context.test_file_content = b"Test document content."
    
    files = {"file": (filename, BytesIO(context.test_file_content), "application/pdf")}
    data = {"collection": collection_name}
    
    response = context.test_client.post(
        f"{context.base_url}/documents/upload",
        files=files,
        data=data
    )
    print(f"✓ Document '{filename}' uploaded to '{collection_name}'")


@given('có {num:d} tài liệu đã được tải lên trong collection "{collection_name}"')
def step_multiple_documents_in_collection(context, num, collection_name):
    """Multiple documents have been uploaded in collection."""
    step_collection_exists(context, collection_name)
    
    if not hasattr(context, 'uploaded_documents'):
        context.uploaded_documents = []
    
    for i in range(num):
        filename = f"document_{i+1}.pdf"
        content = f"This is document number {i+1} content.".encode()
        
        files = {"file": (filename, BytesIO(content), "application/pdf")}
        data = {"collection": collection_name}
        
        response = context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
        context.uploaded_documents.append(filename)
        print(f"✓ Uploaded {filename}")


@given('tài liệu "{filename}" đã tồn tại trong collection "{collection_name}"')
def step_document_exists_in_collection(context, filename, collection_name):
    """Document exists in collection."""
    step_document_with_filename_in_collection(context, filename, collection_name)


@given('hệ thống RAG đã được khởi động')
def step_system_started(context):
    """RAG system has been started."""
    assert context.test_client is not None
    print("✓ RAG system started")


@given('tôi có một file định dạng không được hỗ trợ là "{filename}"')
def step_unsupported_file_format(context, filename):
    """I have a file with unsupported format."""
    context.test_filename = filename
    context.test_file_content = b"This is an unsupported file format."
    context.test_content_type = "application/octet-stream"
    print(f"✓ Created unsupported file: {filename}")


# When steps

@when('tôi gửi POST request tới endpoint "{endpoint}" với file và collection name')
def step_post_document_upload(context, endpoint):
    """Send POST request to upload document."""
    files = {"file": (context.test_filename, BytesIO(context.test_file_content), context.test_content_type)}
    data = {"collection": "research_docs"}
    
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
    
    print(f"→ POST request to {endpoint}, status: {context.status_code}")


@when('tôi gửi POST request tới endpoint "{endpoint}" với file')
def step_post_upload_unsupported(context, endpoint):
    """Send POST request with unsupported file."""
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
    
    print(f"→ POST request to {endpoint} with unsupported file, status: {context.status_code}")


@when('tôi gửi DELETE request tới endpoint "{endpoint}"')
def step_delete_document(context, endpoint):
    """Send DELETE request to document."""
    context.response = context.test_client.delete(f"{context.base_url}{endpoint}")
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ DELETE request to {endpoint}, status: {context.status_code}")


@when('tôi gửi GET request tới endpoint "{endpoint}"')
def step_get_request(context, endpoint):
    """Send GET request."""
    context.response = context.test_client.get(f"{context.base_url}{endpoint}")
    context.status_code = context.response.status_code
    
    try:
        context.response_data = context.response.json()
    except:
        context.response_data = None
    
    print(f"→ GET request to {endpoint}, status: {context.status_code}")


# Then steps

@then('status code trả về là {status_code:d}')
def step_status_code(context, status_code):
    """Status code should be the given value."""
    assert context.status_code == status_code, \
        f"Expected status {status_code}, got {context.status_code}. Response: {context.response.text}"
    print(f"✓ Status code is {status_code}")


@then('response chứa document_id hợp lệ')
def step_response_has_document_id(context):
    """Response should contain a valid document_id."""
    assert context.response_data is not None, "Response data is None"
    assert "document_id" in context.response_data, "No document_id in response"
    assert context.response_data["document_id"] is not None
    print(f"✓ Document ID: {context.response_data['document_id']}")


@then('response có filename đúng với file đã upload')
def step_response_has_correct_filename(context):
    """Response should have correct filename."""
    assert "filename" in context.response_data, "No filename in response"
    assert context.response_data["filename"] == context.test_filename
    print(f"✓ Filename matches: {context.response_data['filename']}")


@then('response có collection_name là "{collection_name}"')
def step_response_has_collection_name(context, collection_name):
    """Response should have correct collection_name."""
    assert "collection" in context.response_data or "collection_name" in context.response_data, \
        "No collection field in response"
    
    collection_field = context.response_data.get("collection") or context.response_data.get("collection_name")
    assert collection_field == collection_name
    print(f"✓ Collection name: {collection_field}")


@then('chunk_count lớn hơn {min_count:d}')
def step_chunk_count_greater_than(context, min_count):
    """chunk_count should be greater than given value."""
    assert "chunk_count" in context.response_data, "No chunk_count in response"
    assert context.response_data["chunk_count"] > min_count
    print(f"✓ Chunk count: {context.response_data['chunk_count']}")


@then('response chứa thông báo lỗi về định dạng file không hỗ trợ')
def step_error_unsupported_format(context):
    """Response should contain error about unsupported format."""
    assert context.response_data is not None, "Response data is None"
    detail = context.response_data.get("detail", {})
    
    # Check for error message in various formats
    error_message = str(detail) if isinstance(detail, (str, dict)) else ""
    assert "unsupport" in error_message.lower() or "format" in error_message.lower() or \
           "validate" in error_message.lower(), \
           f"Expected unsupported format error, got: {error_message}"
    print(f"✓ Unsupported format error returned")


@then('success trong response là True')
def step_success_is_true(context):
    """Success field in response should be True."""
    assert context.response_data is not None, "Response data is None"
    assert context.response_data.get("success") is True, \
        f"Expected success=True, got: {context.response_data}"
    print(f"✓ Success is True")


@then('response chứa thông báo xóa thành công')
def step_delete_success_message(context):
    """Response should contain success message."""
    assert context.response_data is not None, "Response data is None"
    assert "message" in context.response_data, "No message in response"
    assert "deleted" in context.response_data["message"].lower() or \
           "xóa" in context.response_data["message"].lower(), \
           f"Expected delete success message, got: {context.response_data['message']}"
    print(f"✓ Delete success message: {context.response_data['message']}")


@then('response chứa danh sách documents')
def step_response_has_documents_list(context):
    """Response should contain documents list."""
    assert context.response_data is not None, "Response data is None"
    assert "documents" in context.response_data, "No documents in response"
    assert isinstance(context.response_data["documents"], list)
    print(f"✓ Documents list contains {len(context.response_data['documents'])} items")


@then('total documents là {count:d}')
def step_total_documents_count(context, count):
    """Total documents should be the given count."""
    assert "total" in context.response_data, "No total in response"
    assert context.response_data["total"] == count, \
        f"Expected {count} documents, got {context.response_data['total']}"
    print(f"✓ Total documents: {context.response_data['total']}")


@then('content type là "{content_type}"')
def step_content_type(context, content_type):
    """Content type should be the given value."""
    # Check response headers
    content_type_header = context.response.headers.get("content-type", "")
    assert content_type in content_type_header.lower(), \
        f"Expected content-type {content_type}, got {content_type_header}"
    print(f"✓ Content type: {content_type_header}")

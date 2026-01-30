"""
Step definitions for batch processing feature.
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

@given('hệ thống đã được cấu hình để xử lý tài liệu')
def step_system_configured(context):
    """System is configured for document processing."""
    assert context.test_client is not None
    print("✓ System configured for document processing")


@given('tôi có 3 tài liệu PDF để upload')
def step_have_3_pdfs(context):
    """I have 3 PDF documents to upload."""
    context.test_files = []
    for i in range(3):
        filename = f"batch_doc_{i+1}.pdf"
        content = f"This is PDF document number {i+1} content.".encode()
        context.test_files.append((filename, content, "application/pdf"))
    print(f"✓ Created {len(context.test_files)} test PDF documents")


@given('tôi có 5 tài liệu, trong đó 1 file có định dạng không hỗ trợ')
def step_have_mixed_files(context):
    """I have 5 documents, 1 of which has unsupported format."""
    context.test_files = []
    
    # 4 valid PDF files
    for i in range(4):
        filename = f"valid_doc_{i+1}.pdf"
        content = f"Valid document {i+1}.".encode()
        context.test_files.append((filename, content, "application/pdf"))
    
    # 1 invalid file
    invalid_filename = "invalid_file.exe"
    invalid_content = b"Invalid executable file content."
    context.test_files.append((invalid_filename, invalid_content, "application/octet-stream"))
    
    print(f"✓ Created {len(context.test_files)} files (4 valid, 1 invalid)")


@given('collection "{collection_name}" đã tồn tại')
def step_collection_exists(context, collection_name):
    """A collection with given name already exists."""
    create_collection_if_not_exists(context, collection_name)
    context.test_collection_name = collection_name
    print(f"✓ Collection '{collection_name}' exists")


@given('tôi có 10 tài liệu PDF')
def step_have_10_pdfs(context):
    """I have 10 PDF documents."""
    context.test_files = []
    for i in range(10):
        filename = f"batch_pdf_{i+1}.pdf"
        content = f"This is PDF batch document number {i+1}.".encode()
        context.test_files.append((filename, content, "application/pdf"))
    print(f"✓ Created {len(context.test_files)} test PDF documents")


# When steps

@when('tôi upload từng tài liệu một lần')
def step_upload_each_document(context):
    """Upload each document one by one."""
    context.upload_results = []
    
    for filename, content, content_type in context.test_files:
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"collection": "test_collection"}
        
        response = context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
        
        result = {
            "filename": filename,
            "status_code": response.status_code
        }
        
        try:
            result["data"] = response.json()
        except:
            result["data"] = None
        
        context.upload_results.append(result)
        print(f"→ Uploaded {filename}, status: {response.status_code}")


@when('tôi upload tất cả 10 tài liệu vào collection này')
def step_upload_all_to_collection(context):
    """Upload all 10 documents to this collection."""
    context.upload_results = []
    collection_name = context.test_collection_name
    
    for filename, content, content_type in context.test_files:
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"collection": collection_name}
        
        response = context.test_client.post(
            f"{context.base_url}/documents/upload",
            files=files,
            data=data
        )
        
        result = {
            "filename": filename,
            "status_code": response.status_code
        }
        
        try:
            result["data"] = response.json()
        except:
            result["data"] = None
        
        context.upload_results.append(result)
        print(f"→ Uploaded {filename} to {collection_name}, status: {response.status_code}")


# Then steps

@then('tất cả 3 tài liệu được upload thành công')
def step_all_3_success(context):
    """All 3 documents uploaded successfully."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    assert len(context.upload_results) == 3, f"Expected 3 uploads, got {len(context.upload_results)}"
    
    for result in context.upload_results:
        assert result["status_code"] in [200, 201], \
            f"Document {result['filename']} failed with status {result['status_code']}"
    
    print("✓ All 3 documents uploaded successfully")


@then('mỗi tài liệu trả về status code 201')
def step_each_status_201(context):
    """Each document returns status code 201."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    
    for result in context.upload_results:
        assert result["status_code"] == 201, \
            f"Document {result['filename']} has status {result['status_code']}, expected 201"
    
    print("✓ All documents returned status code 201")


@then('mỗi tài liệu có document_id hợp lệ')
def step_each_has_valid_id(context):
    """Each document has a valid document_id."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    
    for result in context.upload_results:
        data = result.get("data")
        assert data is not None, f"No data for {result['filename']}"
        assert "document_id" in data, f"No document_id for {result['filename']}"
        assert data["document_id"] is not None, f"document_id is None for {result['filename']}"
    
    print("✓ All documents have valid document_ids")


@then('4 tài liệu hợp lệ được upload thành công')
def step_4_valid_success(context):
    """4 valid documents uploaded successfully."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    
    valid_count = 0
    for result in context.upload_results:
        if result["status_code"] in [200, 201]:
            valid_count += 1
    
    assert valid_count == 4, f"Expected 4 successful uploads, got {valid_count}"
    print(f"✓ {valid_count} valid documents uploaded successfully")


@then('1 tài liệu không hợp lệ bị từ chối với status code 400')
def step_1_invalid_rejected(context):
    """1 invalid document is rejected with status code 400."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    
    rejected_count = 0
    for result in context.upload_results:
        if result["status_code"] == 400:
            rejected_count += 1
    
    assert rejected_count == 1, f"Expected 1 rejection, got {rejected_count}"
    print(f"✓ {rejected_count} invalid document rejected with status code 400")


@then('mỗi lỗi có thông báo rõ ràng')
def step_each_error_clear_message(context):
    """Each error has a clear message."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    
    error_count = 0
    for result in context.upload_results:
        if result["status_code"] >= 400:
            error_count += 1
            data = result.get("data")
            assert data is not None, f"No data for error response {result['filename']}"
            
            # Check for error message
            response_text = str(data).lower()
            assert "error" in response_text or "invalid" in response_text or "not supported" in response_text, \
                f"No clear error message for {result['filename']}: {data}"
    
    assert error_count > 0, "Expected at least one error"
    print(f"✓ All {error_count} errors have clear messages")


@then('tất cả 10 tài liệu được upload thành công')
def step_all_10_success(context):
    """All 10 documents uploaded successfully."""
    assert hasattr(context, 'upload_results'), "No upload results found"
    assert len(context.upload_results) == 10, f"Expected 10 uploads, got {len(context.upload_results)}"
    
    success_count = 0
    for result in context.upload_results:
        if result["status_code"] in [200, 201]:
            success_count += 1
    
    assert success_count == 10, f"Expected 10 successful uploads, got {success_count}"
    print("✓ All 10 documents uploaded successfully")


@then('listing documents trong collection trả về 10 tài liệu')
def step_listing_returns_10(context):
    """Listing documents in collection returns 10 documents."""
    response = context.test_client.get(
        f"{context.base_url}/documents/list/{context.test_collection_name}"
    )
    
    assert response.status_code == 200, f"Listing failed with status {response.status_code}"
    
    data = response.json()
    assert "total" in data, "No 'total' in response"
    assert data["total"] == 10, f"Expected 10 documents, got {data['total']}"
    
    print(f"✓ Collection listing shows {data['total']} documents")

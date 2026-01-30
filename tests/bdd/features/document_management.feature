# language: vi
Chức năng: Quản lý tài liệu
  Người dùng muốn tải lên, quản lý và truy xuất tài liệu trong hệ thống RAG
  Để có thể thực hiện truy vấn thông tin từ các tài liệu này

  Scenario: Tải lên tài liệu thành công
    Given hệ thống RAG đã được khởi động và hoạt động bình thường
    And tôi có một tài liệu PDF hợp lệ
    And tôi có một collection tên "research_docs" đã tồn tại
    When tôi gửi POST request tới endpoint "/documents/upload" với file và collection name
    Then status code trả về là 201
    And response chứa document_id hợp lệ
    And response có filename đúng với file đã upload
    And response có collection_name là "research_docs"
    And chunk_count lớn hơn 0

  Scenario: Tải lên tài liệu không hỗ trợ
    Given hệ thống RAG đã được khởi động
    And tôi có một file định dạng không được hỗ trợ là "test.exe"
    When tôi gửi POST request tới endpoint "/documents/upload" với file
    Then status code trả về là 400
    And response chứa thông báo lỗi về định dạng file không hỗ trợ

  Scenario: Xóa tài liệu thành công
    Given một tài liệu đã được tải lên và xử lý xong
    And tài liệu có filename là "test.pdf" trong collection "test_collection"
    When tôi gửi DELETE request tới endpoint "/documents/test_collection/test.pdf"
    Then status code trả về là 200
    And success trong response là True
    And response chứa thông báo xóa thành công

  Scenario: Lấy danh sách tài liệu
    Given có 5 tài liệu đã được tải lên trong collection "research_docs"
    When tôi gửi GET request tới endpoint "/documents/list/research_docs"
    Then status code trả về là 200
    And response chứa danh sách documents
    And total documents là 5

  Scenario: Tải xuống tài liệu
    Given tài liệu "test.pdf" đã tồn tại trong collection "test_collection"
    When tôi gửi GET request tới endpoint "/documents/download/test_collection/test.pdf"
    Then status code trả về là 200
    And content type là "application/pdf"

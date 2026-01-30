# language: vi
Chức năng: Quản lý Collections
  Người dùng muốn tạo, quản lý và xóa collections
  Để tổ chức tài liệu theo các chủ đề khác nhau

  Scenario: Tạo collection mới thành công
    Given hệ thống RAG đang hoạt động
    When tôi gửi POST request tới endpoint "/collections/" với name="test_collection"
    Then status code trả về là 200 hoặc 201
    And response chứa collection có name là "test_collection"
    And collection status là "ready"

  Scenario: Tạo collection trùng tên
    Given collection "existing_collection" đã tồn tại
    When tôi gửi POST request tới endpoint "/collections/" với name="existing_collection"
    Then status code trả về là 409
    Hoặc status code trả về là 500
    And response chứa thông báo về collection đã tồn tại

  Scenario: Liệt kê tất cả collections
    Given có 3 collections đã được tạo: "research", "medical", "legal"
    When tôi gửi GET request tới endpoint "/collections/"
    Then status code trả về là 200
    And response chứa danh sách collections
    And total collections là 3

  Scenario: Lấy thông tin collection
    Given collection "test_collection" đã tồn tại
    When tôi gửi GET request tới endpoint "/collections/test_collection"
    Then status code trả về là 200
    And response chứa collection name là "test_collection"
    Và response có vector_count
    Và response có dimension

  Scenario: Xóa collection thành công
    Given collection "to_delete" đã tồn tại
    When tôi gửi DELETE request tới endpoint "/collections/to_delete"
    Then status code trả về là 200
    And success trong response là True
    And response chứa thông báo xóa thành công

  Scenario: Xóa collection không tồn tại
    Given collection "nonexistent" không tồn tại
    When tôi gửi DELETE request tới endpoint "/collections/nonexistent"
    Then status code trả về là 404
    And response chứa thông báo collection không tồn tại

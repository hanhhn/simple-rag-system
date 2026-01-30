# language: vi
Chức năng: Tìm kiếm và Similarity Search
  Người dùng muốn tìm kiếm tài liệu tương tự
  Để khám phá nội dung liên quan

  Scenario: Similarity search trả về kết quả liên quan
    Given có 100 tài liệu đã được index trong collection
    And embedding model đã được load
    When tôi thực hiện query với nội dung liên quan đến tài liệu trong collection
    Then hệ thống trả về top-k tài liệu có điểm similarity cao nhất
    And mỗi kết quả có score
    Và kết quả được sắp xếp theo score giảm dần

  Scenario: Filter kết quả theo score threshold
    Given similarity search trả về nhiều kết quả
    And score threshold được set là 0.7
    When hệ thống lọc kết quả
    Then chỉ các kết quả có score >= 0.7 được trả về
    Và các kết quả thấp hơn threshold bị loại bỏ

  Scenario: Query với collection không tồn tại
    Given collection "nonexistent" không tồn tại
    When tôi gửi POST request tới endpoint "/query/" với collection này
    Then status code trả về là 404 hoặc 200
    Và retrieved_count là 0

  Scenario: Query trả về không có kết quả
    Given có tài liệu trong collection
    And tôi query về một chủ đề không liên quan
    When tôi gửi POST request tới endpoint "/query/"
    Then status code trả về là 200
    Và retrieved_count là 0
    Hoặc answer thông báo không tìm thấy kết quả

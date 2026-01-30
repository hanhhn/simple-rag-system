# language: vi
Chức năng: Error Handling và Validation
  Hệ thống cần xử lý lỗi và validation một cách chính xác
  Để đảm bảo tính ổn định và UX tốt

  Scenario: Validate query rỗng
    Given hệ thống đang hoạt động
    When tôi gửi POST request tới endpoint "/query/" với query rỗng
    Then status code trả về là 400
    And response chứa thông báo validation error

  Scenario: Validate query quá dài
    Given hệ thống đang hoạt động
    When tôi gửi POST request tới endpoint "/query/" với query dài hơn 1000 ký tự
    Then status code trả về là 400
    And response chứa thông báo về query quá dài

  Scenario: Validate top_k không hợp lệ
    Given hệ thống đang hoạt động
    When tôi gửi POST request tới endpoint "/query/" với top_k=0
    Then status code trả về là 400
    And response chứa thông báo validation error
    When tôi gửi POST request với top_k=10000
    Then status code trả về là 400
    And response chứa thông báo về top_k quá lớn

  Scenario: Validate collection name không hợp lệ
    Given hệ thống đang hoạt động
    When tôi gửi POST request tới endpoint "/collections/" với name rỗng
    Then status code trả về là 400
    And response chứa thông báo validation error
    When tôi gửi POST request với name có ký tự đặc biệt không hợp lệ
    Then status code trả về là 400
    And response chứa thông báo về name không hợp lệ

  Scenario: Validate file size quá lớn
    Given hệ thống có giới hạn file size là 10MB
    And tôi có một file lớn hơn 10MB
    When tôi gửi POST request tới endpoint "/documents/upload" với file này
    Then status code trả về là 400
    And response chứa thông báo về file size quá lớn

  Scenario: Validate file type không hỗ trợ
    Given hệ thống chỉ hỗ trợ PDF, TXT, MD, DOCX
    And tôi có một file .xyz không được hỗ trợ
    When tôi gửi POST request tới endpoint "/documents/upload" với file này
    Then status code trả về là 400
    And response chứa thông báo về file type không được hỗ trợ

  Scenario: Validate dimension không hợp lệ
    Given hệ thống đang hoạt động
    When tôi gửi POST request tới endpoint "/collections/" với dimension=0
    Then status code trả về là 400
    And response chứa thông báo validation error
    When tôi gửi POST request với dimension=10000
    Then status code trả về là 400
    And response chứa thông báo về dimension quá lớn

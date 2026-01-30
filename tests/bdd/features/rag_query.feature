# language: vi
Chức năng: Truy vấn thông tin với RAG
  Người dùng muốn đặt câu hỏi và nhận câu trả lời dựa trên tài liệu
  Để có được thông tin chính xác và có trích dẫn nguồn

  Scenario: Truy vấn thành công với kết quả có sẵn
    Given có tài liệu về "machine learning" đã được index trong collection
    And tôi có collection tên "ml_collection"
    When tôi gửi POST request tới endpoint "/query/" với query "What is supervised learning?"
    Then status code trả về là 200
    And response chứa answer có nghĩa
    And response có retrieved_documents không rỗng
    And mỗi retrieved_document có document_id và score

  Scenario: Truy vấn với collection rỗng
    Given collection "empty_collection" không có tài liệu nào
    When tôi gửi POST request tới endpoint "/query/" với query trỏ tới collection này
    Then status code trả về là 200
    And retrieved_count là 0
    Hoặc answer là "No relevant documents found"

  Scenario: Truy vấn với top_k parameter
    Given có 20 tài liệu trong collection
    When tôi gửi POST request tới endpoint "/query/" với top_k=5
    Then response chỉ trả về tối đa 5 retrieved_documents
    And retrieval_count là 5 hoặc nhỏ hơn

  Scenario: Truy vấn với score threshold
    Given có tài liệu trong collection
    When tôi gửi POST request tới endpoint "/query/" với score_threshold=0.7
    Then tất cả retrieved_documents có score >= 0.7
    Hoặc không có documents nào được trả về nếu không thỏa mãn

  Scenario: Truy vấn với RAG disabled
    Given có tài liệu trong collection
    When tôi gửi POST request tới endpoint "/query/" với use_rag=false
    Then status code trả về là 200
    And answer có thể là None hoặc empty
    Nhưng retrieved_documents vẫn được trả về

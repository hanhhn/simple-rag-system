# language: vi
Chức năng: Xử lý hàng loạt tài liệu
  Người dùng muốn tải lên nhiều tài liệu cùng lúc
  Để tiết kiệm thời gian và tối ưu hóa quy trình

  Scenario: Upload nhiều tài liệu thành công
    Given hệ thống đã được cấu hình để xử lý tài liệu
    And tôi có 3 tài liệu PDF để upload
    When tôi upload từng tài liệu một lần
    Then tất cả 3 tài liệu được upload thành công
    And mỗi tài liệu trả về status code 201
    And mỗi tài liệu có document_id hợp lệ

  Scenario: Upload batch với một file lỗi
    Given tôi có 5 tài liệu, trong đó 1 file có định dạng không hỗ trợ
    When tôi upload từng tài liệu
    Then 4 tài liệu hợp lệ được upload thành công
    And 1 tài liệu không hợp lệ bị từ chối với status code 400
    And mỗi lỗi có thông báo rõ ràng

  Scenario: Upload nhiều tài liệu vào cùng collection
    Given collection "batch_collection" đã tồn tại
    And tôi có 10 tài liệu PDF
    When tôi upload tất cả 10 tài liệu vào collection này
    Then tất cả 10 tài liệu được upload thành công
    And listing documents trong collection trả về 10 tài liệu

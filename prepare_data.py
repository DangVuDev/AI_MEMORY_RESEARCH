"""
Test Data Preparation
Generate corpus and test questions for Memory AI hypothesis validation
"""

import json
import os

def prepare_test_corpus():
    """
    Create a test corpus of FAQ/product documentation (~500 chunks equivalent)
    Based on product documentation scenario from docs
    """
    
    corpus = [
        # Product & Features
        "Chính sách hoàn tiền: Chúng tôi hoàn 100% giá trị sản phẩm trong vòng 30 ngày nếu bạn không hoàn toàn hài lòng. Không cần giải thích gì.",
        "Sản phẩm A có tính năng tích hợp machine learning tự động phân loại dữ liệu với độ chính xác 98%.",
        "Sản phẩm B là đối thủ cạnh tranh có tính năng API tích hợp mạnh mẽ nhưng đắt hơn 40% so với sản phẩm A.",
        "Chúng tôi cung cấp dịch vụ Premium với hỗ trợ 24/7 qua email, live chat, và điện thoại.",
        "Gói Standard giá $29/tháng bao gồm 100GB lưu trữ và hỗ trợ email cơ bản.",
        "Gói Enterprise được tùy chỉnh cho các doanh nghiệp lớn với giá thương lượng và SLA 99.9%.",
        "Ứng dụng mobile hỗ trợ cả iOS (phiên bản 12+) và Android (phiên bản 8+).",
        "Bạn có thể nâng cấp hoặc hạ cấp gói bất kỳ lúc nào mà không có phí hủy hợp đồng.",
        
        # Billing & Payment
        "Phương thức thanh toán được hỗ trợ: Credit Card, Debit Card, PayPal, và Bank Transfer.",
        "Hóa đơn được gửi tự động vào ngày 1 hàng tháng đến email đã đăng ký.",
        "Chúng tôi chấp nhận thanh toán hàng năm với giảm giá 20% so với thanh toán hàng tháng.",
        "Khoá tài khoản sẽ bị vô hiệu hóa nếu quá hạn thanh toán 7 ngày.",
        
        # Technical Support
        "Lỗi 404 thường xuất hiện khi trang không tồn tại hoặc bị xóa khỏi máy chủ.",
        "Khôi phục mật khẩu bằng liên kết được gửi đến địa chỉ email đã đăng ký trong vòng 5 phút.",
        "Đội ngũ kỹ thuật làm việc 24/7 để cải thiện sản phẩm và giải quyết sự cố.",
        "Bạn có thể liên hệ support qua số hotline hoặc live chat để được hỗ trợ ngay lập tức.",
        
        # Company & Team
        "Công ty XYZ được thành lập năm 2020 bởi hai founder Alice và Bob.",
        "Alice là CEO và quản lý phòng Product & Marketing.",
        "Bob là CTO và quản lý phòng Engineering & Infrastructure.",
        "Công ty hiện có 50 nhân viên phân bố ở các phòng: Engineering (20), Product (8), Sales (12), Support (10).",
        
        # Projects & Organization
        "Dự án A (CloudCore): Hiện đại hóa cơ sở hạ tầng cloud, dự kiến hoàn thành Q4 2024.",
        "Dự án B (DataMind): Xây dựng platform ML, được quản lý bởi Bob, khởi động Q2 2024.",
        "Dự án C (Portal): Xây dựng customer portal, quản lý bởi Alice, hiện ở phase 2.",
        "Alice và Bob cùng tham gia vào dự án A từ giai đoạn lập kế hoạch.",
        "Phòng Engineering có 20 người làm việc trên 3 dự án chính: CloudCore, DataMind, Portal.",
        
        # Roadmap
        "Roadmap 2024 bao gồm hoàn thành CloudCore, khởi động DataMind, và nâng cấp Portal.",
        "Roadmap 2025 tập trung vào mở rộng thị trường quốc tế và phát triển các sản phẩm mới.",
        "Trong Q3 2024, chúng tôi sẽ ra mắt phiên bản 2.0 với tính năng AI mới.",
        
        # Account & Login
        "Đăng ký tài khoản bằng email, Google, hoặc Microsoft Account.",
        "Bạn cần xác minh email trong vòng 24 giờ để kích hoạt tài khoản.",
        "Hai yếu tố xác thực (2FA) có thể bật trong cài đặt tài khoản để tăng bảo mật.",
        
        # Data & Privacy
        "Dữ liệu người dùng được mã hóa end-to-end trên máy chủ của chúng tôi.",
        "Chúng tôi không bao giờ chia sẻ dữ liệu cá nhân với bên thứ ba mà không được phép.",
        "GDPR compliance được thực hiện đầy đủ cho người dùng ở EU.",
        "Bạn có thể tải xuống toàn bộ dữ liệu hoặc xóa tài khoản theo yêu cầu.",
        
        # Integration
        "API tích hợp hỗ trợ REST và GraphQL cho tối đa linh hoạt.",
        "Webhooks cho phép bạn nhận thông báo real-time khi có sự thay đổi.",
        "SDK sẵn có cho Python, JavaScript, Java, Go, và Ruby.",
        
        # Performance & Scaling
        "Thời gian phản hồi trung bình dưới 200ms từ khắp thế giới.",
        "Hệ thống được thiết kế để xử lý 10,000 yêu cầu/giây.",
        "Auto-scaling đảm bảo hiệu suất ổn định ngay cả khi lưu lượng cao.",
        
        # Training & Documentation
        "Tài liệu API đầy đủ với ví dụ code cho từng ngôn ngữ.",
        "Video hướng dẫn (tutorials) sẵn có trên YouTube channel chính thức.",
        "Chúng tôi cung cấp training sessions hàng tuần cho Enterprise customers.",
        
        # Competitor Analysis
        "So với sản phẩm B: Chúng tôi rẻ hơn 40%, hỗ trợ tốt hơn, nhưng ít tính năng nâng cao hơn.",
        "So với sản phẩm C: Chúng tôi có interface tốt hơn, tính năng AI tốt hơn, nhưng scale kém hơn.",
        "Lợi thế cạnh tranh chính của chúng tôi là giá cả, hỗ trợ, và machine learning tích hợp.",
    ]
    
    # Duplicate corpus to simulate ~500 chunks equivalent
    corpus = corpus * 4  # ~160 documents
    
    return corpus[:150]  # Use first 150 for consistency


def prepare_test_questions():
    """
    Create test questions with 3 complexity levels
    - Simple (Fact Lookup): 5 câu
    - Multi-hop: 10 câu
    - Synthesis: 5 câu
    """
    
    questions = {
        "simple": [
            ("Chính sách hoàn tiền của công ty là gì?", "Chính sách hoàn tiền"),
            ("Giá gói Standard bao nhiêu tiền mỗi tháng?", "Giá gói Standard"),
            ("Ứng dụng mobile hỗ trợ những nền tảng nào?", "Ứng dụng mobile"),
            ("Công ty XYZ được thành lập năm nào?", "Công ty XYZ thành lập"),
            ("Phương thức thanh toán nào được hỗ trợ?", "Phương thức thanh toán"),
        ],
        "multi_hop": [
            ("Alice và Bob là ai trong công ty, và họ cùng quản lý dự án nào?", "Alice Bob dự án A"),
            ("So sánh sản phẩm A và B về giá cả và tính năng?", "So sánh sản phẩm A B"),
            ("Phòng Engineering có bao nhiêu người và họ làm việc trên những dự án nào?", "Engineering projects"),
            ("Roadmap 2024 bao gồm những gì và ai là người quản lý từng dự án?", "Roadmap 2024 managers"),
            ("Dịch vụ Premium cung cấp những hỗ trợ nào và thông qua kênh nào?", "Premium support channels"),
            ("Khác nhau giữa gói Standard và Enterprise là gì?", "Standard vs Enterprise"),
            ("Bob quản lý phòng nào và dự án nào?", "Bob phòng dự án"),
            ("CloudCore project có mục tiêu gì và dự kiến hoàn thành khi nào?", "CloudCore mục tiêu"),
            ("Làm thế nào để kích hoạt tài khoản sau khi đăng ký?", "Kích hoạt tài khoản"),
            ("Những phương thức thanh toán nào được hỗ trợ và có chiết khấu nào không?", "Thanh toán chiết khấu"),
        ],
        "synthesis": [
            ("Tóm tắt các lỗi phổ biến nhất và cách khắc phục?", "Lỗi phổ biến khắc phục"),
            ("Tổng hợp toàn bộ chiến lược bảo mật của công ty?", "Chiến lược bảo mật"),
            ("Phân tích lợi thế cạnh tranh của công ty so với các đối thủ?", "Lợi thế cạnh tranh"),
            ("Mô tả cấu trúc tổ chức và chức năng của từng phòng ban?", "Cấu trúc tổ chức"),
            ("Lên kế hoạch nâng cấp từ gói Standard lên Enterprise cần những gì?", "Kế hoạch nâng cấp"),
        ]
    }
    
    return questions


def save_corpus_to_file(corpus):
    """Save corpus to JSON file for scenario use"""
    os.makedirs("data", exist_ok=True)
    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "count": len(corpus)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    corpus = prepare_test_corpus()
    questions = prepare_test_questions()
    
    save_corpus_to_file(corpus)
    
    print(f"✓ Test corpus: {len(corpus)} documents")
    print(f"✓ Test questions: {sum(len(q) for q in questions.values())} câu")
    print(f"  - Simple: {len(questions['simple'])}")
    print(f"  - Multi-hop: {len(questions['multi_hop'])}")
    print(f"  - Synthesis: {len(questions['synthesis'])}")
    print(f"\n✓ Saved to: data/corpus.json")

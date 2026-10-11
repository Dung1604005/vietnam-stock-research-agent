"""Manual real-data smoke test for the required conversation memory flow."""

from stock_multi_agent.models.chat import ConversationState
from stock_multi_agent.services.chat_service import ChatService


def main() -> None:
    service = ChatService()
    conversation = ConversationState()
    queries = [
        "Phân tích FPT cho tôi",
        "Cho tôi biểu đồ 6 tháng",
        "RSI thế nào?",
        "So với CMG?",
        "Có cổ phiếu công nghệ nào đáng chú ý?",
        "Tình hình chứng khoán Việt Nam hiện tại?",
    ]
    for query in queries:
        response = service.handle(query, conversation)
        route = " → ".join(response.metadata.get("routing_plan", []))
        print(f"{query} => {response.response_type.value} | {response.tickers} | {route} | {response.data_timestamp}")


if __name__ == "__main__":
    main()

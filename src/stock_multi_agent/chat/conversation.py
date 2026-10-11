from __future__ import annotations

from stock_multi_agent.models.chat import ConversationState


class InMemoryConversationStore:
    def __init__(self, conversations: dict[str, ConversationState] | None = None) -> None:
        self._items = conversations or {}

    def create(self) -> ConversationState:
        conversation = ConversationState()
        self._items[conversation.conversation_id] = conversation
        return conversation

    def get(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._items:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        return self._items[conversation_id]

    def save(self, conversation: ConversationState) -> None:
        self._items[conversation.conversation_id] = conversation

    def delete(self, conversation_id: str) -> None:
        self._items.pop(conversation_id, None)

    def list(self) -> list[ConversationState]:
        return sorted(self._items.values(), key=lambda item: item.updated_at, reverse=True)

    @property
    def items(self) -> dict[str, ConversationState]:
        return self._items

from __future__ import annotations

from stock_multi_agent.models.chat import ConversationState


class InMemoryConversationStore:
    def __init__(self, conversations: dict[str, ConversationState] | None = None) -> None:
        self._items = conversations if conversations is not None else {}

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


class SQLiteConversationStore(InMemoryConversationStore):
    """Local single-user chat history, separate from minimal request telemetry."""

    def __init__(self, conversations=None, path=None):
        import sqlite3
        from pathlib import Path
        from stock_multi_agent.services.run_checkpoints import runtime_database
        super().__init__(conversations)
        self.path = Path(path) if path is not None else runtime_database('conversations.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            for identifier, payload in db.execute('SELECT id, payload FROM conversations'):
                if identifier not in self._items:
                    self._items[identifier] = ConversationState.model_validate_json(payload)

    def create(self):
        conversation = super().create()
        self.save(conversation)
        return conversation

    def save(self, conversation):
        import sqlite3
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT OR REPLACE INTO conversations VALUES (?,?)',
                       (conversation.conversation_id, conversation.model_dump_json()))
        super().save(conversation)

    def delete(self, conversation_id):
        import sqlite3
        with sqlite3.connect(self.path) as db:
            db.execute('DELETE FROM conversations WHERE id=?', (conversation_id,))
        super().delete(conversation_id)

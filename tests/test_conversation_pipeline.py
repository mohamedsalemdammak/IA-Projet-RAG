import json

from src.conversation_memory import ConversationMemoryStore


def test_memory_store_keeps_five_messages(tmp_path):
    store = ConversationMemoryStore(tmp_path / "memory.json")
    store.save(
        "conversation-1",
        {
            "summary": "Résumé",
            "keywords": ["AGP", "AGP", "validation"],
            "messages": [{"role": "user", "content": str(index)} for index in range(7)],
        },
    )
    memory = store.load("conversation-1")
    assert [message["content"] for message in memory["messages"]] == ["2", "3", "4", "5", "6"]
    assert memory["keywords"] == ["AGP", "validation"]


def test_memory_store_rejects_corrupt_json(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{invalid", encoding="utf-8")
    store = ConversationMemoryStore(path)
    try:
        store.load("conversation-1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Une mémoire corrompue doit être signalée.")

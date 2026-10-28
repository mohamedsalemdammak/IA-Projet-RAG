"""Stockage JSON simple de la mémoire par conversation."""

import json
import re
from pathlib import Path

from config import CONVERSATIONS_FILE


class ConversationMemoryStore:
    def __init__(self, path=CONVERSATIONS_FILE):
        self.path = Path(path)

    def load(self, conversation_id):
        self._validate_id(conversation_id)
        if not self.path.exists():
            return {"summary": "", "keywords": [], "messages": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            memory = data.get(conversation_id, {})
            if not isinstance(memory, dict):
                raise ValueError("Mémoire de conversation invalide.")
            messages = memory.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("Historique de conversation invalide.")
            return {
                "summary": str(memory.get("summary", "") or ""),
                "keywords": self._keywords(memory.get("keywords", [])),
                "messages": messages[-5:],
            }
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("La mémoire conversationnelle est corrompue.") from error

    def save(self, conversation_id, memory):
        self._validate_id(conversation_id)
        current = {}
        if self.path.exists():
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("Impossible de lire la mémoire existante.") from error
        if not isinstance(current, dict):
            raise RuntimeError("Le fichier de mémoire doit contenir un objet JSON.")
        current[conversation_id] = {
            "summary": str(memory.get("summary", "") or ""),
            "keywords": self._keywords(memory.get("keywords", [])),
            "messages": list(memory.get("messages", []))[-5:],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _validate_id(conversation_id):
        if not isinstance(conversation_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,100}", conversation_id
        ):
            raise ValueError("conversation_id invalide.")

    @staticmethod
    def _keywords(values):
        if not isinstance(values, list):
            raise ValueError("Les mots-clés doivent être une liste.")
        result = []
        seen = set()
        for value in values:
            item = str(value).strip()
            if item and item.casefold() not in seen:
                seen.add(item.casefold())
                result.append(item)
        return result[-15:]

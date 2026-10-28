import json
import os

from config import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    GROQ_API_KEY_ENV,
    GROQ_MODEL,
)


class GroqLLM:
    def __init__(self, api_key=None, model=None):
        try:
            from groq import Groq
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Le package groq est requis pour la commande ask. "
                "Installez les dépendances avec: pip install -r requirements.txt"
            ) from error
        self.api_key = api_key or os.getenv(GROQ_API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"La variable d'environnement {GROQ_API_KEY_ENV} est requise. "
                "Dans PowerShell, utilisez: "
                "$env:GROQ_API_KEY = 'votre-cle-groq'"
            )
        self.model = model or os.getenv('GROQ_MODEL', GROQ_MODEL)
        self.client = Groq(api_key=self.api_key)

    def complete(
        self,
        system_prompt,
        user_prompt,
        temperature=ANSWER_TEMPERATURE,
        max_tokens=ANSWER_MAX_TOKENS,
    ):
        return self.complete_messages(
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_messages(
        self,
        messages,
        temperature=ANSWER_TEMPERATURE,
        max_tokens=ANSWER_MAX_TOKENS,
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError('Groq a retourné une réponse vide.')
        return content.strip()

    def complete_json(self, system_prompt, user_prompt, max_tokens=300):
        raw = self.complete(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            lines = cleaned.splitlines()
            cleaned = '\n'.join(lines[1:-1]).strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise RuntimeError('La réponse Groq n’est pas un JSON valide.') from error
        if not isinstance(result, dict):
            raise RuntimeError('La réponse Groq doit être un objet JSON.')
        return result

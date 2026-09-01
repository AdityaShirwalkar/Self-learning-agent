"""Memory-enabled chat agent with one Groq request per user message."""

import json
import os
import re
import threading

from groq import Groq
from mem0 import Memory

from config import CHAT_MODEL, MEM0_CONFIG

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CONTEXT_MEMORIES = 30


class SelfLearningAgent:
    _shared_memory: Memory | None = None
    _memory_lock = threading.Lock()

    def __init__(self, user_id: str = "default_user"):
        if not USER_ID_PATTERN.fullmatch(user_id):
            raise ValueError("User ID must be 1-64 letters, numbers, hyphens, or underscores.")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or your host's secret settings.")

        self.user_id = user_id
        self.memory = self._get_shared_memory()
        self.groq_client = Groq(api_key=api_key)
        self.last_memory_error: str | None = None

    @classmethod
    def _get_shared_memory(cls) -> Memory:
        """Load the local embedding model once per server process."""
        with cls._memory_lock:
            if cls._shared_memory is None:
                cls._shared_memory = Memory.from_config(MEM0_CONFIG)
            return cls._shared_memory

    def _get_context(self) -> str:
        """Read saved facts directly instead of relying on semantic ranking."""
        try:
            results = self.memory.get_all(filters={"user_id": self.user_id})
        except Exception as error:
            self.last_memory_error = f"Could not read memories: {error}"
            return "No stored facts are available."

        memories = results.get("results", []) if isinstance(results, dict) else results
        if not memories:
            return "No stored facts are available yet."
        selected = memories[-MAX_CONTEXT_MEMORIES:]
        return "\n".join(f"- {item['memory']}" for item in selected)

    def _save_facts(self, facts: list[str]) -> None:
        """Persist facts without mem0 inference or an additional LLM request."""
        clean_facts = []
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                clean_facts.append(fact.strip()[:500])
        if not clean_facts:
            return

        try:
            self.memory.add(
                [{"role": "user", "content": fact} for fact in clean_facts],
                user_id=self.user_id,
                infer=False,
            )
        except Exception as error:
            self.last_memory_error = f"Could not save memories: {error}"

    def chat(self, user_message: str) -> str:
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("Message cannot be empty.")

        self.last_memory_error = None
        context = self._get_context()
        system_prompt = f"""You are a helpful personal assistant.

Known facts about this user:
{context}

Return a JSON object with exactly two keys:
- "reply": your helpful answer to the user.
- "memories": an array of 0-3 short, durable facts the user explicitly shared
  in this message. Store preferences, personal details, goals, and important
  constraints. Do not store questions, temporary chatter, assistant statements,
  or inferred facts. If the user asks about a known fact above, answer directly.
"""

        response = self.groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw_content)
            reply = str(payload["reply"]).strip()
            facts = payload.get("memories", [])
            if not isinstance(facts, list):
                facts = []
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"Groq returned an invalid structured response: {error}") from error

        self._save_facts(facts)
        return reply or "I could not generate a response."

    def get_all_memories(self) -> list:
        results = self.memory.get_all(filters={"user_id": self.user_id})
        return results.get("results", []) if isinstance(results, dict) else results

    def forget_everything(self) -> None:
        self.memory.delete_all(user_id=self.user_id)

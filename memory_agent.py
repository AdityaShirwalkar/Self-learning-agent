"""
SelfLearningAgent
------------------
A chat agent that remembers facts about each user across sessions using mem0,
and gets smarter/more personalized over time as it accumulates memories.

Flow on every turn:
  1. Search mem0 for memories relevant to the new user message (semantic
     search only — no LLM call, so it's free of rate-limit concerns).
  2. Inject those memories into the system prompt as context.
  3. Ask Groq (free LLM) for a reply.
  4. Store the exchange in mem0 with infer=False — i.e. store it directly
     (embed + save) instead of running mem0's own LLM-based fact-extraction
     step first.

Why infer=False: as of Groq's current free tier, every lightweight chat
model (gpt-oss-120b, gpt-oss-20b, qwen3.6-27b, qwen3.8-27b) is capped at
8,000 tokens per minute. mem0's default fact-extraction call uses a fairly
large built-in prompt that alone runs ~8-9k tokens, so it can exceed that
cap on literally any message, even a one-word one. Storing directly with
infer=False skips that extra LLM call entirely (memory add becomes just an
embedding + save operation), so it works reliably within the free tier.
The trade-off: memories are the raw exchange rather than a distilled,
deduplicated fact — still fully searchable and persists across sessions,
which is what "self-learning" here relies on.
"""

import os
import re
import threading
from groq import Groq
from mem0 import Memory

from config import MEM0_CONFIG, CHAT_MODEL

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FAVORITE_FOOD_PATTERN = re.compile(
    r"\bmy favou?rite food (?:is|are) (?P<food>[A-Za-z][A-Za-z -]{0,60})",
    re.IGNORECASE,
)
LOVE_PATTERN = re.compile(r"\bI love (?P<thing>[A-Za-z][A-Za-z-]{0,40})\b", re.IGNORECASE)
FAVORITE_INTENT_PATTERN = re.compile(
    r"\bI love (?P<food>[A-Za-z][A-Za-z-]{0,40})\s+so\s+(?:that'?s|thats)\s+it\b",
    re.IGNORECASE,
)


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

    @classmethod
    def _get_shared_memory(cls) -> Memory:
        """Load the embedding model once per Python process, not per rerun."""
        with cls._memory_lock:
            if cls._shared_memory is None:
                cls._shared_memory = Memory.from_config(MEM0_CONFIG)
            return cls._shared_memory

    def _retrieve_context(self, query: str, limit: int = 5) -> str:
        """Search mem0 for memories relevant to the current query.

        Note: current mem0ai versions require entity IDs (user_id, agent_id,
        run_id) to be passed inside `filters={}` for search()/get_all() —
        a top-level user_id= kwarg raises ValueError. add() is the one
        exception and still accepts user_id directly.
        """
        try:
            results = self.memory.search(
                query=query, filters={"user_id": self.user_id}, limit=limit
            )
        except Exception as e:
            print(f"[warning] memory search failed, continuing without context: {e}")
            return "No prior memories available for this turn."

        memories = results.get("results", []) if isinstance(results, dict) else results

        if not memories:
            return "No prior memories about this user yet."

        lines = [f"- {m['memory']}" for m in memories]
        return "\n".join(lines)

    @staticmethod
    def _extract_user_facts(user_message: str) -> list[str]:
        """Create concise, searchable memories without a second LLM request."""
        facts = []
        favorite_food = FAVORITE_FOOD_PATTERN.search(user_message)
        if favorite_food:
            food = favorite_food.group("food").strip(" .,!?")
            facts.append(f"The user's favorite food is {food}.")

        favorite_intent = FAVORITE_INTENT_PATTERN.search(user_message)
        if favorite_intent:
            facts.append(f"The user's favorite food is {favorite_intent.group('food').strip()}.")

        for loved_thing in LOVE_PATTERN.findall(user_message):
            facts.append(f"The user loves {loved_thing.strip()}.")

        return facts or [f"The user said: {user_message}"]

    def _store_exchange(self, user_message: str) -> None:
        """Persist user facts only; never store long assistant responses."""
        memories = [
            {"role": "user", "content": fact}
            for fact in self._extract_user_facts(user_message)
        ]
        try:
            self.memory.add(memories, user_id=self.user_id, infer=False)
        except Exception as e:
            print(f"[warning] could not save this exchange to memory: {e}")

    def chat(self, user_message: str) -> str:
        context = self._retrieve_context(user_message)

        system_prompt = (
            "You are a helpful personal assistant. The context below contains known "
            "facts the user explicitly shared. Use those facts naturally. If the user "
            "asks a direct question answered by a fact in the context, answer it "
            "directly; do not say you do not know. Never reveal private information "
            "unless the user asks for it.\n\n"
            f"Known user facts:\n{context}"
        )

        response = self.groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=600,
        )

        reply = response.choices[0].message.content

        # Learn from this exchange for next time
        self._store_exchange(user_message)

        return reply

    def get_all_memories(self) -> list:
        results = self.memory.get_all(filters={"user_id": self.user_id})
        return results.get("results", []) if isinstance(results, dict) else results

    def forget_everything(self) -> None:
        self.memory.delete_all(user_id=self.user_id)

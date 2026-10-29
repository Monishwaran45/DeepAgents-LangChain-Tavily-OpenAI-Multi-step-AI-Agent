import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from src.config import AgentConfig

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """Represents a single exchange between user and the research agent."""
    user_query: str
    agent_response: str
    search_queries_used: List[str] = Field(default_factory=list)
    citations_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryManager:
    """
    Advanced Memory Manager featuring:
    1. Sliding window for verbatim recent turns.
    2. Progressive running summarization for older turns.
    3. JSON file-based session persistence.
    4. Key research insights repository.
    """

    def __init__(
        self,
        session_id: str = "default_session",
        max_recent_turns: int = AgentConfig.MAX_HISTORY_TURNS,
        persistence_dir: str = AgentConfig.PERSISTENCE_DIR,
    ):
        self.session_id = session_id
        self.max_recent_turns = max_recent_turns
        self.persistence_dir = persistence_dir
        self.history: List[ConversationTurn] = []
        self.running_summary: str = ""
        self.key_facts: List[str] = []

        # Ensure persistence directory exists
        os.makedirs(self.persistence_dir, exist_ok=True)
        self.load_session()

    def add_turn(
        self,
        user_query: str,
        agent_response: str,
        search_queries_used: Optional[List[str]] = None,
        citations_count: int = 0,
        summarizer_llm: Optional[Any] = None,
    ):
        """Add a conversation turn, trigger summarization if necessary, and persist."""
        turn = ConversationTurn(
            user_query=user_query,
            agent_response=agent_response,
            search_queries_used=search_queries_used or [],
            citations_count=citations_count,
        )
        self.history.append(turn)

        # Check if history exceeds max_recent_turns and needs summarization
        if len(self.history) > self.max_recent_turns and summarizer_llm:
            self._compact_memory(summarizer_llm)

        self.save_session()

    def _compact_memory(self, summarizer_llm: Any):
        """Compress older turns into a cumulative running summary."""
        turns_to_summarize = self.history[:-self.max_recent_turns]
        self.history = self.history[-self.max_recent_turns:]

        transcript_to_add = "\n".join(
            [f"User: {t.user_query}\nAgent: {t.agent_response[:300]}..." for t in turns_to_summarize]
        )

        prompt = (
            "You are the memory manager for an AI research agent. "
            "Update the running summary of previous conversation turns to preserve critical facts, "
            "topics explored, user requirements, and established conclusions.\n\n"
            f"Previous Running Summary:\n{self.running_summary or 'None'}\n\n"
            f"New Completed Turns:\n{transcript_to_add}\n\n"
            "Provide an updated, concise summary in 3-5 bullet points without fluff."
        )

        try:
            response = summarizer_llm.invoke(prompt)
            summary_content = getattr(response, "content", str(response)).strip()
            self.running_summary = summary_content
            logger.info("Memory compacted successfully.")
        except Exception as e:
            logger.warning(f"Memory compaction failed: {e}")

    def get_memory_context(self) -> str:
        """Format the memory state (summary + recent turns) for prompt injection."""
        sections = []

        if self.running_summary:
            sections.append(f"### Previous Research Summary:\n{self.running_summary}")

        if self.history:
            recent_lines = []
            for t in self.history[-self.max_recent_turns:]:
                recent_lines.append(f"User: {t.user_query}")
                recent_lines.append(f"Assistant: {t.agent_response[:400]}... (summary)")
            sections.append("### Recent Conversation History:\n" + "\n".join(recent_lines))

        return "\n\n".join(sections) if sections else "No prior conversation history."

    def get_session_file_path(self) -> str:
        """Derive safe file path preventing directory traversal attacks."""
        safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in self.session_id]).strip("._")
        if not safe_name:
            safe_name = "default_session"

        abs_dir = os.path.abspath(self.persistence_dir)
        target_path = os.path.abspath(os.path.join(abs_dir, f"{safe_name}.json"))

        # Verify target_path stays strictly within persistence_dir
        if not target_path.startswith(abs_dir):
            raise ValueError(f"Security error: Invalid session path traversal detected: {self.session_id}")

        return target_path

    def save_session(self):
        """Persist memory state to disk."""
        data = {
            "session_id": self.session_id,
            "running_summary": self.running_summary,
            "key_facts": self.key_facts,
            "history": [t.model_dump() for t in self.history],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.get_session_file_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

    def load_session(self):
        """Load memory state from disk if it exists."""
        file_path = self.get_session_file_path()
        if not os.path.exists(file_path):
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.running_summary = data.get("running_summary", "")
            self.key_facts = data.get("key_facts", [])
            self.history = [ConversationTurn(**t) for t in data.get("history", [])]
            logger.info(f"Loaded session '{self.session_id}' with {len(self.history)} previous turns.")
        except Exception as e:
            logger.warning(f"Could not load previous session: {e}")

    def clear(self):
        """Reset conversation memory."""
        self.history = []
        self.running_summary = ""
        self.key_facts = []
        self.save_session()

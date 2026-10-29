import json
import logging
import re
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import AgentConfig
from src.tools.search import TavilySearchTool, SearchResultItem
from src.context.context_manager import ContextManager, ProcessedContext
from src.memory.memory_manager import MemoryManager
from src.agent.hallucination_guard import HallucinationGuard, GroundingAuditResult
from src.agent.prompts import (
    OMNI_RESEARCH_SYSTEM_PROMPT,
    OMNI_PLANNER_PROMPT,
    FOLLOWUP_QUESTIONS_PROMPT,
)

logger = logging.getLogger(__name__)


class SourceCard(BaseModel):
    """Source card metadata badge."""
    index: int
    title: str
    url: str
    domain: str
    snippet: str


class OmniResearchResponse(BaseModel):
    """Output package from the Omni Deep Research Engine."""
    query: str
    content: str
    sources: List[SourceCard] = Field(default_factory=list)
    related_questions: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    focus_mode: str = "web"
    grounding_score: float = 1.0
    tokens_used_estimate: int = 0
    session_id: str = "default_session"


class OmniResearchAgent:
    """
    State-of-the-art Deep Research Engine.
    Features:
    1. Focus Modes: web, academic, finance, coding, deep_research.
    2. Multi-Hop Recursive Search (Deep Research mode).
    3. Source Cards / Favicon Badges with clean numeric citations [1], [2].
    4. Proactive Follow-up / Related Questions discovery.
    5. Factual Grounding & Hallucination Guard.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        session_id: str = "default_session",
        focus_mode: str = "web",
        enable_verification: bool = True,
        max_search_results: int = AgentConfig.MAX_SEARCH_RESULTS,
    ):
        self.groq_api_key = groq_api_key or AgentConfig.GROQ_API_KEY
        self.tavily_api_key = tavily_api_key or AgentConfig.TAVILY_API_KEY
        self.model_name = model_name or AgentConfig.DEFAULT_GROQ_MODEL
        self.session_id = session_id
        self.focus_mode = focus_mode.lower()
        self.enable_verification = enable_verification
        self.max_search_results = max_search_results

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key or "missing_key",
            model_name=self.model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

        self.search_tool = TavilySearchTool(api_key=self.tavily_api_key)
        self.context_mgr = ContextManager()
        self.memory_mgr = MemoryManager(session_id=self.session_id)

    def _extract_domain(self, url: str) -> str:
        """Extract clean root domain for badge (e.g. reuters.com)."""
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc.replace("www.", "")
        except Exception:
            return "web"

    def _generate_search_plan(self, user_query: str) -> List[str]:
        """Generate focus-mode-adapted search queries."""
        memory_ctx = self.memory_mgr.get_memory_context()
        planner_input = (
            f"User Query: {user_query}\n"
            f"Selected Focus Mode: {self.focus_mode}\n"
            f"Conversation Context:\n{memory_ctx}\n\n"
            "Task: Generate 3 to 4 optimized search queries as a JSON array of strings."
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=OMNI_PLANNER_PROMPT),
                HumanMessage(content=planner_input),
            ])
            match = re.search(r"\[.*\]", resp.content, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list) and len(queries) > 0:
                    return [str(q).strip() for q in queries if str(q).strip()]
        except Exception as e:
            logger.warning(f"Error planning queries: {e}")

        # Fallback query
        return [user_query]

    def _execute_search(self, queries: List[str]) -> List[SearchResultItem]:
        """Execute searches with focus mode tuning."""
        topic = "finance" if self.focus_mode == "finance" else "general"
        depth = "advanced" if self.focus_mode in ("deep_research", "academic") else "basic"

        adapted_queries = []
        for q in queries:
            if self.focus_mode == "academic" and "paper" not in q.lower() and "arxiv" not in q.lower():
                adapted_queries.append(f"{q} research paper study")
            elif self.focus_mode == "coding" and "github" not in q.lower() and "doc" not in q.lower():
                adapted_queries.append(f"{q} documentation github")
            else:
                adapted_queries.append(q)

        all_results = []
        for q in adapted_queries:
            sub = self.search_tool.search(
                query=q,
                max_results=self.max_search_results,
                search_depth=depth,
                topic=topic,
            )
            all_results.extend(sub)

        return all_results

    def _recursive_deep_dive(self, user_query: str, initial_results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Multi-hop search for Deep Research mode: identify gaps and run targeted secondary search."""
        if self.focus_mode != "deep_research" or not initial_results:
            return initial_results

        snippets_sample = "\n".join([f"- {r.title}: {r.content[:200]}" for r in initial_results[:6]])
        gap_prompt = (
            f"User Query: {user_query}\n\n"
            f"Preliminary Search Findings:\n{snippets_sample}\n\n"
            "Task: Identify what critical detail, benchmark, or counter-argument is missing from these preliminary findings. "
            "Output exactly 2 targeted follow-up search queries as a JSON array of strings to fill these gaps: [\"query 1\", \"query 2\"]"
        )

        try:
            resp = self.llm.invoke([HumanMessage(content=gap_prompt)])
            match = re.search(r"\[.*\]", resp.content, re.DOTALL)
            if match:
                gap_queries = json.loads(match.group(0))
                if isinstance(gap_queries, list) and len(gap_queries) > 0:
                    secondary_results = self.search_tool.multi_query_search(gap_queries, max_results_per_query=3)
                    initial_results.extend(secondary_results)
        except Exception as e:
            logger.info(f"Recursive hop completed: {e}")

        return initial_results

    def _generate_related_questions(self, user_query: str, synthesis: str) -> List[str]:
        """Generate 3 intelligent follow-up / related questions."""
        prompt = (
            f"Original Query: {user_query}\n\n"
            f"Research Findings (Summary):\n{synthesis[:1500]}\n\n"
            "Generate exactly 3 progressive, curious follow-up questions as a JSON array of strings."
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=FOLLOWUP_QUESTIONS_PROMPT),
                HumanMessage(content=prompt),
            ])
            match = re.search(r"\[.*\]", resp.content, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                if isinstance(questions, list) and len(questions) == 3:
                    return [str(q).strip() for q in questions]
        except Exception as e:
            logger.warning(f"Error generating follow-ups: {e}")

        return [
            f"What are the primary technical bottlenecks of {user_query}?",
            f"How does this compare to competing alternatives in 2025?",
            "What are the expected roadmap milestones over the next 12 months?",
        ]

    def research(
        self,
        query: str,
        focus_mode: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> OmniResearchResponse:
        """Execute the full deep research pipeline."""
        if focus_mode:
            self.focus_mode = focus_mode.lower()

        def report(stage: str, data: Dict[str, Any]):
            if progress_callback:
                progress_callback(stage, data)

        clean_query = "".join(c for c in query if c.isprintable() or c in "\n\t").strip()
        if not clean_query:
            raise ValueError("Research query cannot be empty.")
        query = clean_query[:2000]

        # 1. Planning
        report("planning", {"message": f"Analyzing query in '{self.focus_mode}' focus mode..."})
        search_queries = self._generate_search_plan(query)
        report("queries_ready", {"queries": search_queries})

        # 2. Search & Multi-hop
        report("searching", {"message": f"Searching across {len(search_queries)} queries via Tavily..."})
        raw_results = self._execute_search(search_queries)

        if self.focus_mode == "deep_research":
            report("deep_dive", {"message": "Conducting multi-hop recursive gap exploration..."})
            raw_results = self._recursive_deep_dive(query, raw_results)

        # 3. Context Processing & Source Badges
        report("context_processing", {"message": "Deduplicating and indexing sources..."})
        processed_ctx = self.context_mgr.build_context(raw_results, query)

        # Format source cards
        source_cards: List[SourceCard] = []
        for idx, s in enumerate(processed_ctx.sources, start=1):
            source_cards.append(
                SourceCard(
                    index=idx,
                    title=s.title,
                    url=s.url,
                    domain=self._extract_domain(s.url),
                    snippet=s.content[:200],
                )
            )

        report("context_ready", {
            "sources_count": len(source_cards),
            "estimated_tokens": processed_ctx.total_tokens_estimated,
        })

        # 4. Grounded Synthesis with [1], [2] Citations
        report("synthesizing", {"message": f"Synthesizing report using {self.model_name}..."})
        memory_ctx = self.memory_mgr.get_memory_context()

        synthesis_prompt = (
            f"User Research Query:\n{query}\n\n"
            f"Conversation History:\n{memory_ctx}\n\n"
            f"Focus Mode: {self.focus_mode.upper()}\n\n"
            f"Verified Reference Sources ({len(source_cards)} sources indexed):\n"
            f"{processed_ctx.formatted_context}\n\n"
            "Instructions:\n"
            "1. Deliver an immediate, high-impact Executive Summary answering the user's question.\n"
            "2. Provide an exhaustive Deep Dive structured with headers, comparisons, and key metrics.\n"
            "3. Cite every single claim, number, and date using numeric bracket citations: [1], [2], or [1, 3] matching the source numbers.\n"
            "4. Transparently document any data gaps or missing information."
        )

        response = self.llm.invoke([
            SystemMessage(content=OMNI_RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        draft_content = response.content.strip()

        # 5. Hallucination & Citation Verification
        report("verifying", {"message": "Verifying numeric citations and factual grounding..."})
        audit = HallucinationGuard.audit_and_sanitize(draft_content, processed_ctx.sources)
        final_content = audit.sanitized_content

        # 6. Related / Follow-up Questions
        report("discovering", {"message": "Generating related follow-up inquiries..."})
        related_questions = self._generate_related_questions(query, final_content)

        # 7. Memory Update
        self.memory_mgr.add_turn(
            user_query=query,
            agent_response=final_content,
            search_queries_used=search_queries,
            citations_count=audit.valid_citations,
            summarizer_llm=self.llm,
        )

        report("completed", {"message": "Research complete."})

        return OmniResearchResponse(
            query=query,
            content=final_content,
            sources=source_cards,
            related_questions=related_questions,
            search_queries=search_queries,
            focus_mode=self.focus_mode,
            grounding_score=audit.grounding_score,
            tokens_used_estimate=processed_ctx.total_tokens_estimated,
            session_id=self.session_id,
        )

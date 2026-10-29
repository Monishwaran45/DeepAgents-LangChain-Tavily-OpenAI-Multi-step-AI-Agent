import json
import logging
import re
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import AgentConfig
from src.tools.search import TavilySearchTool, SearchResultItem
from src.context.context_manager import ContextManager, ProcessedContext
from src.memory.memory_manager import MemoryManager
from src.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    CRITIC_VERIFICATION_PROMPT,
)
from src.agent.hallucination_guard import HallucinationGuard, GroundingAuditResult

logger = logging.getLogger(__name__)


class ResearchResponse(BaseModel):
    """Encapsulates the agent's verified output and metadata."""
    query: str
    content: str
    search_queries: List[str] = Field(default_factory=list)
    sources: List[SearchResultItem] = Field(default_factory=list)
    tokens_used_estimate: int = 0
    verification_passed: bool = True
    grounding_score: float = 1.0
    total_citations: int = 0
    session_id: str = "default_session"


class DeepResearchAgent:
    """
    Advanced Multi-step Research Agent leveraging Groq LPU inference,
    Tavily search, intelligent memory management, context pruning,
    and factual accuracy verification.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        session_id: str = "default_session",
        enable_verification: bool = True,
        max_search_results: int = AgentConfig.MAX_SEARCH_RESULTS,
        search_depth: str = AgentConfig.SEARCH_DEPTH,
    ):
        self.groq_api_key = groq_api_key or AgentConfig.GROQ_API_KEY
        self.tavily_api_key = tavily_api_key or AgentConfig.TAVILY_API_KEY
        self.model_name = model_name or AgentConfig.DEFAULT_GROQ_MODEL
        self.session_id = session_id
        self.enable_verification = enable_verification
        self.max_search_results = max_search_results
        self.search_depth = search_depth

        # Initialize LLM with ChatGroq
        self.llm = self._init_llm(self.model_name)

        # Initialize Modules
        self.search_tool = TavilySearchTool(api_key=self.tavily_api_key)
        self.context_mgr = ContextManager()
        self.memory_mgr = MemoryManager(session_id=self.session_id)

    def _init_llm(self, model_name: str) -> ChatGroq:
        """Initialize ChatGroq with API key and parameters."""
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY is not set or placeholder. LLM calls will fail until key is configured.")

        return ChatGroq(
            groq_api_key=self.groq_api_key or "missing_key",
            model_name=model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

    def _generate_search_plan(self, user_query: str) -> List[str]:
        """Step 1: Plan and formulate 2-4 precision search queries."""
        memory_ctx = self.memory_mgr.get_memory_context()
        user_message_content = (
            f"User Research Query: {user_query}\n\n"
            f"Session Context:\n{memory_ctx}\n\n"
            "Generate 2-4 optimal, distinct search queries as a JSON array of strings."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=user_message_content),
            ])
            text = response.content.strip()

            # Extract JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list) and len(queries) > 0:
                    return [str(q).strip() for q in queries if str(q).strip()]

        except Exception as e:
            logger.warning(f"Error in search query planning: {e}. Falling back to default query.")

        return [user_query]

    def _synthesize_research(
        self,
        user_query: str,
        processed_context: ProcessedContext,
    ) -> str:
        """Step 3: Synthesize comprehensive, citation-grounded research report."""
        memory_ctx = self.memory_mgr.get_memory_context()

        synthesis_prompt = (
            f"User Research Query:\n{user_query}\n\n"
            f"Prior Conversation Memory:\n{memory_ctx}\n\n"
            f"Verified Web Sources ({processed_context.num_sources_used} sources used):\n"
            f"{processed_context.formatted_context}\n\n"
            "Instructions:\n"
            "1. Synthesize a comprehensive, authoritative research report addressing the user query.\n"
            "2. Anchor every key fact, statistic, and conclusion with inline markdown citations like [Title](URL).\n"
            "3. Provide an Executive Summary, Detailed Analysis, Key Findings, and a full Source References section at the end.\n"
            "4. If certain questions cannot be definitively answered from the sources, clearly state the data gap."
        )

        response = self.llm.invoke([
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        return response.content.strip()

    def _verify_and_critique(
        self,
        draft: str,
        processed_context: ProcessedContext,
    ) -> str:
        """Step 4: Fact-checker critic loop to enforce accuracy and valid citations."""
        verification_input = (
            f"Draft Research Report:\n{draft}\n\n"
            f"Original Reference Sources:\n{processed_context.formatted_context}\n\n"
            "Task:\n"
            "Examine every factual claim and citation. If all claims are grounded and cited correctly, "
            "return the report. If there are hallucinations, ungrounded claims, or broken citation links, "
            "correct them directly."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=CRITIC_VERIFICATION_PROMPT),
                HumanMessage(content=verification_input),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Verification step encountered error: {e}. Proceeding with initial draft.")
            return draft

    def research(
        self,
        query: str,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> ResearchResponse:
        """
        Execute the full research pipeline:
        1. Formulate search queries.
        2. Execute Tavily search.
        3. Context deduplication & budgeting.
        4. Grounded synthesis.
        5. Verification and fact-checking.
        6. Memory update.
        """
        def report(stage: str, data: Dict[str, Any]):
            if progress_callback:
                progress_callback(stage, data)

        # Input sanitization and length bounds
        cleaned_query = "".join(c for c in query if c.isprintable() or c in "\n\t").strip()
        if len(cleaned_query) > 2000:
            logger.warning("Query exceeds 2000 characters. Truncating for safety.")
            cleaned_query = cleaned_query[:2000]

        if not cleaned_query:
            raise ValueError("Input query cannot be empty.")

        query = cleaned_query

        # 1. Planning
        report("planning", {"message": "Formulating precision search queries with Groq..."})
        search_queries = self._generate_search_plan(query)
        report("queries_ready", {"queries": search_queries})

        # 2. Search Execution
        report("searching", {"message": f"Executing Tavily web search for {len(search_queries)} queries..."})
        raw_results = self.search_tool.multi_query_search(
            queries=search_queries,
            max_results_per_query=self.max_search_results,
            search_depth=self.search_depth,
        )
        report("search_completed", {"raw_results_count": len(raw_results)})

        # 3. Context Management (Deduplication, Reranking, Token Budgeting)
        report("context_processing", {"message": "Deduplicating and budgeting context..."})
        processed_ctx = self.context_mgr.build_context(raw_results, query)
        report("context_ready", {
            "sources_used": processed_ctx.num_sources_used,
            "sources_dropped": processed_ctx.num_sources_dropped,
            "estimated_tokens": processed_ctx.total_tokens_estimated,
        })

        # 4. Synthesis
        report("synthesizing", {"message": f"Synthesizing report using Groq ({self.model_name})..."})
        draft_report = self._synthesize_research(query, processed_ctx)

        # 5. Verification & Accuracy Check
        if self.enable_verification:
            report("verifying", {"message": "Validating factual grounding and citations..."})
            critiqued_report = self._verify_and_critique(draft_report, processed_ctx)
        else:
            critiqued_report = draft_report

        # 6. Automated Hallucination & Citation Verification Guard
        audit = HallucinationGuard.audit_and_sanitize(critiqued_report, processed_ctx.sources)
        final_report = audit.sanitized_content
        report("grounding_verified", {
            "grounding_score": f"{int(audit.grounding_score * 100)}%",
            "valid_citations": audit.valid_citations,
            "hallucinated_citations": len(audit.hallucinated_citations),
        })

        # 7. Memory Management Update
        self.memory_mgr.add_turn(
            user_query=query,
            agent_response=final_report,
            search_queries_used=search_queries,
            citations_count=audit.valid_citations,
            summarizer_llm=self.llm,
        )

        response = ResearchResponse(
            query=query,
            content=final_report,
            search_queries=search_queries,
            sources=processed_ctx.sources,
            tokens_used_estimate=processed_ctx.total_tokens_estimated,
            verification_passed=True,
            grounding_score=audit.grounding_score,
            total_citations=audit.total_citations,
            session_id=self.session_id,
        )

        report("completed", {"message": f"Research completed. Grounding Score: {int(audit.grounding_score * 100)}%"})
        return response

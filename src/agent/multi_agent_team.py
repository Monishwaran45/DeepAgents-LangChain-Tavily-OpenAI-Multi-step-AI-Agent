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
from src.agent.hallucination_guard import HallucinationGuard, GroundingAuditResult
from src.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    CRITIC_VERIFICATION_PROMPT,
    USER_OPTIMIZER_PROMPT,
)

logger = logging.getLogger(__name__)


class MultiAgentResearchResponse(BaseModel):
    """Output package from the collaborative multi-agent research team."""
    query: str
    content: str
    search_queries: List[str] = Field(default_factory=list)
    sources: List[SearchResultItem] = Field(default_factory=list)
    evidence_dossier: str = ""
    grounding_score: float = 1.0
    total_citations: int = 0
    tokens_used_estimate: int = 0
    session_id: str = "default_session"
    agent_team: List[str] = Field(default_factory=lambda: [
        "PlannerAgent",
        "WebScoutAgent",
        "DataAnalystAgent",
        "SynthesisWriterAgent",
        "CriticVerifierAgent",
        "HallucinationGuard"
    ])


class MultiAgentResearchTeam:
    """
    Orchestrates a collaborative team of specialized open-source AI agents:
    - PlannerAgent: Strategic scope deconstruction & query planning
    - WebScoutAgent: Precision Tavily search, deduplication & budgeting
    - DataAnalystAgent: Fact extraction, cross-source triage & contradiction audit
    - SynthesisWriterAgent: Authoritative, deeply cited synthesis
    - CriticVerifierAgent: Adversarial quality assurance & factual verification
    - HallucinationGuard: Programmatic link validation & grounding score computation
    """

    def __init__(
        self,
        primary_model: Optional[str] = None,
        analyst_model: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        session_id: str = "default_session",
        enable_verification: bool = True,
        max_search_results: int = AgentConfig.MAX_SEARCH_RESULTS,
        search_depth: str = AgentConfig.SEARCH_DEPTH,
    ):
        self.groq_api_key = groq_api_key or AgentConfig.GROQ_API_KEY
        self.tavily_api_key = tavily_api_key or AgentConfig.TAVILY_API_KEY
        self.primary_model_name = primary_model or AgentConfig.DEFAULT_GROQ_MODEL
        self.analyst_model_name = analyst_model or self.primary_model_name
        self.session_id = session_id
        self.enable_verification = enable_verification
        self.max_search_results = max_search_results
        self.search_depth = search_depth

        # Initialize LLM instances with zero-temperature for maximum factual accuracy
        self.primary_llm = self._create_llm(self.primary_model_name)
        self.analyst_llm = self._create_llm(self.analyst_model_name)

        # Core Tools & Managers
        self.search_tool = TavilySearchTool(api_key=self.tavily_api_key)
        self.context_mgr = ContextManager()
        self.memory_mgr = MemoryManager(session_id=self.session_id)

    def _create_llm(self, model_name: str) -> ChatGroq:
        """Create ChatGroq instance configured for deterministic zero-temperature generation."""
        return ChatGroq(
            groq_api_key=self.groq_api_key or "missing_key",
            model_name=model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

    # --------------------------------------------------------------------------
    # Agent 1: PlannerAgent
    # --------------------------------------------------------------------------
    def _run_planner_agent(self, user_query: str) -> List[str]:
        """Formulate multi-angle, high-precision search strategy."""
        memory_ctx = self.memory_mgr.get_memory_context()
        planner_input = (
            f"User Research Query: {user_query}\n\n"
            f"Prior Conversation Memory:\n{memory_ctx}\n\n"
            "Task: Generate 3 to 5 distinct, keyword-dense search queries covering foundational concepts, "
            "metrics/benchmarks, recent updates, and known limitations."
        )

        try:
            response = self.primary_llm.invoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=planner_input),
            ])
            text = response.content.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                if isinstance(queries, list) and len(queries) > 0:
                    return [str(q).strip() for q in queries if str(q).strip()]
        except Exception as e:
            logger.warning(f"PlannerAgent error: {e}. Falling back to default query.")

        return [user_query]

    # --------------------------------------------------------------------------
    # Agent 2: WebScoutAgent
    # --------------------------------------------------------------------------
    def _run_web_scout_agent(self, search_queries: List[str], user_query: str) -> ProcessedContext:
        """Execute Tavily search, deduplicate content, and budget tokens."""
        raw_results = self.search_tool.multi_query_search(
            queries=search_queries,
            max_results_per_query=self.max_search_results,
            search_depth=self.search_depth,
        )
        processed_ctx = self.context_mgr.build_context(raw_results, user_query)
        return processed_ctx

    # --------------------------------------------------------------------------
    # Agent 3: DataAnalystAgent
    # --------------------------------------------------------------------------
    def _run_data_analyst_agent(self, user_query: str, processed_ctx: ProcessedContext) -> str:
        """Triage evidence, extract concrete metrics, and flag contradictions/gaps."""
        analyst_input = (
            f"User Research Query:\n{user_query}\n\n"
            f"Retrieved Source Context:\n{processed_ctx.formatted_context}\n\n"
            "Task:\n"
            "1. Extract all verified quantitative metrics, dates, and specifications.\n"
            "2. Identify any conflicting statements between sources.\n"
            "3. State any explicit information gaps missing from the sources."
        )

        try:
            response = self.analyst_llm.invoke([
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=analyst_input),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"DataAnalystAgent error: {e}. Continuing with direct context.")
            return "No pre-analyzed dossier available."

    # --------------------------------------------------------------------------
    # Agent 4: SynthesisWriterAgent
    # --------------------------------------------------------------------------
    def _run_synthesis_writer_agent(
        self,
        user_query: str,
        processed_ctx: ProcessedContext,
        evidence_dossier: str,
    ) -> str:
        """Author comprehensive, deeply structured, and citation-grounded report."""
        memory_ctx = self.memory_mgr.get_memory_context()

        synthesis_input = (
            f"User Research Query:\n{user_query}\n\n"
            f"Prior Conversation Memory:\n{memory_ctx}\n\n"
            f"Data Analyst Evidence Dossier:\n{evidence_dossier}\n\n"
            f"Verified Reference Sources ({processed_ctx.num_sources_used} sources):\n"
            f"{processed_ctx.formatted_context}\n\n"
            "Task:\n"
            "Synthesize an authoritative research report adhering strictly to the closed-world grounding rules.\n"
            "Anchor every factual statement with inline citations like [Source Title](URL).\n"
            "Include an Executive Summary, Technical Architecture, Comparative Benchmark Matrix, "
            "Data Gaps & Limitations, and a complete Verified Source Directory."
        )

        response = self.primary_llm.invoke([
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_input),
        ])
        return response.content.strip()

    # --------------------------------------------------------------------------
    # Agent 5: CriticVerifierAgent
    # --------------------------------------------------------------------------
    def _run_critic_verifier_agent(self, draft_report: str, processed_ctx: ProcessedContext) -> str:
        """Perform adversarial check against source excerpts to eliminate hallucinations."""
        critic_input = (
            f"Draft Research Report:\n{draft_report}\n\n"
            f"Original Reference Sources:\n{processed_ctx.formatted_context}\n\n"
            "Task:\n"
            "Audit every factual claim and citation link. Strip any unsupported statements or speculative leaps. "
            "Ensure citations accurately point to authentic URLs from the sources. Return the verified report."
        )

        try:
            response = self.primary_llm.invoke([
                SystemMessage(content=CRITIC_VERIFICATION_PROMPT),
                HumanMessage(content=critic_input),
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"CriticVerifierAgent error: {e}. Using initial draft.")
            return draft_report

    # --------------------------------------------------------------------------
    # Main Team Execution Pipeline
    # --------------------------------------------------------------------------
    def conduct_research(
        self,
        query: str,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> MultiAgentResearchResponse:
        """Execute the collaborative multi-agent research workflow."""
        def report(stage: str, data: Dict[str, Any]):
            if progress_callback:
                progress_callback(stage, data)

        # 0. Query Sanitization & Security Boundary
        clean_query = "".join(c for c in query if c.isprintable() or c in "\n\t").strip()
        if len(clean_query) > 2000:
            clean_query = clean_query[:2000]
        if not clean_query:
            raise ValueError("Research query cannot be empty.")
        query = clean_query

        # Step 1: PlannerAgent
        report("planner", {"agent": "PlannerAgent", "message": "Analyzing scope and formulating research queries..."})
        search_queries = self._run_planner_agent(query)
        report("queries_ready", {"agent": "PlannerAgent", "queries": search_queries})

        # Step 2: WebScoutAgent
        report("web_scout", {"agent": "WebScoutAgent", "message": f"Executing Tavily search across {len(search_queries)} queries..."})
        processed_ctx = self._run_web_scout_agent(search_queries, query)
        report("context_ready", {
            "agent": "WebScoutAgent",
            "sources_used": processed_ctx.num_sources_used,
            "sources_dropped": processed_ctx.num_sources_dropped,
            "estimated_tokens": processed_ctx.total_tokens_estimated,
        })

        # Step 3: DataAnalystAgent
        report("analyst", {"agent": "DataAnalystAgent", "message": "Triaging evidence, extracting atomic facts & checking contradictions..."})
        evidence_dossier = self._run_data_analyst_agent(query, processed_ctx)

        # Step 4: SynthesisWriterAgent
        report("writer", {"agent": "SynthesisWriterAgent", "message": f"Authoring comprehensive cited report with {self.primary_model_name}..."})
        draft_report = self._run_synthesis_writer_agent(query, processed_ctx, evidence_dossier)

        # Step 5: CriticVerifierAgent
        if self.enable_verification:
            report("critic", {"agent": "CriticVerifierAgent", "message": "Conducting adversarial fact-check and citation audit..."})
            verified_report = self._run_critic_verifier_agent(draft_report, processed_ctx)
        else:
            verified_report = draft_report

        # Step 6: HallucinationGuard
        report("guard", {"agent": "HallucinationGuard", "message": "Programmatic citation cross-validation & URL audit..."})
        audit = HallucinationGuard.audit_and_sanitize(verified_report, processed_ctx.sources)
        final_content = audit.sanitized_content

        report("grounding_verified", {
            "agent": "HallucinationGuard",
            "grounding_score": f"{int(audit.grounding_score * 100)}%",
            "valid_citations": audit.valid_citations,
            "hallucinated_citations": len(audit.hallucinated_citations),
        })

        # Step 7: Memory Update
        self.memory_mgr.add_turn(
            user_query=query,
            agent_response=final_content,
            search_queries_used=search_queries,
            citations_count=audit.valid_citations,
            summarizer_llm=self.primary_llm,
        )

        response = MultiAgentResearchResponse(
            query=query,
            content=final_content,
            search_queries=search_queries,
            sources=processed_ctx.sources,
            evidence_dossier=evidence_dossier,
            grounding_score=audit.grounding_score,
            total_citations=audit.total_citations,
            tokens_used_estimate=processed_ctx.total_tokens_estimated,
            session_id=self.session_id,
        )

        report("completed", {
            "message": f"Collaborative research completed successfully. Grounding Score: {int(audit.grounding_score * 100)}%"
        })
        return response

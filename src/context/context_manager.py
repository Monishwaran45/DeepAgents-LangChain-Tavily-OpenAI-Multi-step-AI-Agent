import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel
from src.tools.search import SearchResultItem
from src.config import AgentConfig


class ProcessedContext(BaseModel):
    """Cleaned, deduplicated, and budgeted context package for synthesis."""
    sources: List[SearchResultItem]
    formatted_context: str
    total_tokens_estimated: int
    num_sources_used: int
    num_sources_dropped: int


class ContextManager:
    """Manages search results, deduplication, token budgeting, and formatting."""

    def __init__(self, max_context_tokens: int = AgentConfig.MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic estimation: ~4 chars per token for English text."""
        return max(1, len(text) // 4)

    @staticmethod
    def normalize_url(url: str) -> str:
        """Strip trailing slash, query parameters, and fragments for reliable deduplication."""
        try:
            parsed = urlparse(url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            return clean_url.lower()
        except Exception:
            return url.strip().lower()

    def clean_snippet(self, text: str) -> str:
        """
        Clean noise, normalize whitespace, and defang potential prompt injection payloads
        embedded in untrusted web content.
        """
        # Remove null bytes and non-printable control characters
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")

        # Neutralize common LLM prompt injection markers and delimiters
        injection_delimiters = [
            r"<\|im_start\|>", r"<\|im_end\|>", r"<\|endoftext\|>",
            r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>",
            r"(?i)\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
            r"(?i)\bsystem\s+prompt\s+override\b",
        ]
        for pattern in injection_delimiters:
            text = re.sub(pattern, "[sanitized]", text)

        # Remove raw URLs embedded inside text snippets and collapse spaces
        text = re.sub(r"(https?://\S+)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def deduplicate(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Deduplicate results based on normalized URL and content fingerprint."""
        seen_urls = set()
        seen_snippets = set()
        unique_results: List[SearchResultItem] = []

        for item in results:
            norm_url = self.normalize_url(item.url)
            # Create a 60-character snippet fingerprint to catch identical syndicated articles
            clean_text = self.clean_snippet(item.content)
            snippet_fp = clean_text[:80].lower()

            if norm_url in seen_urls or (snippet_fp and snippet_fp in seen_snippets):
                continue

            seen_urls.add(norm_url)
            if snippet_fp:
                seen_snippets.add(snippet_fp)

            # Store cleaned content
            item.content = clean_text
            unique_results.append(item)

        return unique_results

    def rank_and_filter(
        self,
        results: List[SearchResultItem],
        query: str,
    ) -> List[SearchResultItem]:
        """Rank results by Tavily relevance score and keyword overlap with the query."""
        query_words = set(re.findall(r"\w+", query.lower()))

        def compute_score(item: SearchResultItem) -> float:
            base_score = item.score or 0.5
            text_words = set(re.findall(r"\w+", (item.title + " " + item.content).lower()))
            overlap = len(query_words.intersection(text_words)) / max(1, len(query_words))
            return base_score + (0.5 * overlap)

        sorted_results = sorted(results, key=compute_score, reverse=True)
        return sorted_results

    def build_context(
        self,
        results: List[SearchResultItem],
        query: str,
    ) -> ProcessedContext:
        """Deduplicate, rank, truncate to token budget, and format sources."""
        unique_results = self.deduplicate(results)
        ranked_results = self.rank_and_filter(unique_results, query)

        selected_sources: List[SearchResultItem] = []
        formatted_blocks: List[str] = []
        running_token_count = 0
        dropped_count = 0

        for idx, item in enumerate(ranked_results, start=1):
            source_block = (
                f"[Source {idx}]\n"
                f"Title: {item.title}\n"
                f"URL: {item.url}\n"
                f"Excerpt: {item.content}\n"
            )
            block_tokens = self.estimate_tokens(source_block)

            if running_token_count + block_tokens > self.max_context_tokens:
                dropped_count += 1
                continue

            running_token_count += block_tokens
            selected_sources.append(item)
            formatted_blocks.append(source_block)

        formatted_context = "\n---\n".join(formatted_blocks)

        return ProcessedContext(
            sources=selected_sources,
            formatted_context=formatted_context,
            total_tokens_estimated=running_token_count,
            num_sources_used=len(selected_sources),
            num_sources_dropped=dropped_count,
        )

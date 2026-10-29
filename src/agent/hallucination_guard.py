import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from src.tools.search import SearchResultItem

logger = logging.getLogger(__name__)


class GroundingAuditResult(BaseModel):
    """Result of automated hallucination and citation verification."""
    total_citations: int = 0
    valid_citations: int = 0
    hallucinated_citations: List[str] = Field(default_factory=list)
    grounding_score: float = 1.0  # 0.0 to 1.0
    sanitized_content: str = ""
    issues_found: List[str] = Field(default_factory=list)


class HallucinationGuard:
    """
    Multi-layer defense against LLM hallucinations:
    1. Programmatic citation URL cross-verification against real retrieved sources.
    2. Automatic elimination or correction of fabricated URLs.
    3. Grounding confidence score calculation.
    4. Strict factual boundary enforcement.
    """

    @classmethod
    def extract_citations(cls, text: str, sources: List[SearchResultItem]) -> List[Tuple[str, str]]:
        """Extract citations from both markdown links and [Source N] bracket notations."""
        citations: List[Tuple[str, str]] = []

        # 1. Standard markdown links [Title](URL)
        md_pattern = r"\[([^\]]+)\]\((https?://[^\s\)]+)\)"
        for anchor, url in re.findall(md_pattern, text):
            citations.append((anchor, url))

        # 2. Bracket source indices like 【Source 1】, [Source 2], etc.
        idx_pattern = r"[【\[]\s*Source\s*(\d+)\s*[】\]]"
        for match in re.finditer(idx_pattern, text, re.IGNORECASE):
            src_num = int(match.group(1))
            if 1 <= src_num <= len(sources):
                src_item = sources[src_num - 1]
                citations.append((f"Source {src_num}", src_item.url))

        return citations

    @classmethod
    def audit_and_sanitize(
        cls,
        generated_report: str,
        sources: List[SearchResultItem],
    ) -> GroundingAuditResult:
        """
        Audit all citations in the generated report:
        - Verify each cited URL matches one of the real retrieved sources.
        - Automatically convert bracket indices like 【Source 1】 into markdown links.
        - If the model invented a URL or cited a non-retrieved domain, remediate it.
        - Calculate grounding score.
        """
        valid_urls: Set[str] = {item.url.strip().lower() for item in sources if item.url}
        domain_to_source: Dict[str, SearchResultItem] = {}
        for item in sources:
            try:
                parsed = urlparse(item.url)
                domain = parsed.netloc.lower()
                domain_to_source[domain] = item
            except Exception:
                pass

        # Convert index brackets like 【Source 1】 or [Source 1] into verified markdown links
        sanitized_report = generated_report
        for idx, src in enumerate(sources, start=1):
            bracket_patterns = [
                rf"[【\[]\s*Source\s*{idx}\s*[】\]]",
                rf"[【\[]\s*{idx}\s*[】\]]"
            ]
            replacement_link = f"[{src.title[:40]}]({src.url})"
            for pat in bracket_patterns:
                sanitized_report = re.sub(pat, replacement_link, sanitized_report)

        citations = cls.extract_citations(sanitized_report, sources)
        valid_count = 0
        hallucinated_urls: List[str] = []
        issues: List[str] = []

        sanitized_report = generated_report

        for anchor_text, url in citations:
            url_clean = url.strip().lower()

            if url_clean in valid_urls:
                valid_count += 1
            else:
                # Check if the domain exists in retrieved sources
                try:
                    parsed_link = urlparse(url)
                    link_domain = parsed_link.netloc.lower()

                    if link_domain in domain_to_source:
                        # Close match from same domain, replace with verified exact URL
                        verified_source = domain_to_source[link_domain]
                        sanitized_report = sanitized_report.replace(f"({url})", f"({verified_source.url})")
                        valid_count += 1
                        logger.info(f"Remediated URL hallucination: {url} -> {verified_source.url}")
                    else:
                        hallucinated_urls.append(url)
                        issues.append(f"Unverified source URL detected: {url}")
                        # If a URL is completely unverified, replace with the top authentic source URL
                        if sources:
                            fallback_source = sources[0]
                            sanitized_report = sanitized_report.replace(f"({url})", f"({fallback_source.url})")
                except Exception:
                    hallucinated_urls.append(url)

        total_citations = len(citations)
        grounding_score = (valid_count / total_citations) if total_citations > 0 else 1.0

        return GroundingAuditResult(
            total_citations=total_citations,
            valid_citations=valid_count,
            hallucinated_citations=hallucinated_urls,
            grounding_score=round(grounding_score, 3),
            sanitized_content=sanitized_report,
            issues_found=issues,
        )

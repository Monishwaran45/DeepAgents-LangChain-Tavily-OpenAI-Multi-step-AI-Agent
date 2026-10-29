"""
Quickstart Example for Deep Research AI.
Demonstrates programmatic usage of both OmniResearchAgent and MultiAgentResearchTeam
powered by Groq LPU and Tavily Web Search.
"""

from src.agent.omni_research_agent import OmniResearchAgent
from src.agent.multi_agent_team import MultiAgentResearchTeam


def run_omni_research_example():
    print("=" * 60)
    print("RUNNING OMNI DEEP RESEARCH AGENT (Academic Mode)")
    print("=" * 60)

    agent = OmniResearchAgent(
        focus_mode="academic",
        session_id="quantum_research_session",
    )

    topic = "What are the latest breakthroughs in topological quantum computing in 2024-2025?"
    print(f"Researching: '{topic}'\n")

    def log_progress(stage: str, data: dict):
        print(f"[{stage.upper()}] {data.get('message', data)}")

    response = agent.research(topic, progress_callback=log_progress)

    print("\n" + "=" * 60)
    print("VERIFIED RESEARCH REPORT:")
    print("=" * 60)
    print(response.content)
    print("\n" + "=" * 60)
    print("Sources Indexed:")
    for s in response.sources:
        print(f"  [{s.index}] {s.domain} - {s.title[:50]}")

    print("\nRelated Questions Generated:")
    for i, q in enumerate(response.related_questions, start=1):
        print(f"  [{i}] {q}")

    print(f"\nGrounding Score: {int(response.grounding_score * 100)}%")


if __name__ == "__main__":
    run_omni_research_example()

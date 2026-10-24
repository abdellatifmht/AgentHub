from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from researchpilot.agents.analyst import analysis_agent
from researchpilot.agents.researcher import research_agent


class ResearchPilotState(TypedDict):
    query: str
    research_output: str
    sources: list[dict[str, Any]]
    analysis_report: dict[str, Any]


def run_researcher(state: ResearchPilotState) -> dict:
    """Run research and preserve both its summary and source records."""
    try:
        result = research_agent(state["query"])
        return {
            "research_output": result.get("summary", ""),
            "sources": result.get("sources") or [],
        }
    except Exception as exc:
        raise RuntimeError("Research stage failed.") from exc


def run_analysis(state: ResearchPilotState) -> dict:
    """Analyze the sources collected by the Research Agent."""
    sources = state.get("sources") or []
    if not sources:
        return {
            "analysis_report": {
                "source_summaries": [],
                "atomic_facts": [],
                "contradictions": [],
            }
        }
    try:
        return {"analysis_report": analysis_agent(sources)}
    except Exception as exc:
        raise RuntimeError("Analysis stage failed.") from exc


def build_graph():
    """Build the main ResearchPilot workflow."""

    builder = StateGraph(ResearchPilotState)

    builder.add_node("researcher", run_researcher)
    builder.add_node("analyst", run_analysis)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", END)

    return builder.compile()


research_pilot_graph = build_graph()


def run_research_pilot(query: str) -> dict[str, Any]:
    """Run research and source analysis for a research question."""
    if not query.strip():
        raise ValueError("Research query cannot be empty.")

    result = research_pilot_graph.invoke(
        {
            "query": query,
            "research_output": "",
            "sources": [],
            "analysis_report": {},
        }
    )

    return {
        "research_output": result["research_output"],
        "sources": result["sources"],
        "analysis_report": result["analysis_report"],
    }

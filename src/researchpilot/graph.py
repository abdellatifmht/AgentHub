from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from researchpilot.agents.researcher import researcher_graph


class ResearchPilotState(TypedDict):
    query: str
    research_output: str


def run_researcher(state: ResearchPilotState) -> dict:
    """Run the Research Agent on a given research question and return the final summary."""

    result = researcher_graph.invoke(
        {
            "messages": [
                HumanMessage(content=state["query"])
            ]
        },
        config={
            "recursion_limit": 20,
        }
    )

    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, str):
            return {
                "research_output": content
            }
    
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        content = "\n".join(text_parts)
        return {
            "research_output": content
        }

    return {
        "research_output": str(content)
    }


def buid_graph():
    """Build the main graph for the Research Pilot agent."""

    builder = StateGraph(ResearchPilotState)

    builder.add_node("researcher", run_researcher)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)

    return builder.compile()

research_pilot_graph = buid_graph()

def run_research_pilot(query: str) -> str:
    """Run the Research Pilot agent on a given research question and return the final summary."""

    result = research_pilot_graph.invoke(
        {
            "query": query,
            "research_output": ""
        }
    )

    return result["research_output"]

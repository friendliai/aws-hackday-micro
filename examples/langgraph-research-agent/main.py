"""
LangGraph Research Agent with FriendliAI

This example demonstrates a sophisticated multi-agent research system using LangGraph and FriendliAI.
The system orchestrates multiple specialized agents to conduct comprehensive research on any topic.

Key Features:
- Stateful graph-based workflow with LangGraph
- Multiple specialized agents (Researcher, Analyst, Critic)
- Conditional branching based on research quality
- Memory persistence across research sessions
- Streaming responses with real-time status updates
"""

import asyncio
import json
import os
from collections import deque
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

from friendli_app.sdk import AgentApp

app = AgentApp()

client = OpenAI(
    api_key=os.getenv("FRIENDLI_TOKEN"),
    base_url="https://api.friendli.ai/serverless/v1",
)

DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class ResearchState(TypedDict):
    """State schema for the research workflow"""

    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    research_findings: list[dict[str, Any]]
    analysis_results: dict[str, Any]
    critique: dict[str, Any]
    final_report: str | None
    iteration_count: int
    max_iterations: int
    quality_score: float
    quality_threshold: float
    status: str


class ResearchRequest(BaseModel):
    topic: str
    max_iterations: int = Field(default=3, ge=1, le=5)
    depth: Literal["quick", "standard", "comprehensive"] = "standard"
    stream: bool = False
    quality_threshold: float = Field(default=0.8, ge=0.1, le=1.0)


class ResearchResponse(BaseModel):
    topic: str
    status: str
    report: str | None = None
    findings: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}
    quality_score: float = 0.0
    timestamp: str


@tool
def web_search_tool(query: str) -> str:
    """Simulate web search for research purposes"""
    return f"Search results for '{query}': [Simulated findings about {query}. In production, this would use real search APIs.]"


@tool
def analyze_sources_tool(sources: list[str]) -> dict[str, Any]:
    """Analyze and validate information sources"""
    return {
        "credibility_score": 0.85,
        "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
        "contradictions": [],
        "confidence": "high",
    }


@tool
def generate_citations_tool(content: str) -> list[str]:
    """Generate proper citations for research content"""
    return ["[1] Source Title. Author Name. (2024)", "[2] Another Source. Different Author. (2024)"]


def create_researcher_agent(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create the researcher agent using FriendliAI"""
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("FRIENDLI_TOKEN"),
        base_url="https://api.friendli.ai/serverless/v1",
        temperature=0.7,
        streaming=True,
    )


def create_analyst_agent(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create the analyst agent using FriendliAI"""
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("FRIENDLI_TOKEN"),
        base_url="https://api.friendli.ai/serverless/v1",
        temperature=0.3,
        streaming=True,
    )


def create_critic_agent(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create the critic agent using FriendliAI"""
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("FRIENDLI_TOKEN"),
        base_url="https://api.friendli.ai/serverless/v1",
        temperature=0.5,
        streaming=True,
    )


def research_node(state: ResearchState) -> dict[str, Any]:
    """Research node that gathers information on the topic"""
    researcher = create_researcher_agent()

    research_prompt = f"""You are a research specialist.
    Research the topic: {state["topic"]}

    Current iteration: {state["iteration_count"]}/{state["max_iterations"]}

    Provide comprehensive findings including:
    1. Key concepts and definitions
    2. Current state and trends
    3. Important facts and statistics
    4. Relevant examples

    Be thorough and factual."""

    response = researcher.invoke(
        [
            SystemMessage(content="You are an expert researcher."),
            HumanMessage(content=research_prompt),
        ]
    )

    findings = {
        "iteration": state["iteration_count"],
        "content": response.content,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    state["research_findings"].append(findings)
    state["messages"].append(
        AIMessage(content=f"Research iteration {state['iteration_count']} completed.")
    )
    state["status"] = f"Research iteration {state['iteration_count']} completed"

    # Increment iteration count here, not in should_continue
    state["iteration_count"] += 1

    return state


def analysis_node(state: ResearchState) -> dict[str, Any]:
    """Analysis node that synthesizes research findings"""
    analyst = create_analyst_agent()

    all_findings = "\n\n".join(
        [f"Iteration {f['iteration']}: {f['content']}" for f in state["research_findings"]]
    )

    analysis_prompt = f"""You are an analytical expert.
    Analyze the following research findings on the topic: {state["topic"]}

    Research Findings:
    {all_findings}

    Provide:
    1. Synthesis of key insights
    2. Patterns and connections
    3. Implications and conclusions
    4. Confidence assessment
    """

    response = analyst.invoke(
        [SystemMessage(content="You are an expert analyst."), HumanMessage(content=analysis_prompt)]
    )

    state["analysis_results"] = {
        "synthesis": response.content,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    state["messages"].append(AIMessage(content="Analysis completed."))
    state["status"] = "Analysis completed"

    return state


def critique_node(state: ResearchState) -> dict[str, Any]:
    """Critique node that evaluates the research quality"""
    critic = create_critic_agent()

    critique_prompt = f"""You are a critical reviewer.
    Evaluate the research and analysis on: {state["topic"]}

    Analysis Results:
    {state["analysis_results"].get("synthesis", "")}

    Current iteration: {state["iteration_count"]}/{state["max_iterations"]}
    Depth mode: {"quick" if state["max_iterations"] == 1 else "standard" if state["max_iterations"] == 2 else "comprehensive"}

    Provide your assessment as a JSON object with these exact keys:
    - quality_score: A float between 0 and 1 (be conservative, especially for early iterations)
    - strengths: List of research strengths
    - weaknesses: List of gaps or weaknesses
    - recommendations: List of improvement suggestions

    For quick research (1 iteration), scores should typically be 0.5-0.7
    For standard research (2 iterations), scores should typically be 0.6-0.8
    For comprehensive research (3+ iterations), scores can be 0.7-0.95

    IMPORTANT: Return ONLY the JSON object, no other text.
    """

    response = critic.invoke(
        [
            SystemMessage(content="You are a critical reviewer. Always respond with valid JSON."),
            HumanMessage(content=critique_prompt),
        ]
    )

    try:
        # Try to extract JSON from the response
        content = response.content
        # Find JSON in the response (it might be wrapped in other text)
        import re

        json_match = re.search(r'\{[^{}]*"quality_score"[^{}]*\}', content, re.DOTALL)
        if json_match:
            critique_data = json.loads(json_match.group())
            state["quality_score"] = float(critique_data.get("quality_score", 0.7))
        else:
            # If no JSON found, use default
            state["quality_score"] = 0.7
            critique_data = {"quality_score": 0.7, "assessment": content}
    except Exception as e:
        # Fallback to default score
        state["quality_score"] = 0.7
        critique_data = {"quality_score": 0.7, "assessment": response.content, "error": str(e)}

    state["critique"] = critique_data
    state["messages"].append(
        AIMessage(content=f"Critique completed. Quality score: {state['quality_score']}")
    )
    state["status"] = f"Critique completed (Quality: {state['quality_score']:.2f})"

    return state


def report_node(state: ResearchState) -> dict[str, Any]:
    """Generate final research report"""
    reporter = create_researcher_agent()

    report_prompt = f"""Generate a comprehensive research report on: {state["topic"]}

    Based on:
    - Research Findings: {len(state["research_findings"])} iterations
    - Analysis: {state["analysis_results"].get("synthesis", "")}
    - Quality Score: {state["quality_score"]}

    Create a well-structured report with:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Conclusions
    5. Recommendations

    Make it comprehensive yet concise."""

    response = reporter.invoke(
        [SystemMessage(content="You are a report writer."), HumanMessage(content=report_prompt)]
    )

    state["final_report"] = response.content
    state["messages"].append(AIMessage(content="Final report generated."))
    state["status"] = "Research completed"

    return state


def should_continue(state: ResearchState) -> str:
    """Decide whether to continue research or generate report"""
    # Check if quality threshold is met
    if (
        state["quality_score"] >= state.get("quality_threshold", 0.8)
        or state["iteration_count"] > state["max_iterations"]
    ):
        return "report"
    else:
        return "research"


def create_research_graph() -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(ResearchState)

    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("research")

    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "critique")

    workflow.add_conditional_edges(
        "critique", should_continue, {"research": "research", "report": "report"}
    )

    workflow.add_edge("report", END)

    # Compile the workflow without checkpointer
    return workflow.compile()


research_graph = create_research_graph()


@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "name": "LangGraph Research Agent",
        "description": "Multi-agent research system using LangGraph and FriendliAI",
        "version": "1.0.0",
        "endpoints": {
            "/research": "POST - Conduct comprehensive research on a topic",
            "/research/explain": "POST - Explain how the workflow operates",
            "/callbacks/health": "POST - Health check",
        },
        "features": [
            "Stateful graph-based workflow with LangGraph",
            "Multiple specialized agents (Researcher, Analyst, Critic)",
            "Conditional branching based on research quality",
            "Memory persistence across research sessions",
            "Streaming responses with real-time updates",
        ],
        "workflow": {
            "nodes": ["research", "analysis", "critique", "report"],
            "flow": "Research → Analysis → Critique → (Continue or Report)",
        },
        "model": {"default": DEFAULT_MODEL, "provider": "FriendliAI"},
        "environment_required": ["FRIENDLI_TOKEN"],
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "ready",
    }


@app.callback
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "langgraph-research-agent",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/research")
async def conduct_research(request: ResearchRequest) -> Any:
    """
    Conduct comprehensive research on a topic using LangGraph workflow.

    Args:
        request: Research request with topic and parameters
    """
    if not request.topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Adjust max_iterations and quality_threshold based on depth
    depth_config = {
        "quick": {"max_iterations": 1, "quality_threshold": 0.6},
        "standard": {"max_iterations": 2, "quality_threshold": 0.75},
        "comprehensive": {"max_iterations": 3, "quality_threshold": 0.85},
    }

    config = depth_config.get(request.depth, depth_config["standard"])

    # Use request values if explicitly set, otherwise use depth-based defaults
    max_iterations = (
        request.max_iterations if request.max_iterations != 3 else config["max_iterations"]
    )
    quality_threshold = (
        request.quality_threshold
        if request.quality_threshold != 0.8
        else config["quality_threshold"]
    )

    initial_state = ResearchState(
        messages=[],
        topic=request.topic,
        research_findings=[],
        analysis_results={},
        critique={},
        final_report=None,
        iteration_count=1,
        max_iterations=max_iterations,
        quality_score=0.0,
        quality_threshold=quality_threshold,
        status="Starting research",
    )

    if request.stream:
        return StreamingResponse(
            stream_research_progress(initial_state), media_type="text/event-stream"
        )
    else:
        try:
            final_state = await asyncio.to_thread(
                research_graph.invoke, initial_state, {"recursion_limit": 15}
            )

            return ResearchResponse(
                topic=request.topic,
                status="completed",
                report=final_state.get("final_report"),
                findings=final_state.get("research_findings", []),
                analysis=final_state.get("analysis_results", {}),
                quality_score=final_state.get("quality_score", 0.0),
                timestamp=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_research_progress(initial_state: ResearchState):
    """Stream research progress updates"""
    try:
        status_queue = deque()
        final_state_data = {}

        async def run_graph():
            try:
                for output in research_graph.stream(initial_state, {"recursion_limit": 15}):
                    for key, value in output.items():
                        # Update final state data with latest values
                        if isinstance(value, dict):
                            final_state_data.update(value)

                        if "status" in value:
                            status_queue.append(
                                {
                                    "event": "status",
                                    "node": key,
                                    "status": value["status"],
                                    "iteration": value.get("iteration_count", 0),
                                    "quality_score": value.get("quality_score", 0.0),
                                }
                            )

                # Use accumulated state data instead of get_state
                status_queue.append(
                    {
                        "event": "complete",
                        "report": final_state_data.get("final_report", ""),
                        "quality_score": final_state_data.get("quality_score", 0.0),
                    }
                )
            except Exception as e:
                status_queue.append({"event": "error", "error": str(e)})

        task = asyncio.create_task(run_graph())

        while True:
            if status_queue:
                update = status_queue.popleft()
                yield f"data: {json.dumps(update)}\n\n"

                if update["event"] in ["complete", "error"]:
                    break
            else:
                await asyncio.sleep(0.1)
                if task.done() and not status_queue:
                    break

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"event": "error", "error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/research/explain")
async def explain_workflow():
    """
    Explain how the LangGraph research workflow operates.
    """
    return {
        "workflow": "LangGraph Research Agent",
        "description": "Multi-agent research system with iterative improvement",
        "nodes": {
            "research": "Gathers information on the topic",
            "analysis": "Synthesizes and analyzes findings",
            "critique": "Evaluates quality and completeness",
            "report": "Generates final comprehensive report",
        },
        "flow": [
            "1. Research node gathers initial information",
            "2. Analysis node synthesizes findings",
            "3. Critique node evaluates quality (0-1 score)",
            "4. If quality < 0.8 and iterations < max: return to research",
            "5. Otherwise: generate final report",
        ],
        "features": [
            "Stateful workflow with memory",
            "Conditional branching based on quality",
            "Multiple specialized agents",
            "Streaming progress updates",
            "Iterative refinement",
        ],
        "models": {"default": DEFAULT_MODEL, "provider": "FriendliAI Serverless"},
    }


def main():
    """Main entry point for the application"""
    print("🔬 LangGraph Research Agent with FriendliAI")
    print("=" * 50)
    print("📋 Available endpoints:")
    print("  - POST /research - Conduct research on a topic")
    print("  - POST /research/explain - Explain the workflow")
    print("  - POST /callbacks/health - Health check")
    print("\n⚙️ Features:")
    print("  ✅ Multi-agent orchestration with LangGraph")
    print("  ✅ Iterative research with quality control")
    print("  ✅ Conditional workflow branching")
    print("  ✅ Real-time streaming updates")
    print("  ✅ Powered by FriendliAI serverless")
    print("\n🌐 Server running on http://0.0.0.0:8080")
    print("💡 Set FRIENDLI_TOKEN environment variable!")

    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()

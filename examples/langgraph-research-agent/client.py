#!/usr/bin/env python3
"""
Client for testing the LangGraph Research Agent
"""

import json
import sys

import requests


def conduct_research(
    topic: str,
    max_iterations: int = 3,
    depth: str = "standard",
    stream: bool = False,
    base_url: str = "http://localhost:8080",
) -> None:
    """
    Conduct research on a topic using the LangGraph Research Agent.

    Args:
        topic: Research topic
        max_iterations: Maximum number of research iterations
        depth: Research depth (quick, standard, comprehensive)
        stream: Enable streaming for real-time updates
        base_url: Base URL of the research agent server
    """

    url = f"{base_url}/research"

    # Only send max_iterations if it's not the default value
    # Let the server determine based on depth
    payload = {"topic": topic, "depth": depth, "stream": stream}

    # Only add max_iterations if explicitly set (not default)
    if max_iterations != 3:
        payload["max_iterations"] = max_iterations

    # Determine display iterations based on depth if using default
    if max_iterations == 3:
        depth_iterations = {"quick": 1, "standard": 2, "comprehensive": 3}
        display_iterations = depth_iterations.get(depth, 2)
    else:
        display_iterations = max_iterations

    print(f"🔬 Starting research on: {topic}")
    print(f"   Depth: {depth}")
    print(f"   Expected iterations: {display_iterations}")
    print(f"   Streaming: {stream}")
    print("-" * 50)

    try:
        if stream:
            # Streaming request
            response = requests.post(
                url, json=payload, stream=True, headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            print("\n✅ Research completed!")
                            break

                        try:
                            data = json.loads(data_str)
                            if data.get("event") == "status":
                                print(f"📊 [{data.get('node')}] {data.get('status')}")
                                if data.get("quality_score", 0) > 0:
                                    print(f"   Quality Score: {data['quality_score']:.2f}")
                            elif data.get("event") == "complete":
                                print("\n📄 Final Report:")
                                print("-" * 50)
                                print(data.get("report", "No report generated"))
                                print("-" * 50)
                                print(f"Final Quality Score: {data.get('quality_score', 0):.2f}")
                            elif data.get("event") == "error":
                                print(f"❌ Error: {data.get('error')}")
                        except json.JSONDecodeError:
                            continue
        else:
            # Non-streaming request
            response = requests.post(url, json=payload)
            response.raise_for_status()

            result = response.json()

            print(f"Status: {result.get('status')}")
            print(f"Quality Score: {result.get('quality_score', 0):.2f}")
            print(f"Research Iterations: {len(result.get('findings', []))}")

            print("\n📄 Final Report:")
            print("-" * 50)
            print(result.get("report", "No report generated"))
            print("-" * 50)

            if result.get("findings"):
                print(f"\n📊 Research conducted in {len(result['findings'])} iterations")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def explain_workflow(base_url: str = "http://localhost:8080") -> None:
    """
    Get explanation of the LangGraph workflow.

    Args:
        base_url: Base URL of the research agent server
    """

    url = f"{base_url}/research/explain"

    try:
        response = requests.post(url)
        response.raise_for_status()

        result = response.json()

        print("🔬 LangGraph Research Agent Workflow")
        print("=" * 50)
        print(f"Description: {result.get('description')}")

        print("\n📋 Workflow Nodes:")
        for node, description in result.get("nodes", {}).items():
            print(f"  • {node}: {description}")

        print("\n🔄 Workflow Flow:")
        for step in result.get("flow", []):
            print(f"  {step}")

        print("\n✨ Features:")
        for feature in result.get("features", []):
            print(f"  • {feature}")

        print("\n🤖 Model Configuration:")
        models = result.get("models", {})
        print(f"  • Default Model: {models.get('default')}")
        print(f"  • Provider: {models.get('provider')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def health_check(base_url: str = "http://localhost:8080") -> None:
    """
    Check health of the research agent server.

    Args:
        base_url: Base URL of the research agent server
    """

    url = f"{base_url}/callbacks/health"

    try:
        response = requests.post(url)
        response.raise_for_status()

        result = response.json()
        print(f"✅ Server Status: {result.get('status')}")
        print(f"   Service: {result.get('service')}")
        print(f"   Timestamp: {result.get('timestamp')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not responding: {e}")
        sys.exit(1)


def main():
    """Main function with example usage"""

    import argparse

    parser = argparse.ArgumentParser(description="LangGraph Research Agent Client")
    parser.add_argument(
        "action", choices=["research", "explain", "health"], help="Action to perform"
    )
    parser.add_argument("--topic", "-t", type=str, help="Research topic (for research action)")
    parser.add_argument(
        "--iterations", "-i", type=int, default=3, help="Maximum iterations (1-5, default: 3)"
    )
    parser.add_argument(
        "--depth",
        "-d",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Research depth",
    )
    parser.add_argument("--stream", "-s", action="store_true", help="Enable streaming mode")
    parser.add_argument(
        "--url", "-u", type=str, default="http://localhost:8080", help="Base URL of the server"
    )

    args = parser.parse_args()

    if args.action == "research":
        if not args.topic:
            print("❌ Error: --topic is required for research action")
            sys.exit(1)
        conduct_research(
            topic=args.topic,
            max_iterations=args.iterations,
            depth=args.depth,
            stream=args.stream,
            base_url=args.url,
        )
    elif args.action == "explain":
        explain_workflow(base_url=args.url)
    elif args.action == "health":
        health_check(base_url=args.url)


if __name__ == "__main__":
    main()

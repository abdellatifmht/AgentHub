from .graph import run_research_pilot


def main():
    try:
        query = input("Enter your research query: ")
    except EOFError:
        return
    except Exception:
        return

    result = run_research_pilot(query)

    print("Research Agent Result:")
    if isinstance(result, dict):
        print(result.get("research_output", ""))
        print("\nSources:")
        for source in result.get("sources") or []:
            if isinstance(source, dict):
                print(f"- {source.get('url', '')}")
        print("\nAnalysis Report:")
        analysis_report = result.get("analysis_report") or {}
        print("Source Summaries:", analysis_report.get("source_summaries", []))
        print("Atomic Facts:", analysis_report.get("atomic_facts", []))
        print("Contradictions:", analysis_report.get("contradictions", []))
    else:
        print(result)

if __name__ == "__main__":
    main()

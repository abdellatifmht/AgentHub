from researchpilot.agents.researcher import research_agent


def main():

    query = input("Enter your research query: ")

    if not query.strip():
        print("Error: Search query cannot be empty.")
        return

    print("Starting research...")
    research_agent(query)


if __name__ == "__main__":
    main()

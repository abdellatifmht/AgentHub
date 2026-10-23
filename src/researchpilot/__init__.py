from .graph import run_research_pilot


def main():
    query = input("Enter your research query: ")

    result = run_research_pilot(query)

    print("Research Agent Result:")
    print(result)

if __name__ == "__main__":
    main()

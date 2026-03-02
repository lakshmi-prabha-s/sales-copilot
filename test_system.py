import argparse
import os
import pandas as pd
from cli import SalesCopilot


def initialize_copilot(context: str) -> SalesCopilot:
    print(f"Initializing Copilot for {context}...")
    return SalesCopilot()


def print_section(title: str, char: str = "=", width: int = 40):
    print("\n" + char * width)
    print(title)
    print(char * width)

EVALUATION_TEST_CASES = [
    {
        "query": "What are the primary pains flagged by Priya in the initial demo call?",
        "expected_topic": "Slow onboarding, lack of structured insight, no way to surface buying signals."
    },
    {
        "query": "How many AI minutes does the Growth tier include and what is the overage cost?",
        "expected_topic": "5,000 AI minutes, overage is ₹0.75/min."
    },
    {
        "query": "What is the contractual SLA for data deletion requests (DSR)?",
        "expected_topic": "21 days hard SLA, 30 days remedy credit."
    }
]


def run_tests(copilot=None):
    if copilot is None:
        copilot = initialize_copilot("testing")
    
    # --- 1. TEST INGESTION ---
    print_section("TEST 1: INGESTION")
    
    test_file_path = "data/test_dummy_call.txt"
    test_content = "[00:00] Prospect (Bob): Hello, I am calling to discuss the new AI SDLC project. The secret launch code is OMEGA-99."
    
    # Create a temporary dummy file
    with open(test_file_path, "w") as f:
        f.write(test_content)
    print(f"Created temporary mock file: {test_file_path}")
    
    # Run the ingestion method
    ingest_result = copilot.ingest_transcript(test_file_path)
    print(f"System Output: {ingest_result}")
    
    if "Successfully ingested" in ingest_result:
        print("Ingestion test passed.")
    else:
        print("Ingestion test failed.")

    # --- 2. TEST RETRIEVAL (New Data) ---
    print_section("TEST 2: RETRIEVAL (Dynamically Ingested Data)")
    
    dummy_query = "What is the secret launch code mentioned by Bob?"
    print(f"Query: {dummy_query}")
    dummy_response = copilot.ask(dummy_query)
    print(f"System Output: \n{dummy_response}\n")
    
    if "OMEGA-99" in dummy_response:
        print("Dynamic retrieval test passed.")
    else:
        print("Dynamic retrieval test failed.")

    # --- 3. TEST RETRIEVAL (Existing Context) ---
    print_section("TEST 3: RETRIEVAL (Existing Vector Store)")
    
    real_query = "What are the primary pains flagged by Priya in the initial demo call?"
    print(f"Query: {real_query}")
    real_response = copilot.ask(real_query)
    print(f"System Output: \n{real_response}\n")
    
    # Simple validation looking for keywords we know should be there
    if "onboarding" in real_response.lower() and "insight" in real_response.lower():
        print("Existing context retrieval test passed.")
    else:
        print("Retrieval might have failed. Please review the output manually to ensure accuracy.")

    # --- CLEANUP ---
    print_section("CLEANUP")
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"Deleted mock file: {test_file_path}")
        
    print("\nAll tests complete!")


def run_evaluation(copilot=None, output_file="evaluation_results.csv"):
    if copilot is None:
        copilot = initialize_copilot("evaluation")

    results = []

    print_section("Running Evaluation Suite...", char="-", width=30)
    for i, test in enumerate(EVALUATION_TEST_CASES, 1):
        print(f"Running Test {i}/{len(EVALUATION_TEST_CASES)}...")
        actual_response = copilot.ask(test["query"])

        results.append({
            "Query": test["query"],
            "Expected Context": test["expected_topic"],
            "Actual Response": actual_response
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print("\nEvaluation Complete.")
    print(f"Results saved to {output_file}. Please review manually for accuracy and citation formatting.")

    for _, row in df.iterrows():
        print(f"\nQ: {row['Query']}")
        print(f"Expected: {row['Expected Context']}")
        print(f"Got:\n{row['Actual Response']}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sales Copilot system and evaluation tests.")
    parser.add_argument(
        "--mode",
        choices=["system", "evaluation", "all"],
        default="all",
        help="Select which tests to run. Default: all"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "system":
        run_tests()
    elif args.mode == "evaluation":
        run_evaluation()
    else:
        run_evaluation()
        run_tests()
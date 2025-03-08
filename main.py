from decision_engine.decision_engine import DecisionEngine

if __name__ == "__main__":
    file_path = "sample_data.csv"

    engine = DecisionEngine(file_path)
    engine.run_pipeline()

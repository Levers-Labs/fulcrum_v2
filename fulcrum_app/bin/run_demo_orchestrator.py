import os
from fulcrum_app.metric_graph.graph_parser import parse_metric_graph_toml
from fulcrum_app.metric_graph.graph_service import GraphService
from fulcrum_app.query_manager.local_csv_query_manager import LocalCSVQueryManager
from fulcrum_app.intelligence_engine.patterns.performance_status import PerformanceStatusPattern
from fulcrum_app.storytelling_engine.story_service import StoryService
from fulcrum_app.intelligence_engine.data_structures import PatternOutput

def main():
    # 1. Load Metric Graph
    mg_path = "fulcrum_app/config/metric_graph.toml"
    with open(mg_path, "r") as f:
        mg_toml_str = f.read()
    graph_svc = GraphService()
    graph_svc.load_metric_graph(parse_metric_graph_toml(mg_toml_str))
    graph_svc.validate_metric_graph()

    # 2. Set up local CSV Query Manager
    csv_query_mgr = LocalCSVQueryManager(data_folder="local_data")

    # 3. For each metric, run a pattern
    pstatus = PerformanceStatusPattern()
    story_svc = StoryService()
    all_stories = []

    for mdef in graph_svc.all_metrics():
        # skip if there's no CSV, or handle exceptions
        try:
            df = csv_query_mgr.fetch_metric_time_series(mdef.id)
            # 4. Run Pattern
            window = {"start_date": str(df["date"].min()), "end_date": str(df["date"].max())}
            p_out = pstatus.run(metric_id=mdef.id, data=df, analysis_window=window, threshold=0.05)

            # 5. Generate Story from this PatternOutput
            new_stories = story_svc.generate_stories_for_outputs([p_out])
            all_stories.extend(new_stories)
        except FileNotFoundError:
            print(f"No CSV found for metric {mdef.id}; skipping...")

    # 6. Print or save stories
    for s in all_stories:
        print(f"\n--- STORY for metric {s.title} ---")
        print(s.body)
        print(f"Payload: {s.payload}")

if __name__ == "__main__":
    main()

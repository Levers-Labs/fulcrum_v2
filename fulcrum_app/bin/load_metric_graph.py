import os
import pandas as pd
from fulcrum_app.metric_graph.graph_parser import parse_metric_graph_toml
from fulcrum_app.metric_graph.graph_service import GraphService

def main():
    path = "fulcrum_app/config/metric_graph.toml"
    with open(path, "r") as f:
        toml_str = f.read()

    mg = parse_metric_graph_toml(toml_str)
    svc = GraphService()
    svc.load_metric_graph(mg)
    svc.validate_metric_graph()
    print("Metric Graph successfully loaded and validated!")
    # Optionally print out the metrics
    for m in svc.all_metrics():
        print(m)

if __name__ == "__main__":
    main()

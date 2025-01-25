# Metric Graph Module

This folder contains the data classes, parsers, and services for working with:
- **Metric Definitions** (including formulas, influences, and dimensions)
- **Funnel Definitions** (step-by-step metrics representing user flows)
- **Semantic Sync Checks** to validate references exist in the external data model

## Key Components

1. **graph_models.py**  
   - Defines `MetricDefinition`, `MetricGraph`, and associated sub-classes for formulas, influences, etc.

2. **graph_parser.py**  
   - Provides a `parse_metric_graph_toml(toml_str)` function that loads a TOML definition into `MetricGraph`.

3. **graph_service.py**  
   - `GraphService` loads and stores a `MetricGraph` in memory.
   - Provides validation (e.g., influences must reference existing metrics).
   - Future: can be extended to parse formulas and confirm referenced metrics exist.

4. **funnel_models.py, funnel_parser.py, funnel_service.py**  
   - Similar approach but for funnel definitions.
   - Each funnel is an ordered series of steps, each referencing a metric ID.

5. **semantic_sync_service.py**  
   - `sync_check_with_semantic_layer(graph_service, connector)` ensures all metric definitions map to actual measures in the semantic layer.

6. **mock_semantic_connector.py**  
   - A simple mock for demonstration. In production, you'd replace this with a real connector for Cube or dbt.

## Typical Flow

1. **Parse the Metric Graph**  
   ```python
   from graph_parser import parse_metric_graph_toml
   mg = parse_metric_graph_toml(my_toml_string)

2. **Load into GraphService**  
   ```python
   gs = GraphService()
   gs.load_metric_graph(mg)
   gs.validate_metric_graph()

3. **Parse Funnel Definitions**
   ```python
   from funnel_parser import parse_funnel_defs_toml
   fc = parse_funnel_defs_toml(my_funnel_toml_string)

4. **Load into FunnelService**
   ```python
   fs = FunnelService()
   fs.load_funnel_collection(fc)
   fs.validate_funnels(set(gs.get_graph().get_metric_ids()))

5. **Sync Check**
   ```python
   from mock_semantic_connector import MockSemanticConnector
   from semantic_sync_service import sync_check_with_semantic_layer
   
   connector = MockSemanticConnector(["metric_1_member", "some_other_metric_member"])
   sync_check_with_semantic_layer(gs, connector)
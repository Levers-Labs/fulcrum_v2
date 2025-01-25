# Intelligence Engine

This directory contains code for **Primitives** (low-level data manipulations) and **Patterns** (higher-level analyses) that produce insights about metrics.

## Overview

1. **Primitives**:  
   - Atomic analytical functions that do one thing well (e.g., calculating PoP growth, classifying metric status, detecting anomalies).  
   - Found in `primitives/` folder. Each file focuses on a domain area (time-series, dimension analysis, etc.).

2. **Patterns**:  
   - Assemblies of one or more Primitives to answer a specific business question: "Performance Status", "Root Cause", "Forecast," etc.  
   - Found in `patterns/` folder. Each pattern typically has a `.run(...)` method returning a `PatternOutput`.

3. **Patterns Manager**:  
   - Coordinates multiple patterns for a given metric/time window.  
   - Caches results (via `PatternCacheService`) so repeated calls don’t re-run expensive logic.

4. **Caching**:  
   - `caching_service.py` currently implements an **in-memory** store keyed by (metric_id, pattern_name, start_date, end_date).  
   - In production, you might store the JSON in a Postgres table or an S3 object store for quick retrieval.

5. **Data Structures**:  
   - `PatternOutput` in `data_structures.py` is the standard result object for all Patterns.  
   - It includes `pattern_name`, `pattern_version`, `metric_id`, `analysis_window`, and a free-form `results` dict.

## Typical Flow

1. **Fetch Data**: The Orchestrator or Patterns Manager obtains time-series data for a metric (e.g., from Query Manager).  
2. **Call a Pattern**: For example, `PerformanceStatusPattern.run(metric_id, df, analysis_window)`.  
3. **Output**: You get a `PatternOutput`.  
4. **Cache**: Patterns Manager or some higher-level orchestrator caches that output.  
5. **Downstream**: The Storytelling Engine will read these `PatternOutput` objects to generate “Stories.”

## Adding a New Pattern

1. Create a new file in `patterns/`, e.g. `forecast.py`.  
2. Write a class with a `.run(metric_id, data, analysis_window, ...) -> PatternOutput`.  
3. Inside `.run()`, call relevant primitives from `primitives/`.  
4. Return a `PatternOutput` with structured `results`.  
5. Optionally, update `PatternsManager` to run this new pattern automatically.

## Testing

- **Unit Tests** for each primitive: `tests/test_primitives.py`.  
- **Unit Tests** for each pattern: `tests/test_{pattern_name}.py`.  
- **Integration**: `tests/test_patterns_manager.py` covers how multiple patterns run in sequence with caching.

## Future Ideas

- **Parallel Execution**: For large data or many metrics, a distributed job runner might parallelize patterns.  
- **Advanced Primitives**: Additional anomaly detection (ML-based), correlation, forecasting with ARIMA, etc.  
- **Domain-Specific Patterns**: E.g., “Funnel Analysis Pattern,” “Cohort Analysis Pattern.”

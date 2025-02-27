# Metric Configuration Guide

This file describes how to maintain the `metrics.yaml` configuration file that defines **metrics**, their **formulas**, their **influence relationships**, and their **dimensions**.

## Overview

- Each **metric** is declared under the top-level `metrics:` key in `metrics.yaml`.
- A metric is identified by its **unique key** (e.g. `total_revenue`, `active_users`).
- All properties about a metric (its semantic reference, formulas, influences, dimensions, etc.) are **grouped together** under that key.

### Key Fields within Each Metric

| Field                | Description                                                                                          |
|----------------------|------------------------------------------------------------------------------------------------------|
| `label`             | A human-readable name that analysts and dashboards display.                                          |
| `type`              | The aggregation type for the metric, e.g. `count`, `sum`, `average`, `percentile`, `min`, `max`, `rate`. |
| `anchor_date`       | Time dimension on which the metric is anchored, e.g. `sale_date` or `user_signup_date`.              |
| `semantic_layer_ref`| The reference/path to the actual semantic-layer definition (e.g. Cube, dbt).                         |
| `dimensions`        | A list of dimension names that can be used to slice/dice the metric.                                 |
| `formulas`          | A **list** of zero or more *component formulas* that define how the metric can be computed from other metrics. |
| `influences`        | A **list** of zero or more *influence relationships*, specifying which other metric(s) are influenced by this metric (and optional metadata). |

### Formulas

Under the `formulas:` key (array of objects):
- `name`: A short identifier for the formula (e.g. `"basic_addition"` or `"advanced_margin"`).
- `expression`: A string with an algebraic expression referencing other metric keys (e.g. `"product_revenue + service_revenue"`).
  - You can use basic operators: `+`, `-`, `*`, `/`, parentheses, etc.
- **Example**:
  ```yaml
  formulas:
    - name: "basic_addition"
      expression: "product_revenue + service_revenue"
    - name: "alt_derivation"
      expression: "(service_revenue - refunds) * 1.08"
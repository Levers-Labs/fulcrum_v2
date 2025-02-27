{
  "schemaVersion": "1.0.0",
  "patternName": "PortfolioPerformance",
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",
  "grain": "day" | "week" | "month",

  "totalMetricsCount": 25,

  // On/off track summary
  "onTrackCount": 15,
  "offTrackCount": 10,
  "onTrackPercent": 60.0,
  "offTrackPercent": 40.0,
  "onTrackMetrics": ["MetricA", "MetricB", "..."],
  "offTrackMetrics": ["MetricC", "..."],

  // Newly flipped status
  "newlyOnTrackCount": 3,
  "newlyOffTrackCount": 2,
  "newlyOnTrackMetrics": ["MetricX", "MetricY"],
  "newlyOffTrackMetrics": ["MetricZ"],

  // Momentum
  "improvingCount": 10,
  "stableCount": 5,
  "decliningCount": 10,
  "improvingMetrics": ["Metric1", "Metric2"],
  "stableMetrics": ["Metric3"],
  "decliningMetrics": ["Metric4", "Metric5"]
}

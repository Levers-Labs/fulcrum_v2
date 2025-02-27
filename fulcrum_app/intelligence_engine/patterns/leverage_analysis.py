{
  "schemaVersion": "1.0.0",
  "patternName": "LeverageAnalysis",
  "metricId": "string",
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  "driverScenarios": [
    {
      "driverMetricId": "MetricX",
      "adjustmentPercent": 5.0,
      "projectedPercentChangeInOutput": 2.0,
      "projectedAbsoluteChangeInOutput": 10.0
    }
  ],
  "bestLeverageDriver": {
    "driverMetricId": "MetricX",
    "elasticityRanking": 1,
    "impactRanking": 2,
    "projectedPercentChangeInOutput": 2.0,
    "projectedAbsoluteChangeInOutput": 10.0
  }
}
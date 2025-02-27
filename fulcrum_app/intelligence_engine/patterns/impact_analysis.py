{
  "schemaVersion": "1.0.0",
  "patternName": "ImpactAnalysis",
  "metricId": "string",  
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",
  "grain": "day" | "week" | "month",

  "observedChangePercent": 10.0,
  "observedChangeAbsolute": 50.0,

  "childImpacts": [
    {
      "childMetricId": "string",
      "childChangePercent": 5.0,
      "childChangeAbsolute": 20.0,
      "shareOfParentImpactPercent": 50.0
    }
  ]
}
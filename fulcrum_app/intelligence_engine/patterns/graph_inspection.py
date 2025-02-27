{
  "patternName": "MetricGraph",
  "schemaVersion": "1.0",
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:MM:SS",
  "metricId": "string",

  "parents": [
    {
      "metricId": "MetricB",
      "relationshipType": "additive" | "subtractive" | ...
    },
    {
      "metricId": "MetricC",
      "relationshipType": "additive"
    }
  ],
  "children": [
    {
      "metricId": "MetricZ",
      "relationshipType": "additive"
    }
  ],
  // Possibly a topological sort or other advanced graph info
  "cycleDetected": false
}
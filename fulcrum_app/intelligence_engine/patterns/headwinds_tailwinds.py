{
  "schemaVersion": "1.0.0",
  "patternName": "HeadwindsTailwinds",
  "metricId": "active_users",
  "grain": "month",

  "analysisDate": "2025-02-01",
  "evaluationTime": "2025-02-01 00:10:00",

  // Leading Indicators
  "leadingIndicators": [
    {
      "driverMetric": "website_traffic",
      "method": "correlation",      // or "regression", etc.
      "trendDirection": "positive", // or "negative"
      "popTrendPercent": 10.0,      // e.g. +10% month over month
      "potentialImpactPercent": 5.0
    }
  ],

  // Seasonal outlook
  "seasonalOutlook": {
    "historicalChangePercent": -8.0,
    "periodLengthGrains": 1,       // e.g. "1 month"
    "expectedEndDate": "2025-03-01",
    "direction": "unfavorable"     // or "favorable"
  },

  // Volatility changes
  "volatility": {
    "currentVolatilityPercent": 12.0,
    "previousVolatilityPercent": 8.0,
    "volatilityChangePercent": 50.0
  },

  // Concentration risk across multiple dimensions
  "concentration": [
    {
      "dimensionName": "region",
      "topSegmentSlices": ["North America", "Europe", "APAC"],
      "topSegmentSharePercent": 70.0
    },
    {
      "dimensionName": "device_type",
      "topSegmentSlices": ["Mobile", "Desktop"],
      "topSegmentSharePercent": 80.0
    }
  ]
}
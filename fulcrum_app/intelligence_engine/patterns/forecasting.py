{
  "schemaVersion": "1.0.0",
  "patternName": "Forecasting",
  "metricId": "string",
  "grain": "day",

  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  // We define multiple forecast periods in one output
  // e.g. "endOfWeek", "endOfMonth", "endOfQuarter", "endOfNextMonth"
  "forecastPeriods": [
    {
      "periodName": "endOfWeek" | "endOfMonth" | "endOfQuarter" | "endOfNextMonth",

      // Statistical forecast
      "statisticalForecastedValue": 500.0,
      "statisticalTargetValue": 480.0,
      "statisticalForecastedGapPercent": 4.17,
      "statisticalForecastStatus": "On Track" | "Off Track",
      "statisticalLowerBound": 480.0,
      "statisticalUpperBound": 520.0,
      "statisticalConfidenceLevel": 0.95,

      // Pacing projection
      "percentOfPeriodElapsed": 50.0,
      "currentCumulativeValue": 220.0,
      "pacingProjectedValue": 440.0,
      "pacingGapPercent": -8.3,
      "pacingStatus": "On Track" | "Off Track",

      // Required performance to hit target
      "requiredPopGrowthPercent": 7.5,
      "pastPopGrowthPercent": 5.0,
      "deltaFromHistoricalGrowth": 2.5,
      "remainingGrainsCount": 2
    },
    // ...
  ],

  // Detailed day-by-day forecast for up to 60 days
  "dailyForecast": [
    {
      "date": "YYYY-MM-DD",
      "forecastedValue": 505.0,
      "lowerBound": 495.0,
      "upperBound": 515.0,
      "confidenceLevel": 0.95
    },
    // ...
  ]
}
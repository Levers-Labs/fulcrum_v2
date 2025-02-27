{
  "schemaVersion": "1.0.0",
  "patternName": "HistoricalPerformance",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  // Key dates/times
  "analysisDate": "YYYY-MM-DD",
  "evaluationTime": "YYYY-MM-DD HH:mm:ss",

  // Lookback info
  "lookbackStart": "YYYY-MM-DD",
  "lookbackEnd": "YYYY-MM-DD",
  "numPeriodsAnalyzed": 12,

  // PoP growth details across each period in the lookback
  // E.g. if grain=week and numPeriodsAnalyzed=12, you might store 12 growth entries
  "popGrowthRatesOverWindow": [
    {
      "periodStart": "YYYY-MM-DD",
      "periodEnd": "YYYY-MM-DD",
      "popGrowthPercent": 3.2
    },
    // ...
  ],

  // The "acceleration" or second derivative of the growth rate
  "accelerationRatesOverWindow": [
    {
      "periodStart": "YYYY-MM-DD",
      "periodEnd": "YYYY-MM-DD",
      "popAccelerationPercent": 1.1  // (growth this period - growth previous period)
    }
    // ...
  ],

  // Summaries for current vs. average growth
  "currentPopGrowthPercent": 8.5,
  "averagePopGrowthPercentOverWindow": 5.0,
  "currentGrowthAcceleration": 3.5,             // difference from average, or from prior

  // If we detect acceleration or slowing, how many periods in a row?
  "numPeriodsAccelerating": 2,
  "numPeriodsSlowing": 0,

  // Trend classification
  "trendType": "Stable" | "New Upward" | "New Downward" | "Plateau" | "None",
  "trendStartDate": "YYYY-MM-DD",
  "trendAveragePopGrowth": 4.2,

  // Previous trend info
  "previousTrendType": "Stable",
  "previousTrendStartDate": "YYYY-MM-DD",
  "previousTrendAveragePopGrowth": 2.1,
  "previousTrendDurationGrains": 6,

  // Record values
  "recordHigh": {
    "value": 1000,
    "rank": 1,
    "numPeriodsCompared": 36,

    "priorRecordHighValue": 950,
    "priorRecordHighDate": "YYYY-MM-DD",
    "absoluteDeltaFromPriorRecord": 50,
    "relativeDeltaFromPriorRecord": 5.26 // (50/950)*100
  },
  "recordLow": {
    "value": 300,
    "rank": 2,
    "numPeriodsCompared": 36,

    "priorRecordLowValue": 305,
    "priorRecordLowDate": "YYYY-MM-DD",
    "absoluteDeltaFromPriorRecord": -5,
    "relativeDeltaFromPriorRecord": -1.64
  },

  // Seasonal analysis
  "seasonality": {
    "isFollowingExpectedPattern": true,
    "expectedChangePercent": 10.0,
    "actualChangePercent": 9.2,
    "deviationPercent": -0.8
  },

  // Benchmark comparisons for typical reference periods
  // For day grain: priorWTD, MTD, priorQTD, priorYTD
  // For week grain: priorWeekOfMonth, priorWeekOfQuarter, priorWeekOfYear
  // For month grain: priorMonthOfQuarter, priorMonthOfYear
  "benchmarkComparisons": [
    {
      "referencePeriod": "priorWTD", // or "MTD", "priorQTD", etc.
      "absoluteChange": 30.0,       // current - reference
      "changePercent": 5.5
    },
    // ...
  ],

  // Trend exceptions (spike/drop), with absolute delta from normal range
  "trendExceptions": [
    {
      "type": "Spike" | "Drop",
      "currentValue": 950,
      "normalRangeLow": 800,
      "normalRangeHigh": 900,
      "absoluteDeltaFromNormalRange": 50,  // e.g. 950 - 900 or 800 - 750
      "magnitudePercent": 5.6             // optional or used similarly
    }
  ]
}
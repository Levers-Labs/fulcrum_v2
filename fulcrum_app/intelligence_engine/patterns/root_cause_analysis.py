{
  "schemaVersion": "1.0.0",
  "patternName": "RootCauseAnalysis",
  "metricId": "conversion_rate",
  "grain": "week",

  "analysisDate": "2025-02-05",
  "evaluationTime": "2025-02-05 03:25:00",

  // T0 and T1 with boundaries plus the metric's value in each window
  "analysisWindow": {
    "t0": {
      "startDate": "2025-01-22",
      "endDate": "2025-01-28",
      "metricValue": 5.2
    },
    "t1": {
      "startDate": "2025-01-29",
      "endDate": "2025-02-04",
      "metricValue": 4.6
    }
  },

  // Overall change in the metric from T0 to T1
  "metricDeltaAbsolute": -0.6,
  "metricDeltaPercent": -11.54,

  // Top 5 factors that contributed to this delta
  "topFactors": [
    {
      "rank": 1,
      "factorSubtype": "event_shock", // "seasonal_effect","component_value","segment_performance",
                                      // "segment_representation","influence_strength","influence_value","event_shock"
      
      "factorMetricName": null,       // Not relevant if it's an event
      "eventName": null,
      "factorDimensionName": null,
      "factorSliceName": null,

      "currentValue": null,           // e.g. no direct numeric measure for an "event shock"
      "priorValue": null,

      "factorChangeAbsolute": null,   // If we cannot quantify the event's own "absolute change," keep null
      "factorChangePercent": null,

      // This factor’s portion of the metric delta
      "contributionAbsolute": -0.3,
      "contributionPercent": 50.0
    },
    {
      "rank": 2,
      "factorSubtype": "segment_performance",
      "factorMetricName": null,       // Because it's still the same "conversion_rate" dimension
      "factorDimensionName": "region",
      "factorSliceName": "North America",

      "currentValue": 6.0,
      "priorValue": 6.5,
      "factorChangeAbsolute": -0.5,
      "factorChangePercent": -7.69,

      "contributionAbsolute": -0.2,
      "contributionPercent": 33.3
    },
    {
      "rank": 3,
      "factorSubtype": "influence_value",
      "factorMetricName": "ad_spend",
      "factorDimensionName": null,
      "factorSliceName": null,

      "currentValue": 12000,
      "priorValue": 15000,
      "factorChangeAbsolute": -3000,
      "factorChangePercent": -20.0,

      "contributionAbsolute": -0.1,
      "contributionPercent": 16.7
    }
    // up to 5
  ]
}
{
  "schemaVersion": "1.0.0",
  "patternName": "PerformanceStatus",
  "metricId": "string",
  "grain": "day" | "week" | "month",

  // Key dates/times
  "analysisDate": "YYYY-MM-DD",       // The period for which we’re analyzing performance
  "evaluationTime": "YYYY-MM-DD HH:mm:ss", // Actual time of this analysis run

  // Current vs. prior values
  "currentValue": 123.45,
  "priorValue": 120.00,
  "absoluteDeltaFromPrior": 3.45,               // e.g. currentValue - priorValue
  "popChangePercent": 2.875,                    // e.g. (3.45 / 120.0) * 100

  // Target-related fields
  // "gap" is only used if we’re off track (i.e. failing target).
  // if the metric is above target, store that in overperformance.
  "targetValue": 118.0,
  "status": "On Track" | "Off Track" | "Overperforming",
  "absoluteGap": 0.0,                            // if Off Track: how far below target are we in absolute terms? null if not below target
  "absoluteGapDelta": -100,                      // how much is the absolute gap shrinking or growing by relative to the last period
  "percentGap": 0.0,                            // if Off Track: how far below target are we in percentage terms? null if not below target
  "percentGapDelta": -1.2,                      // how much is the relative gap shrinking or growing by relative to the last period
  "absoluteOverperformance": 0.0,               // if Overperforming: how far above target in absolute terms? null if not overperforming
  "percentOverperformance": 0.0,                // if Overperforming: how far above target in percent terms? null if not overperforming
  "overperformancePercent": 4.58,               // if Overperforming: how far above target?

  // Status change info
  "statusChange": {
    "hasFlipped": true,
    "oldStatus": "Off Track",
    "newStatus": "On Track",
    "oldStatusDurationGrains": 2
  },

  // Streak info
  "streak": {
    "length": 3,
    "status": "On Track",
    "performanceChangePercentOverStreak": 15.4,
    "absoluteChangeOverStreak": 20.0,              // e.g. total absolute increase over that streak
    "averageChangePercentPerGrain": 5.13,          // 15.4% / 3
    "averageChangeAbsolutePerGrain": 6.67          // 20 / 3
  },

  // "Hold steady" scenario: if the metric is already above or exactly at target
  "holdSteady": {
    "isCurrentlyAtOrAboveTarget": true,
    "timeToMaintainGrains": 3,
    "currentMarginPercent": 2.0
  }
}
"""
DSensei Core Analysis Logic

This module provides standalone implementations of DSensei's key analytics functionality
without external dependencies. It allows for identifying key drivers in data based on
metrics, dimensions, and time periods.

Usage:
    insights = analyze_key_drivers(
        df,
        baseline_date_range=('2023-01-01', '2023-01-31'),
        comparison_date_range=('2023-02-01', '2023-02-28'),
        group_by_columns=['country', 'device', 'channel'],
        metric_column='revenue',
        agg_method='sum',
        date_column='date',
        expected_value=0,
        max_dimensions=3
    )
"""

import datetime
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable


class AggregateMethod(Enum):
    """Aggregation methods for metrics."""
    COUNT = "count"
    DISTINCT = "nunique"
    SUM = "sum"


class FilterOperator(Enum):
    """Filter operators for filtering data."""
    EQ = "eq"
    NEQ = "neq"
    EMPTY = "empty"
    NON_EMPTY = "non_empty"


@dataclass
class Filter:
    """Data filter specification."""
    column: str
    operator: FilterOperator
    values: Optional[List[Any]] = None


@dataclass
class DimensionValuePair:
    """Dimension and value pair for segment identification."""
    dimension: str
    value: str


@dataclass
class PeriodValue:
    """Metrics for a specific time period."""
    slice_count: int = 0
    slice_size: float = 0
    slice_value: float = 0


@dataclass
class SegmentInfo:
    """Information about a specific data segment."""
    key: Tuple[DimensionValuePair, ...] = None
    serialized_key: str = None
    baseline_value: PeriodValue = None
    comparison_value: PeriodValue = None
    impact: float = 0
    change_percentage: float = 0
    change_dev: Optional[float] = None
    absolute_contribution: Optional[float] = None
    confidence: Optional[float] = None
    sort_value: Optional[float] = None


@dataclass
class Dimension:
    """Dimension information with statistical significance."""
    name: str
    score: float = 0
    is_key_dimension: bool = False
    values: Set[str] = field(default_factory=set)


@dataclass
class ValueByDate:
    """Metric value for a specific date."""
    date: str
    value: float


@dataclass
class MetricInsight:
    """Complete analysis for a metric."""
    name: str = None
    total_segments: int = 0
    expected_change_percentage: float = 0
    aggregation_method: str = None
    baseline_num_rows: int = 0
    comparison_num_rows: int = 0
    baseline_value: float = 0
    comparison_value: float = 0
    baseline_value_by_date: List[ValueByDate] = field(default_factory=list)
    comparison_value_by_date: List[ValueByDate] = field(default_factory=list)
    baseline_date_range: List[str] = field(default_factory=list)
    comparison_date_range: List[str] = field(default_factory=list)
    top_driver_slice_keys: List[str] = field(default_factory=list)
    dimensions: Dict[str, Dimension] = field(default_factory=dict)
    dimension_slice_info: Dict[str, SegmentInfo] = field(default_factory=dict)
    key_dimensions: List[str] = field(default_factory=list)
    filters: List[Filter] = field(default_factory=list)


def format_date(date_value):
    """Convert a date value to a consistent string format."""
    if isinstance(date_value, str):
        return date_value
    elif isinstance(date_value, datetime.date):
        return date_value.strftime('%Y-%m-%d')
    return str(date_value)


def parse_date(date_string):
    """Parse a date string to a datetime.date object."""
    if isinstance(date_string, datetime.date):
        return date_string
    
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_string, fmt).date()
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date string: {date_string}")


def apply_filter(df, filter_obj):
    """Apply a filter to a DataFrame."""
    if filter_obj.operator == FilterOperator.EQ:
        return df[df[filter_obj.column].isin(filter_obj.values)]
    elif filter_obj.operator == FilterOperator.NEQ:
        return df[~df[filter_obj.column].isin(filter_obj.values)]
    elif filter_obj.operator == FilterOperator.EMPTY:
        return df[df[filter_obj.column].isna() | (df[filter_obj.column] == '')]
    elif filter_obj.operator == FilterOperator.NON_EMPTY:
        return df[df[filter_obj.column].notna() & (df[filter_obj.column] != '')]
    return df


def apply_filters(df, filters):
    """Apply multiple filters to a DataFrame."""
    if not filters:
        return df
    
    filtered_df = df.copy()
    for filter_obj in filters:
        filtered_df = apply_filter(filtered_df, filter_obj)
    
    return filtered_df


def calculate_aggregation(df, column, method, group_by=None):
    """Calculate aggregated values for a column using specified method."""
    if group_by is None:
        if method == AggregateMethod.SUM:
            return df[column].sum()
        elif method == AggregateMethod.COUNT:
            return df[column].count()
        elif method == AggregateMethod.DISTINCT:
            return df[column].nunique()
    else:
        grouped = df.groupby(group_by)
        if method == AggregateMethod.SUM:
            return grouped[column].sum()
        elif method == AggregateMethod.COUNT:
            return grouped[column].count()
        elif method == AggregateMethod.DISTINCT:
            return grouped[column].nunique()


def filter_by_date_range(df, date_column, start_date, end_date):
    """Filter a DataFrame to include only rows within a date range."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    if isinstance(df[date_column].iloc[0], str):
        # Convert date column to datetime if it's string
        df = df.copy()
        df[date_column] = df[date_column].apply(parse_date)
    
    mask = (df[date_column] >= start) & (df[date_column] <= end)
    return df[mask]


def safe_divide(numerator, denominator):
    """Safely divide numbers, handling division by zero."""
    if denominator == 0:
        return 0 if numerator == 0 else (1 if numerator > 0 else -1)
    return numerator / denominator


def calculate_change_percentage(current, baseline):
    """Calculate percentage change between current and baseline values."""
    return safe_divide(current - baseline, baseline)


def generate_serialized_key(dimension_pairs):
    """Generate a serialized string key from dimension-value pairs."""
    sorted_pairs = sorted(dimension_pairs, key=lambda x: x.dimension)
    return "|".join([f"{pair.dimension}:{pair.value}" for pair in sorted_pairs])


def calculate_total_segments(dimensions):
    """Calculate the total number of potential segments from dimensions."""
    total = 0
    dim_lengths = [len(dim.values) for dim in dimensions.values()]
    
    # Sum up all possible combinations of dimensions
    for i in range(1, len(dim_lengths) + 1):
        for combo in itertools.combinations(dim_lengths, i):
            total += math.prod(combo)
            
    return total


def get_values_by_date(df, date_column, metric_column, agg_method):
    """Get metric values aggregated by date."""
    result = []
    
    # Ensure date column is in date format
    df = df.copy()
    if isinstance(df[date_column].iloc[0], str):
        df[date_column] = df[date_column].apply(parse_date)
    
    # Group by date and calculate metric
    grouped = df.groupby(date_column)
    if agg_method == AggregateMethod.SUM:
        agg_values = grouped[metric_column].sum()
    elif agg_method == AggregateMethod.COUNT:
        agg_values = grouped[metric_column].count()
    elif agg_method == AggregateMethod.DISTINCT:
        agg_values = grouped[metric_column].nunique()
    
    # Convert to list of ValueByDate objects
    for date, value in agg_values.items():
        result.append(ValueByDate(format_date(date), float(value)))
    
    # Sort by date
    result.sort(key=lambda x: x.date)
    return result


def find_key_dimensions(df, dimensions, metric_column, baseline_df, comparison_df, agg_method):
    """Identify the key dimensions driving changes in metrics."""
    key_dimensions = []
    dimension_scores = {}

    # Calculate the weights for each dimension
    for dimension in dimensions:
        # Skip if the dimension has too many values (> 100)
        unique_values = df[dimension].nunique()
        if unique_values > 100:
            continue
            
        baseline_metric_by_dim = calculate_aggregation(
            baseline_df, metric_column, agg_method, dimension)
        comparison_metric_by_dim = calculate_aggregation(
            comparison_df, metric_column, agg_method, dimension)
        
        # Join the two series
        combined = pd.DataFrame({
            'baseline': baseline_metric_by_dim,
            'comparison': comparison_metric_by_dim
        }).fillna(0)
        
        # Calculate the total metrics
        baseline_total = baseline_metric_by_dim.sum()
        comparison_total = comparison_metric_by_dim.sum()
        
        # Calculate weights and changes
        combined['weight'] = (combined['baseline'] + combined['comparison']) / (baseline_total + comparison_total)
        combined['change'] = combined.apply(
            lambda row: calculate_change_percentage(row['comparison'], row['baseline']), 
            axis=1
        )
        
        # Calculate weighted change
        combined['weighted_change'] = combined['weight'] * combined['change']
        weighted_change_mean = combined['weighted_change'].sum()
        
        # Calculate standard deviation of changes
        combined['squared_diff'] = combined['weight'] * (combined['change'] - weighted_change_mean)**2
        weighted_std = math.sqrt(combined['squared_diff'].sum())
        
        # Store the score for this dimension
        dimension_scores[dimension] = weighted_std
    
    # Determine key dimensions (high variance dimensions)
    if dimension_scores:
        mean_score = sum(dimension_scores.values()) / len(dimension_scores)
        for dim, score in dimension_scores.items():
            if score > mean_score or score > 0.02:
                key_dimensions.append(dim)
    
    return key_dimensions, dimension_scores


def analyze_segments(df, baseline_df, comparison_df, dimensions, metric_column, agg_method, expected_value, max_dimensions):
    """Analyze segments to identify drivers of change."""
    segments_info = {}
    baseline_count = len(baseline_df)
    comparison_count = len(comparison_df)
    
    # Calculate the overall metric values
    baseline_total = calculate_aggregation(baseline_df, metric_column, agg_method)
    comparison_total = calculate_aggregation(comparison_df, metric_column, agg_method)
    
    # Generate all possible dimension combinations up to max_dimensions
    all_dimension_combinations = []
    for i in range(1, min(max_dimensions, len(dimensions)) + 1):
        all_dimension_combinations.extend(itertools.combinations(dimensions, i))
    
    # Analyze each dimension combination
    for dim_combo in all_dimension_combinations:
        # Group by the dimension combination
        baseline_grouped = baseline_df.groupby(list(dim_combo))[metric_column]
        comparison_grouped = comparison_df.groupby(list(dim_combo))[metric_column]
        
        if agg_method == AggregateMethod.SUM:
            baseline_agg = baseline_grouped.sum()
            comparison_agg = comparison_grouped.sum()
        elif agg_method == AggregateMethod.COUNT:
            baseline_agg = baseline_grouped.count()
            comparison_agg = comparison_grouped.count()
        elif agg_method == AggregateMethod.DISTINCT:
            baseline_agg = baseline_grouped.nunique()
            comparison_agg = comparison_grouped.nunique()
        
        # Convert to DataFrame and join
        baseline_df_agg = baseline_agg.reset_index()
        comparison_df_agg = comparison_agg.reset_index()
        
        # Merge the two DataFrames
        merged = pd.merge(
            baseline_df_agg, 
            comparison_df_agg, 
            on=list(dim_combo), 
            how='outer', 
            suffixes=('_baseline', '_comparison')
        ).fillna(0)
        
        # Calculate metrics for each segment
        for _, row in merged.iterrows():
            # Create dimension-value pairs
            dimension_pairs = [
                DimensionValuePair(dim, str(row[dim])) 
                for dim in dim_combo
            ]
            
            # Generate serialized key
            serialized_key = generate_serialized_key(dimension_pairs)
            
            # Calculate counts and values
            baseline_value = float(row[f'{metric_column}_baseline'])
            comparison_value = float(row[f'{metric_column}_comparison'])
            
            # Calculate slice counts (approximating from the original DSensei logic)
            baseline_slice_count = baseline_grouped.count().get(
                tuple(row[dim] for dim in dim_combo), 0)
            comparison_slice_count = comparison_grouped.count().get(
                tuple(row[dim] for dim in dim_combo), 0)
            
            # Calculate period values
            baseline_period = PeriodValue(
                baseline_slice_count,
                safe_divide(baseline_slice_count, baseline_count),
                baseline_value
            )
            
            comparison_period = PeriodValue(
                comparison_slice_count,
                safe_divide(comparison_slice_count, comparison_count),
                comparison_value
            )
            
            # Calculate change metrics
            impact = comparison_value - baseline_value
            change_percentage = calculate_change_percentage(comparison_value, baseline_value)
            
            # Calculate absolute contribution (simplified version)
            # How much this segment contributes to the overall change
            segment_contribution = impact / (comparison_total - baseline_total) if comparison_total != baseline_total else 0
            
            # Calculate change deviation (z-score equivalent)
            # How unusual is this change compared to other segments
            change_dev = abs(change_percentage - expected_value)
            
            # Create segment info object
            segment_info = SegmentInfo(
                key=tuple(dimension_pairs),
                serialized_key=serialized_key,
                baseline_value=baseline_period,
                comparison_value=comparison_period,
                impact=impact,
                change_percentage=change_percentage,
                change_dev=change_dev,
                absolute_contribution=segment_contribution,
                sort_value=abs(impact)
            )
            
            segments_info[serialized_key] = segment_info
    
    return segments_info


def create_dimension_objects(df, dimensions, dimension_scores):
    """Create Dimension objects with metadata."""
    dimension_objects = {}
    
    for dim in dimensions:
        values = set(df[dim].astype(str).unique())
        
        # Remove null or empty values
        if None in values:
            values.remove(None)
        if '' in values:
            values.remove('')
        if 'nan' in values:
            values.remove('nan')
        
        # Create dimension object
        score = dimension_scores.get(dim, 0)
        is_key = dim in dimension_scores and (
            score > sum(dimension_scores.values()) / len(dimension_scores) or score > 0.02
        )
        
        dimension_objects[dim] = Dimension(
            name=dim,
            score=score,
            is_key_dimension=is_key,
            values=values
        )
    
    return dimension_objects


def analyze_key_drivers(df, baseline_date_range, comparison_date_range, 
                       group_by_columns, metric_column, agg_method='sum',
                       date_column='date', expected_value=0, filters=None,
                       max_dimensions=3):
    """
    Analyze key drivers in data by comparing metrics between two time periods.
    
    Args:
        df: Pandas DataFrame containing the data
        baseline_date_range: Tuple of (start_date, end_date) for the baseline period
        comparison_date_range: Tuple of (start_date, end_date) for the comparison period
        group_by_columns: List of dimension columns to group by
        metric_column: Column containing the metric to analyze
        agg_method: Aggregation method ('sum', 'count', or 'nunique')
        date_column: Column containing dates
        expected_value: Expected change percentage (default: 0)
        filters: List of Filter objects to apply
        max_dimensions: Maximum number of dimensions to combine
        
    Returns:
        MetricInsight object containing the analysis results
    """
    # Convert string agg_method to enum
    if isinstance(agg_method, str):
        agg_method_map = {
            'sum': AggregateMethod.SUM,
            'count': AggregateMethod.COUNT,
            'nunique': AggregateMethod.DISTINCT
        }
        agg_method = agg_method_map.get(agg_method.lower(), AggregateMethod.SUM)
    
    # Apply filters
    filtered_df = apply_filters(df, filters) if filters else df.copy()
    
    # Filter by date ranges
    baseline_df = filter_by_date_range(
        filtered_df, date_column, 
        baseline_date_range[0], baseline_date_range[1]
    )
    
    comparison_df = filter_by_date_range(
        filtered_df, date_column, 
        comparison_date_range[0], comparison_date_range[1]
    )
    
    # Find key dimensions
    key_dimensions, dimension_scores = find_key_dimensions(
        filtered_df, group_by_columns, metric_column, 
        baseline_df, comparison_df, agg_method
    )
    
    # Create dimension objects
    dimension_objects = create_dimension_objects(
        filtered_df, group_by_columns, dimension_scores
    )
    
    # Analyze segments
    segments_info = analyze_segments(
        filtered_df, baseline_df, comparison_df, 
        group_by_columns, metric_column, agg_method,
        expected_value, max_dimensions
    )
    
    # Get top driver slice keys
    top_drivers = []
    if key_dimensions:
        # Only include segments with key dimensions
        key_segments = {}
        for key, segment in segments_info.items():
            segment_dimensions = {pair.dimension for pair in segment.key}
            if segment_dimensions.issubset(set(key_dimensions)):
                key_segments[key] = segment
        
        # Sort by impact (absolute value)
        sorted_segments = sorted(
            key_segments.items(), 
            key=lambda x: abs(x[1].impact), 
            reverse=True
        )
        
        # Take top 1000 segments
        top_drivers = [key for key, _ in sorted_segments[:1000]]
    
    # Calculate baseline and comparison totals
    baseline_num_rows = len(baseline_df)
    comparison_num_rows = len(comparison_df)
    
    baseline_value = calculate_aggregation(baseline_df, metric_column, agg_method)
    comparison_value = calculate_aggregation(comparison_df, metric_column, agg_method)
    
    # Get values by date
    baseline_values_by_date = get_values_by_date(
        baseline_df, date_column, metric_column, agg_method
    )
    comparison_values_by_date = get_values_by_date(
        comparison_df, date_column, metric_column, agg_method
    )
    
    # Create MetricInsight object
    insight = MetricInsight(
        name=f"{agg_method.name} of {metric_column}",
        total_segments=calculate_total_segments(dimension_objects),
        expected_change_percentage=expected_value,
        aggregation_method=agg_method.name,
        baseline_num_rows=baseline_num_rows,
        comparison_num_rows=comparison_num_rows,
        baseline_value=baseline_value,
        comparison_value=comparison_value,
        baseline_value_by_date=baseline_values_by_date,
        comparison_value_by_date=comparison_values_by_date,
        baseline_date_range=[
            format_date(baseline_date_range[0]), 
            format_date(baseline_date_range[1])
        ],
        comparison_date_range=[
            format_date(comparison_date_range[0]), 
            format_date(comparison_date_range[1])
        ],
        top_driver_slice_keys=top_drivers,
        dimensions=dimension_objects,
        dimension_slice_info=segments_info,
        key_dimensions=key_dimensions,
        filters=filters or []
    )
    
    return insight


def get_dimension_slice_details(insight, slice_key):
    """
    Get detailed information about a specific dimension slice.
    
    Args:
        insight: MetricInsight object from analyze_key_drivers
        slice_key: Serialized key of the slice
        
    Returns:
        SegmentInfo object for the slice if found, None otherwise
    """
    return insight.dimension_slice_info.get(slice_key)


def get_waterfall_insight(df, baseline_date_range, comparison_date_range, 
                         segment_keys, metric_column, agg_method='sum',
                         date_column='date', filters=None):
    """
    Generate waterfall insight showing contribution of each segment.
    
    Args:
        df: Pandas DataFrame containing the data
        baseline_date_range: Tuple of (start_date, end_date) for the baseline period
        comparison_date_range: Tuple of (start_date, end_date) for the comparison period
        segment_keys: List of lists of DimensionValuePair objects defining segments
        metric_column: Column containing the metric to analyze
        agg_method: Aggregation method ('sum', 'count', or 'nunique')
        date_column: Column containing dates
        filters: List of Filter objects to apply
        
    Returns:
        Dictionary mapping segment keys to their contribution metrics
    """
    # Convert string agg_method to enum
    if isinstance(agg_method, str):
        agg_method_map = {
            'sum': AggregateMethod.SUM,
            'count': AggregateMethod.COUNT,
            'nunique': AggregateMethod.DISTINCT
        }
        agg_method = agg_method_map.get(agg_method.lower(), AggregateMethod.SUM)
    
    # Apply filters
    filtered_df = apply_filters(df, filters) if filters else df.copy()
    
    result = {}
    working_df = filtered_df.copy()
    
    for segment_key in segment_keys:
        # Create filtering expression for this segment
        segment_filter = pd.Series(True, index=working_df.index)
        for pair in segment_key:
            segment_filter &= (working_df[pair.dimension].astype(str) == pair.value)
        
        # Extract segment data
        segment_df = working_df[segment_filter]
        
        # Remove this segment from working data for next iterations
        working_df = working_df[~segment_filter]
        
        # Filter by date ranges
        baseline_segment = filter_by_date_range(
            segment_df, date_column, 
            baseline_date_range[0], baseline_date_range[1]
        )
        
        comparison_segment = filter_by_date_range(
            segment_df, date_column, 
            comparison_date_range[0], comparison_date_range[1]
        )
        
        # Calculate values
        baseline_value = calculate_aggregation(baseline_segment, metric_column, agg_method)
        comparison_value = calculate_aggregation(comparison_segment, metric_column, agg_method)
        
        # Create serialized key
        serialized_key = "|".join([
            f"{pair.dimension}:{pair.value}" for pair in segment_key
        ])
        
        # Store result
        result[serialized_key] = {
            "changeWithNoOverlap": comparison_value - baseline_value
        }
    
    return result


def export_insights_to_json(insight):
    """
    Export a MetricInsight object to JSON.
    
    Args:
        insight: MetricInsight object
        
    Returns:
        JSON string representation
    """
    class InsightEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, '__dict__'):
                # For dataclasses or classes with __dict__
                result = {}
                for key, value in obj.__dict__.items():
                    # Convert snake_case to camelCase
                    camel_key = ''.join(word.capitalize() if i > 0 else word 
                                       for i, word in enumerate(key.split('_')))
                    result[camel_key] = value
                return result
            elif isinstance(obj, Enum):
                return obj.name
            return super().default(obj)
    
    return json.dumps(insight, cls=InsightEncoder, indent=2)


def import_insights_from_json(json_str):
    """
    Import a MetricInsight object from JSON.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        MetricInsight object
    """
    data = json.loads(json_str)
    
    # Helper to convert camelCase back to snake_case
    def to_snake_case(name):
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    # Convert the main object
    insight = MetricInsight()
    for key, value in data.items():
        snake_key = to_snake_case(key)
        if snake_key in insight.__dict__:
            # Special handling for nested objects
            if snake_key == 'dimensions':
                dimensions = {}
                for dim_name, dim_data in value.items():
                    dimensions[dim_name] = Dimension(
                        name=dim_data.get('name', dim_name),
                        score=dim_data.get('score', 0),
                        is_key_dimension=dim_data.get('isKeyDimension', False),
                        values=set(dim_data.get('values', []))
                    )
                setattr(insight, snake_key, dimensions)
            elif snake_key == 'dimension_slice_info':
                segments = {}
                for seg_key, seg_data in value.items():
                    # Create dimension value pairs
                    key_parts = []
                    for pair_data in seg_data.get('key', []):
                        key_parts.append(DimensionValuePair(
                            dimension=pair_data.get('dimension', ''),
                            value=pair_data.get('value', '')
                        ))
                    
                    # Create period values
                    baseline = seg_data.get('baselineValue', {})
                    comparison = seg_data.get('comparisonValue', {})
                    
                    baseline_value = PeriodValue(
                        slice_count=baseline.get('sliceCount', 0),
                        slice_size=baseline.get('sliceSize', 0),
                        slice_value=baseline.get('sliceValue', 0)
                    )
                    
                    comparison_value = PeriodValue(
                        slice_count=comparison.get('sliceCount', 0),
                        slice_size=comparison.get('sliceSize', 0),
                        slice_value=comparison.get('sliceValue', 0)
                    )
                    
                    # Create segment info
                    segments[seg_key] = SegmentInfo(
                        key=tuple(key_parts),
                        serialized_key=seg_key,
                        baseline_value=baseline_value,
                        comparison_value=comparison_value,
                        impact=seg_data.get('impact', 0),
                        change_percentage=seg_data.get('changePercentage', 0),
                        change_dev=seg_data.get('changeDev'),
                        absolute_contribution=seg_data.get('absoluteContribution'),
                        confidence=seg_data.get('confidence'),
                        sort_value=seg_data.get('sortValue')
                    )
                
                setattr(insight, snake_key, segments)
            elif snake_key in ['baseline_value_by_date', 'comparison_value_by_date']:
                value_list = []
                for item in value:
                    value_list.append(ValueByDate(
                        date=item.get('date', ''),
                        value=item.get('value', 0)
                    ))
                setattr(insight, snake_key, value_list)
            else:
                setattr(insight, snake_key, value)
    
    return insight


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Sample data
    data = {
        'date': pd.date_range(start='2023-01-01', end='2023-02-28', freq='D'),
        'country': ['US', 'UK', 'CA', 'AU'] * 15,
        'device': ['desktop', 'mobile', 'tablet'] * 20,
        'revenue': [100 + i * 0.5 for i in range(59)]
    }
    
    df = pd.DataFrame(data)
    
    # Analyze key drivers
    insight = analyze_key_drivers(
        df,
        baseline_date_range=('2023-01-01', '2023-01-31'),
        comparison_date_range=('2023-02-01', '2023-02-28'),
        group_by_columns=['country', 'device'],
        metric_column='revenue',
        agg_method='sum',
        date_column='date'
    )
    
    # Export to JSON
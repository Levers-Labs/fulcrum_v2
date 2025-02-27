"""
Base Pattern Class

This module defines the base Pattern class that all analysis patterns should inherit from.
It provides standard structure, error handling, and validation functionality.
"""

from typing import Dict, Any, Optional
import pandas as pd
from fulcrum_app.intelligence_engine.data_structures import PatternOutput

class Pattern:
    """
    Base class for all analysis patterns.
    
    All patterns should inherit from this class and implement the `run` method.
    """
    
    PATTERN_NAME = "base_pattern"
    PATTERN_VERSION = "1.0"
    
    def run(self, 
            metric_id: str, 
            data: pd.DataFrame, 
            analysis_window: Dict[str, str], 
            **kwargs) -> PatternOutput:
        """
        Execute the pattern analysis and return a standardized PatternOutput.
        
        Parameters
        ----------
        metric_id : str
            The ID of the metric being analyzed
        data : pd.DataFrame
            DataFrame containing the metric data
        analysis_window : Dict[str, str]
            Dictionary specifying the analysis time window with keys:
            - 'start_date': Start date of analysis in 'YYYY-MM-DD' format
            - 'end_date': End date of analysis in 'YYYY-MM-DD' format
            - 'grain': Time grain ('day', 'week', 'month')
        **kwargs
            Additional pattern-specific parameters
            
        Returns
        -------
        PatternOutput
            Standardized output object with pattern name, version, metric ID, 
            analysis window, and results dictionary
        """
        # This method should be overridden by subclasses
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results={}
        )
    
    def validate_data(self, data: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that the input DataFrame contains all required columns.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to validate
        required_columns : list
            List of column names that must be present
            
        Returns
        -------
        bool
            True if all required columns are present
            
        Raises
        ------
        ValueError
            If any required column is missing
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True
    
    def validate_analysis_window(self, analysis_window: Dict[str, str]) -> bool:
        """
        Validate that the analysis window contains required fields.
        
        Parameters
        ----------
        analysis_window : Dict[str, str]
            Analysis window dictionary
            
        Returns
        -------
        bool
            True if analysis window is valid
            
        Raises
        ------
        ValueError
            If required fields are missing or invalid
        """
        required_fields = ['start_date', 'end_date']
        missing_fields = [field for field in required_fields if field not in analysis_window]
        if missing_fields:
            raise ValueError(f"Missing required fields in analysis_window: {missing_fields}")
        
        # Validate grain if present
        if 'grain' in analysis_window:
            grain = analysis_window['grain'].lower()
            valid_grains = ['day', 'week', 'month']
            if grain not in valid_grains:
                raise ValueError(f"Invalid grain: {grain}. Must be one of {valid_grains}")
            
        return True
        
    def handle_empty_data(self, metric_id: str, analysis_window: Dict[str, str]) -> PatternOutput:
        """
        Create a standardized PatternOutput for empty or insufficient data.
        
        Parameters
        ----------
        metric_id : str
            The metric ID
        analysis_window : Dict[str, str]
            Analysis window dictionary
            
        Returns
        -------
        PatternOutput
            Empty PatternOutput with an error message
        """
        return PatternOutput(
            pattern_name=self.PATTERN_NAME,
            pattern_version=self.PATTERN_VERSION,
            metric_id=metric_id,
            analysis_window=analysis_window,
            results={"error": "Insufficient data for analysis"}
        )
        
    def extract_date_range(self, data: pd.DataFrame, date_col: str = 'date') -> Dict[str, str]:
        """
        Extract the minimum and maximum dates from the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data
        date_col : str, default 'date'
            Column name containing dates
            
        Returns
        -------
        Dict[str, str]
            Dictionary with 'start_date' and 'end_date' in 'YYYY-MM-DD' format
        """
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
            
        dates = pd.to_datetime(data[date_col])
        return {
            'start_date': dates.min().strftime('%Y-%m-%d'),
            'end_date': dates.max().strftime('%Y-%m-%d')
        }
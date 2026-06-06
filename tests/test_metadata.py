"""Tests for metadata functionality."""
import pytest

import pyita as ta
from pyita.metadata import create_metadata


class TestMetadata:
    """Tests for metadata functionality."""
    
    def test_metadata(self):
        """Test that metadata() can be called without errors."""
        create_metadata()
        result = ta.metadata()
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        for indicator_name, metadata in result.items():
            assert 'name' in metadata
            assert 'signature' in metadata
            assert 'parameters' in metadata
            assert 'output_series' in metadata
            assert 'description' in metadata
            assert metadata['name'] == indicator_name
            
            # Check output_series structure and types
            for series in metadata['output_series']:
                assert 'name' in series
                assert 'type' in series
                assert 'range' in series
                assert isinstance(series['name'], str)
                assert isinstance(series['type'], str)
                assert series['type'] in ('price', 'as_source', 'none'), \
                    f"Invalid series type '{series['type']}' for {indicator_name}.{series['name']}"
                assert series['range'] is None \
                    or series['range'] == 'as_source' \
                    or isinstance(series['range'], dict), \
                    f"Invalid range '{series['range']}' for {indicator_name}.{series['name']}"
                if isinstance(series['range'], dict):
                    assert set(series['range'].keys()) == {'min', 'max'}
                    assert isinstance(series['range']['min'], (int, float))
                    assert isinstance(series['range']['max'], (int, float))
                    assert series['range']['min'] <= series['range']['max']

    def test_metadata_ranges(self):
        """Test selected output series ranges."""
        create_metadata()
        result = ta.metadata()

        def series_range(indicator_name, series_name):
            output_series = result[indicator_name]['output_series']
            for series in output_series:
                if series['name'] == series_name:
                    return series['range']
            raise AssertionError(f"Series {indicator_name}.{series_name} not found")

        assert series_range('rsi', 'rsi') == {'min': 0, 'max': 100}
        assert series_range('mfi', 'mfi') == {'min': 0, 'max': 100}
        assert series_range('williams_r', 'williams_r') == {'min': -100, 'max': 0}
        assert series_range('aroon', 'up') == {'min': 0, 'max': 100}
        assert series_range('aroon', 'down') == {'min': 0, 'max': 100}
        assert series_range('aroon', 'oscillator') == {'min': -100, 'max': 100}
        assert series_range('stochastic', 'oscillator') == {'min': 0, 'max': 100}
        assert series_range('stochastic', 'value_k') == {'min': 0, 'max': 100}
        assert series_range('stochastic', 'value_d') == {'min': 0, 'max': 100}
        assert series_range('parabolic_sar', 'signal') == {'min': -1, 'max': 1}
        assert series_range('zigzag', 'pivot_types') == {'min': -1, 'max': 1}
        assert series_range('sma', 'sma') == 'as_source'
        assert series_range('ema', 'ema') == 'as_source'
        assert series_range('ma', 'move_average') == 'as_source'
        assert series_range('macd', 'macd') is None

    def test_version(self):
        """Test that __version__ is accessible and is a valid version string."""
        assert hasattr(ta, '__version__')
        version = ta.__version__
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should be in format like "1.0.13" or similar
        assert '.' in version



"""Tests for DataSeries slicing functionality."""
import numpy as np
import pytest

import pyita as ta
from pyita.exceptions import PyTAExceptionDataSeriesNonFound
from pyita.indicator_result import IndicatorResult


@pytest.fixture
def sample_quotes():
    """Create sample Quotes object for testing."""
    return ta.Quotes(
        open=np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]),
        high=np.array([105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]),
        low=np.array([99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]),
        close=np.array([102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]),
        volume=np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]),
    )


@pytest.fixture
def sample_result():
    """Create sample IndicatorResult object for testing."""
    return IndicatorResult({
        'ema': np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        'sma': np.array([100.5, 101.5, 102.5, 103.5, 104.5]),
        'rsi': np.array([50.0, 51.0, 52.0, 53.0, 54.0]),
    })


class TestQuotesSlicing:
    """Tests for Quotes slicing with slice objects."""
    
    def test_simple_slice(self, sample_quotes):
        """Test simple slice quotes[1:10]."""
        sliced = sample_quotes[1:10]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 9
        np.testing.assert_array_equal(sliced.close, sample_quotes.close[1:10])
        np.testing.assert_array_equal(sliced.open, sample_quotes.open[1:10])
        np.testing.assert_array_equal(sliced.volume, sample_quotes.volume[1:10])
    
    def test_slice_from_start(self, sample_quotes):
        """Test slice quotes[:10]."""
        sliced = sample_quotes[:10]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 10
        np.testing.assert_array_equal(sliced.close, sample_quotes.close[:10])
    
    def test_slice_to_end(self, sample_quotes):
        """Test slice quotes[5:]."""
        sliced = sample_quotes[5:]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 5
        np.testing.assert_array_equal(sliced.close, sample_quotes.close[5:])
    
    def test_slice_with_step(self, sample_quotes):
        """Test slice quotes[::2]."""
        sliced = sample_quotes[::2]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 5
        np.testing.assert_array_equal(sliced.close, sample_quotes.close[::2])
    
    def test_slice_with_start_end_step(self, sample_quotes):
        """Test slice quotes[1:10:2]."""
        sliced = sample_quotes[1:10:2]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 5
        np.testing.assert_array_equal(sliced.close, sample_quotes.close[1:10:2])
    
    def test_full_slice(self, sample_quotes):
        """Test slice quotes[:]."""
        sliced = sample_quotes[:]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == len(sample_quotes.close)
        np.testing.assert_array_equal(sliced.close, sample_quotes.close)
        assert sliced is not sample_quotes


class TestQuotesIndexing:
    """Tests for Quotes indexing with int."""
    
    def test_first_index(self, sample_quotes):
        """Test quotes[0]."""
        sliced = sample_quotes[0]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 1
        assert sliced.close[0] == sample_quotes.close[0]
        assert sliced.open[0] == sample_quotes.open[0]
    
    def test_middle_index(self, sample_quotes):
        """Test quotes[5]."""
        sliced = sample_quotes[5]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 1
        assert sliced.close[0] == sample_quotes.close[5]
    
    def test_negative_index(self, sample_quotes):
        """Test quotes[-1]."""
        sliced = sample_quotes[-1]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 1
        assert sliced.close[0] == sample_quotes.close[-1]
    
    def test_negative_middle_index(self, sample_quotes):
        """Test quotes[-5]."""
        sliced = sample_quotes[-5]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 1
        assert sliced.close[0] == sample_quotes.close[-5]


class TestQuotesIndexErrors:
    """Tests for Quotes indexing error handling."""
    
    def test_index_out_of_range_positive(self, sample_quotes):
        """Test quotes[1000] raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_quotes[1000]
    
    def test_index_out_of_range_negative(self, sample_quotes):
        """Test quotes[-1000] raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_quotes[-1000]


class TestQuotesEmptySlices:
    """Tests for Quotes empty slice behavior."""
    
    def test_empty_slice_reversed(self, sample_quotes):
        """Test quotes[10:5] returns empty Quotes."""
        sliced = sample_quotes[10:5]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 0
        assert len(sliced.open) == 0
    
    def test_empty_slice_out_of_range(self, sample_quotes):
        """Test quotes[100:200] returns empty Quotes."""
        sliced = sample_quotes[100:200]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 0
    
    def test_empty_slice_zero_length(self, sample_quotes):
        """Test quotes[5:5] returns empty Quotes."""
        sliced = sample_quotes[5:5]
        
        assert isinstance(sliced, ta.Quotes)
        assert len(sliced.close) == 0


class TestQuotesColumnAccess:
    """Tests for Quotes column access (old behavior)."""
    
    def test_column_access_string(self, sample_quotes):
        """Test quotes['close'] returns array."""
        result = sample_quotes['close']
        
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, ta.Quotes)
        np.testing.assert_array_equal(result, sample_quotes.close)
    
    def test_column_access_open(self, sample_quotes):
        """Test quotes['open'] returns array."""
        result = sample_quotes['open']
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, sample_quotes.open)
    
    def test_column_access_nonexistent(self, sample_quotes):
        """Test quotes['nonexistent'] raises exception."""
        with pytest.raises(PyTAExceptionDataSeriesNonFound):
            _ = sample_quotes['nonexistent']


class TestQuotesSlicedDataIntegrity:
    """Tests for Quotes sliced data integrity."""
    
    def test_all_columns_present(self, sample_quotes):
        """Test all columns are present in sliced object."""
        sliced = sample_quotes[1:5]
        
        assert 'open' in sliced._data
        assert 'high' in sliced._data
        assert 'low' in sliced._data
        assert 'close' in sliced._data
        assert 'volume' in sliced._data
    
    def test_all_columns_same_length(self, sample_quotes):
        """Test all columns have same length in sliced object."""
        sliced = sample_quotes[1:5]
        
        lengths = [len(sliced._data[col]) for col in sliced._data.keys()]
        assert all(length == lengths[0] for length in lengths)
        assert lengths[0] == 4
    
    def test_optional_columns_preserved(self, sample_quotes):
        """Test optional columns (volume) are preserved."""
        sliced = sample_quotes[2:7]
        
        assert hasattr(sliced, 'volume')
        np.testing.assert_array_equal(sliced.volume, sample_quotes.volume[2:7])


class TestIndicatorResultSlicing:
    """Tests for IndicatorResult slicing."""
    
    def test_result_slice(self, sample_result):
        """Test result[1:4]."""
        sliced = sample_result[1:4]
        
        assert isinstance(sliced, IndicatorResult)
        assert len(sliced.ema) == 3
        np.testing.assert_array_equal(sliced.ema, sample_result.ema[1:4])
        np.testing.assert_array_equal(sliced.sma, sample_result.sma[1:4])
    
    def test_result_index(self, sample_result):
        """Test result[2]."""
        sliced = sample_result[2]
        
        assert isinstance(sliced, IndicatorResult)
        assert len(sliced.ema) == 1
        assert sliced.ema[0] == sample_result.ema[2]
    
    def test_result_all_columns(self, sample_result):
        """Test all columns are present in sliced result."""
        sliced = sample_result[1:3]
        
        assert 'ema' in sliced._data
        assert 'sma' in sliced._data
        assert 'rsi' in sliced._data
        assert len(sliced.ema) == 2
        assert len(sliced.sma) == 2
        assert len(sliced.rsi) == 2


class TestChainedSlicing:
    """Tests for chained slicing operations."""
    
    def test_slice_then_column(self, sample_quotes):
        """Test quotes[1:10]['close']."""
        sliced = sample_quotes[1:10]
        column = sliced['close']
        
        assert isinstance(column, np.ndarray)
        np.testing.assert_array_equal(column, sample_quotes.close[1:10])
    
    def test_double_slice(self, sample_quotes):
        """Test quotes[::2][:5]."""
        first_slice = sample_quotes[::2]
        second_slice = first_slice[:5]
        
        assert isinstance(second_slice, ta.Quotes)
        assert len(second_slice.close) == 5
        np.testing.assert_array_equal(second_slice.close, sample_quotes.close[::2][:5])


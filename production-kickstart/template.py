"""
Advanced Data Processing and Analysis Module

This module demonstrates expert-level Python coding practices, including:
- Type hinting and runtime type checking
- Comprehensive error handling and custom exceptions
- Asynchronous programming
- Advanced OOP concepts (ABCs, dataclasses)
- Functional programming techniques
- Comprehensive logging
- Performance optimization
- Testing setup (doctests, unit tests, property-based tests)

Requires Python 3.9+
"""

from __future__ import annotations

import asyncio
import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

import aiohttp
from hypothesis import given, strategies as st
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Custom exception hierarchy
class DataProcessingError(Exception):
    """Base exception for all data processing errors."""


class DataFetchError(DataProcessingError):
    """Raised when data fetching fails."""


class DataValidationError(DataProcessingError):
    """Raised when data validation fails."""


# Type definitions
T = TypeVar('T')
ProcessorFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@runtime_checkable
class DataSource(Protocol):
    """Protocol defining the interface for data sources."""

    async def fetch_data(self) -> Dict[str, Any]:
        """Fetch data from the source."""


@dataclass(frozen=True)
class DataPoint:
    """Immutable data point class."""

    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data after initialization."""
        if self.value < 0:
            raise DataValidationError(f"Value must be non-negative: {self.value}")


class DataModel(BaseModel):
    """Pydantic model for data validation."""

    id: str
    name: str
    points: List[DataPoint]

    @validator('points')
    def check_points(cls, v):
        """Validate that there are at least two data points."""
        if len(v) < 2:
            raise DataValidationError("At least two data points are required")
        return v


class DataProcessor(ABC):
    """Abstract base class for data processors."""

    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data."""


class AdvancedDataProcessor(DataProcessor):
    """Advanced data processor implementation."""

    def __init__(self, processors: List[ProcessorFunc]):
        self.processors = processors

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using all registered processor functions.

        Args:
            data: Input data dictionary.

        Returns:
            Processed data dictionary.

        Raises:
            DataProcessingError: If any processor function fails.
        """
        try:
            for processor in self.processors:
                data = processor(data)
            return data
        except Exception as e:
            logger.exception("Error during data processing")
            raise DataProcessingError(f"Processing failed: {str(e)}") from e


class APIDataSource(DataSource):
    """API-based data source implementation."""

    def __init__(self, url: str):
        self.url = url

    async def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch data from the API.

        Returns:
            Dict containing the fetched data.

        Raises:
            DataFetchError: If the API request fails.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch data from {self.url}: {str(e)}")
            raise DataFetchError(f"Failed to fetch data: {str(e)}") from e


@functools.lru_cache(maxsize=None)
def get_cached_data(key: str) -> Any:
    """
    Retrieve cached data (placeholder implementation).

    Args:
        key: Cache key.

    Returns:
        Cached data (if available).
    """
    # Placeholder implementation
    return None


async def process_data_stream(data_source: DataSource, processor: DataProcessor) -> List[DataModel]:
    """
    Process a stream of data from the given source.

    Args:
        data_source: Source of the data stream.
        processor: Data processor to use.

    Returns:
        List of processed DataModel instances.

    Raises:
        DataProcessingError: If data processing fails.
    """
    try:
        raw_data = await data_source.fetch_data()
        processed_data = await processor.process(raw_data)
        return [DataModel(**item) for item in processed_data['items']]
    except (DataFetchError, DataProcessingError) as e:
        logger.error(f"Failed to process data stream: {str(e)}")
        raise
    except Exception as e:
        logger.exception("Unexpected error during data stream processing")
        raise DataProcessingError(f"Unexpected error: {str(e)}") from e


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Set up data source and processor
        data_source = APIDataSource("https://api.example.com/data")
        processor = AdvancedDataProcessor([
            lambda d: {**d, 'processed': True},
            lambda d: {**d, 'items': [{**item, 'value': item['value'] * 2} for item in d['items']]}
        ])

        # Process data
        try:
            results = await process_data_stream(data_source, processor)
            logger.info(f"Processed {len(results)} data models")
        except DataProcessingError as e:
            logger.error(f"Data processing failed: {str(e)}")


    # Run the main coroutine
    asyncio.run(main())


    # Doctests
    def example_processor(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example processor function for demonstration and testing.

        >>> example_processor({'value': 5})
        {'value': 10}
        >>> example_processor({'value': -5})
        Traceback (most recent call last):
            ...
        DataValidationError: Value must be non-negative: -10
        """
        result = {'value': data['value'] * 2}
        DataPoint(timestamp=0, value=result['value'])  # Validate using DataPoint
        return result


    import doctest

    doctest.testmod()


    # Property-based testing
    @given(st.dictionaries(keys=st.text(), values=st.floats(min_value=0)))
    def test_example_processor_properties(data):
        """Test properties of the example processor using hypothesis."""
        result = example_processor(data)
        assert result['value'] == data['value'] * 2


    test_example_processor_properties()

# NOTE: In a real-world scenario, you would typically split this code into multiple
# files (e.g., models.py, processors.py, data_sources.py, etc.) and have separate
# test files. This combined example is for demonstration purposes only.
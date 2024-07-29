# Python Best Practices Compendium

# 1. Code Style and Organization
import this  # The Zen of Python

# 2. Type Hinting
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 3. Docstrings
def calculate_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.

    Args:
        length (float): The length of the rectangle.
        width (float): The width of the rectangle.

    Returns:
        float: The area of the rectangle.
    """
    return length * width

# 4. Exception Handling
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")

# 5. Context Managers
with open('example.txt', 'w') as f:
    f.write('Hello, World!')

# 6. List Comprehensions
squares = [x**2 for x in range(10)]

# 7. Generator Expressions
sum_of_squares = sum(x**2 for x in range(10))

# 8. Decorators
from functools import wraps
import time

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)

# 9. Property Decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# 10. Magic Methods
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

# 11. Iterators and Generators
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 12. Async Programming
import asyncio

async def fetch_data(url):
    # Simulating an API call
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    urls = ['http://example.com', 'http://example.org', 'http://example.net']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

# 13. Functional Programming
from functools import reduce
numbers = [1, 2, 3, 4, 5]
sum_of_numbers = reduce(lambda x, y: x + y, numbers)

# 14. Enum Usage
from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

# 15. Dataclasses
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0

# 16. Type Checking
from typing import List, Dict, Optional

def process_data(data: List[Dict[str, str]]) -> Optional[str]:
    if not data:
        return None
    return ", ".join(item['name'] for item in data)

# 17. Logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        logger.error("Division by zero!")
        return None
    else:
        logger.info(f"Division result: {result}")
        return result

# 18. Configuration Management
import configparser

config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}
with open('config.ini', 'w') as configfile:
    config.write(configfile)

# 19. Unit Testing
import unittest

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

# 20. Performance Profiling
import cProfile

def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

cProfile.run('fibonacci(30)')

# 21. Multiprocessing
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

# 22. Type Checking with mypy
# Run: mypy your_script.py
x: int = 1
y: str = "2"
# z: int = x + y  # This would raise a type error

# 23. Virtual Environments
# python -m venv myenv
# source myenv/bin/activate (Linux/Mac)
# myenv\Scripts\activate (Windows)

# 24. Package Management
# pip install package_name
# pip freeze > requirements.txt

# 25. Code Formatting with Black
# pip install black
# black your_script.py

# 26. Linting with Pylint
# pip install pylint
# pylint your_script.py

# 27. Type Checking with mypy
# pip install mypy
# mypy your_script.py

# 28. Documentation with Sphinx
# pip install sphinx
# sphinx-quickstart

# 29. Continuous Integration
# .github/workflows/python-app.yml
# name: Python application
#
# on:
#   push:
#     branches: [ main ]
#   pull_request:
#     branches: [ main ]
#
# jobs:
#   build:
#     runs-on: ubuntu-latest
#     steps:
#     - uses: actions/checkout@v2
#     - name: Set up Python
#       uses: actions/setup-python@v2
#       with:
#         python-version: 3.8
#     - name: Install dependencies
#       run: |
#         python -m pip install --upgrade pip
#         pip install -r requirements.txt
#     - name: Run tests
#       run: |
#         python -m unittest discover tests

# 30. Code Coverage
# pip install coverage
# coverage run -m unittest discover
# coverage report -m

# ... (Additional best practices would follow a similar pattern)
# Advanced Python Best Practices

# 31. Metaclasses
class Meta(type):
    def __new__(cls, name, bases, attrs):
        # Add a new method to the class
        attrs['custom_method'] = lambda self: print(f"I'm {self.__class__.__name__}")
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
    pass

obj = MyClass()
obj.custom_method()  # Outputs: I'm MyClass

# 32. Abstract Base Classes
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

# 33. Context Managers with contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    try:
        f = open(filename, mode)
        yield f
    finally:
        f.close()

with file_manager('test.txt', 'w') as f:
    f.write('Hello, World!')

# 34. Descriptors
class Validator:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return getattr(obj, f"_{self.name}")

    def __set__(self, obj, value):
        if not self.min_value <= value <= self.max_value:
            raise ValueError(f"{self.name} must be between {self.min_value} and {self.max_value}")
        setattr(obj, f"_{self.name}", value)

class Person:
    age = Validator(0, 150)

# 35. Memory-efficient coding with __slots__
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 36. Lazy property evaluation
class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__

    def __get__(self, obj, type=None) -> object:
        if obj is None:
            return self
        value = self.function(obj)
        setattr(obj, self.name, value)
        return value

class MyClass:
    @LazyProperty
    def expensive_computation(self):
        # Simulating an expensive operation
        import time
        time.sleep(2)
        return 42

# 37. Type checking with Protocol
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing a circle")

def draw_shape(shape: Drawable) -> None:
    shape.draw()

# 38. Structural Pattern Matching (Python 3.10+)
def parse_command(command):
    match command.split():
        case ["quit"]:
            return "Exiting program"
        case ["load", filename]:
            return f"Loading file: {filename}"
        case ["save", filename]:
            return f"Saving file: {filename}"
        case _:
            return "Unknown command"

# 39. Asynchronous context managers
import asyncio

class AsyncResource:
    async def __aenter__(self):
        await asyncio.sleep(1)  # Simulate resource acquisition
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await asyncio.sleep(1)  # Simulate resource release

async def use_resource():
    async with AsyncResource() as resource:
        print("Resource is ready")

# 40. Type-safe enumerations
from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

    @classmethod
    def from_string(cls, color_string: str) -> 'Color':
        try:
            return cls[color_string.upper()]
        except KeyError:
            raise ValueError(f"'{color_string}' is not a valid Color")

# 41. Functools for improved performance
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 42. Type-safe function overloading
from typing import overload, Union

@overload
def process(x: int) -> int: ...

@overload
def process(x: str) -> str: ...

def process(x: Union[int, str]) -> Union[int, str]:
    if isinstance(x, int):
        return x * 2
    elif isinstance(x, str):
        return x.upper()
    else:
        raise TypeError("Unsupported type")

# 43. Customizing class creation
class CustomMeta(type):
    def __new__(cls, name, bases, dct):
        # Convert all method names to uppercase
        uppercase_attr = {
            key.upper(): value
            for key, value in dct.items()
            if not key.startswith("__")
        }
        return super().__new__(cls, name, bases, uppercase_attr)

class MyClass(metaclass=CustomMeta):
    def hello(self):
        return "Hello, World!"

obj = MyClass()
print(obj.HELLO())  # Outputs: Hello, World!

# 44. Advanced logging with structlog
import structlog

logger = structlog.get_logger()
logger.info("user logged in", user="john_doe", status="success")

# 45. Dependency injection
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self, engine: Engine):
        self.engine = engine

    def start(self):
        return self.engine.start()

# Usage
engine = Engine()
car = Car(engine)
print(car.start())  # Outputs: Engine started

# ... (Additional advanced practices would follow a similar pattern)
# Specialized Python Best Practices

# 46. Concurrent Programming with asyncio
import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)  # Simulating network I/O
    return f"Data from {url}"

async def main():
    urls = ['http://example.com', 'http://example.org', 'http://example.net']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())

# 47. High-performance Computing with Numba
from numba import jit
import numpy as np

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

# 48. Memory-efficient Data Processing with Generators
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip().upper()

# Usage
for processed_line in process_large_file('large_file.txt'):
    print(processed_line)

# 49. Type Checking with Protocols
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing a circle")

def draw_shape(shape: Drawable) -> None:
    shape.draw()

# 50. Reactive Programming with RxPY
from rx import of, operators as ops

of(1, 2, 3, 4, 5).pipe(
    ops.map(lambda x: x * 2),
    ops.filter(lambda x: x > 5)
).subscribe(print)

# 51. Aspect-Oriented Programming
from aspectlib import Aspect, Proceed, Return

@Aspect
def log_calls(function):
    def wrapper(*args, **kwargs):
        print(f"Calling {function.__name__}")
        result = Proceed(*args, **kwargs)
        print(f"{function.__name__} returned {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

# 52. Design by Contract
from icontract import require, ensure, invariant

@require(lambda x: x > 0)
@ensure(lambda result: result > 0)
def square_root(x):
    return x ** 0.5

# 53. Property-Based Testing
from hypothesis import given
from hypothesis.strategies import integers

@given(integers(), integers())
def test_addition_commutativity(a, b):
    assert a + b == b + a

# 54. Automatic Code Formatting with Black in Pre-commit Hooks
# In .pre-commit-config.yaml:
# repos:
# -   repo: https://github.com/psf/black
#     rev: 22.3.0
#     hooks:
#     - id: black
#       language_version: python3.9

# 55. Static Type Checking with Pyright
# In pyrightconfig.json:
# {
#     "include": [
#         "src"
#     ],
#     "exclude": [
#         "**/node_modules",
#         "**/__pycache__"
#     ],
#     "ignore": [
#         "src/oldstuff"
#     ],
#     "defineConstant": {
#         "DEBUG": true
#     },
#     "stubPath": "src/stubs",
#     "venv": "env"
# }

# 56. Micro-Optimization with Cython
# In my_module.pyx:
# def fast_function(int x, int y):
#     cdef int result = x + y
#     return result

# 57. Distributed Computing with Dask
import dask.dataframe as dd

df = dd.read_csv('*.csv')
result = df.groupby('column').mean().compute()

# 58. Web Scraping with Scrapy
# In my_spider.py:
# import scrapy
#
# class MySpider(scrapy.Spider):
#     name = 'myspider'
#     start_urls = ['http://example.com']
#
#     def parse(self, response):
#         yield {'title': response.css('h1::text').get()}

# 59. GUI Development with PyQt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
import sys

app = QApplication(sys.argv)
window = QWidget()
button = QPushButton('Click me', window)
window.show()
sys.exit(app.exec_())

# 60. Machine Learning Model Deployment
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')

# Load model
loaded_model = joblib.load('model.joblib')

# Use model
predictions = loaded_model.predict(X_test)

# ... (Additional specialized practices would follow a similar pattern)
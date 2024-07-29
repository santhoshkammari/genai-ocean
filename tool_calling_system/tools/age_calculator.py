import time


def calculate_age(birth_year: int) -> int:
    """Calculate the age of a person based on their birth year."""
    current_year = time.localtime().tm_year
    return current_year - int(birth_year)
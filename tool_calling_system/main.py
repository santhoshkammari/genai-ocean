# main.py
import time

import requests
from arxiv import arxiv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from hermes_thoth import HermesThoth
from tools import EnhancedYouTubeSearchTool, calculate_age
from tool_calling_chain import ToolCallingChain
# wikipedia_query_run = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())
# Existing tools
youtube_search = EnhancedYouTubeSearchTool()

def calculate_age(birth_year: int) -> int:
    """Calculate the age of a person based on their birth year."""
    current_year = time.localtime().tm_year
    return current_year - int(birth_year)

# New tools
class WeatherTool:
    def get_weather(self, location: str) -> str:
        """Get the weather for a given location."""
        # Simulated weather data
        return f"The weather in {location} is sunny with a high of 25°C."

class CurrencyConverterTool:
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert currency from one type to another."""
        # Simulated conversion rates
        rates = {"USD": 1, "EUR": 0.85, "GBP": 0.73, "JPY": 110}
        return amount * (rates[to_currency] / rates[from_currency])

class StockPriceTool:
    def get_stock_price(self, symbol: str) -> float:
        """Get the current stock price for a given symbol."""
        # Simulated stock price
        return 150.25

class RandomFactTool:
    def get_random_fact(self) -> str:
        """Get a random fact."""
        return "The Great Wall of China is not visible from space with the naked eye."

class QuoteTool:
    def get_quote(self) -> str:
        """Get a random quote."""
        return "Be the change you wish to see in the world. - Mahatma Gandhi"

class MathsCalculatorTool:
    def maths_calculate(self, expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression)

class DateTimeTool:
    def get_current_datetime(self) -> str:
        """Get the current date and time."""
        return time.strftime("%Y-%m-%d %H:%M:%S")

class MovieInfoTool:
    def get_movie_info(self, title: str) -> str:
        """Get information about a movie."""
        return f"Movie: {title}\nRelease Year: 2023\nDirector: John Doe\nRating: 8.5/10"

class NewsHeadlineTool:
    def get_top_headline(self) -> str:
        """Get the top news headline."""
        return "Scientists Make Breakthrough in Renewable Energy Technology"


class WikipediaSearchTool:
    def search_wikipedia(self, query: str) -> str:
        """
        Search Wikipedia and return a summary of the top result.
        """
        limit: int = 1
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit={limit}&namespace=0&format=json"
        response = requests.get(search_url)
        data = response.json()

        if not data[1]:  # If no results found
            return f"No Wikipedia results found for '{query}'."

        results = []
        for i in range(min(limit, len(data[1]))):
            title = data[1][i]
            url = data[3][i]
            page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={title}&format=json"
            page_response = requests.get(page_url)
            page_data = page_response.json()
            page_id = list(page_data['query']['pages'].keys())[0]
            summary = page_data['query']['pages'][page_id]['extract']
            results.append(f"Title: {title}\nSummary: {summary[:500]}...\nURL: {url}")

        return "\n\n".join(results)


class ArxivSearchTool:
    def search_arxiv(self, query: str) -> str:
        """Computer Science research tool, useful when you need papers from arxiv website."""
        limit: int = 1
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        for paper in search.results():
            results.append(
                f"Title: {paper.title}\nAuthors: {', '.join(author.name for author in paper.authors)}\nSummary: {paper.summary[:500]}...\nURL: {paper.pdf_url}")

        if not results:
            return f"No arXiv results found for '{query}'."

        return "\n\n".join(results)

# Initialize the CustomChatOllama and ToolCallingChain
llm = HermesThoth(
    # host_type="hf",
    # api_url = "http://192.168.162.147:8000",
    # model = "meta-llama/Meta-Llama-3-8B-Instruct",#"Amu/supertiny-llama3-0.25B-v0.1"
    model="qwen2:0.5b",
    temperature=1e-3,
    max_new_tokens=100,
    torch_dtype="float16"
).bind_tools([
    calculate_age,
    WeatherTool().get_weather,
    CurrencyConverterTool().convert_currency,
    StockPriceTool().get_stock_price,
    RandomFactTool().get_random_fact,
    QuoteTool().get_quote,
    MathsCalculatorTool().maths_calculate,
    DateTimeTool().get_current_datetime,
    MovieInfoTool().get_movie_info,

    ArxivSearchTool().search_arxiv,
    # WikipediaSearchTool().search_wikipedia,
# wikipedia_query_run,
    youtube_search

])

chain = ToolCallingChain(llm=llm, verbose=True)

# Test cases
test_inputs = [
    # "What's the weather like in London?",
    # "Convert 100 USD to EUR",
    # "Translate 'Hello, world!' to Spanish",
    # "What's the current stock price of AAPL?",
    # "Tell me a random fact",
    # "Give me an inspirational quote",
    # "Calculate 15 * 7 + 22",
    # "What's the current date and time?",
    # "Give me information about the movie 'Inception'",
    # "What's the top news headline today?",
    # "Search Wikipedia for 'Artificial Intelligence'",
    # "Find recent arXiv papers on 'quantum computing'",
    # "Give me 5 points about Jawaharlal Nehru you can use wikipedia",
    "Give me lisa rockstar official song link",
    # "covering schedule document in trade finance"
]

for user_input in test_inputs:
    print(f"\nTesting: {user_input}")
    start_time = time.perf_counter()
    # result = chain.invoke({"input": user_input})
    for res in chain.stream({"input": user_input}):
        print(res["output"],end="",flush=True)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

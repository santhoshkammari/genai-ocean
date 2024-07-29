import streamlit as st
import time
import requests
import arxiv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from hermes_thoth import HermesThoth
from tools import EnhancedYouTubeSearchTool
from tool_calling_chain import ToolCallingChain

# Initialize tools
wikipedia_query_run = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
youtube_search = EnhancedYouTubeSearchTool()

def calculate_age(birth_year: int) -> int:
    current_year = time.localtime().tm_year
    return current_year - int(birth_year)

class WeatherTool:
    def get_weather(self, location: str) -> str:
        return f"The weather in {location} is sunny with a high of 25°C."

class CurrencyConverterTool:
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        rates = {"USD": 1, "EUR": 0.85, "GBP": 0.73, "JPY": 110}
        return amount * (rates[to_currency] / rates[from_currency])

class TranslatorTool:
    def translate(self, text: str, target_language: str) -> str:
        return f"Translated '{text}' to {target_language}: [Translation here]"

class StockPriceTool:
    def get_stock_price(self, symbol: str) -> float:
        return 150.25

class RandomFactTool:
    def get_random_fact(self) -> str:
        return "The Great Wall of China is not visible from space with the naked eye."

class QuoteTool:
    def get_quote(self) -> str:
        return "Be the change you wish to see in the world. - Mahatma Gandhi"

class MathsCalculatorTool:
    def maths_calculate(self, expression: str) -> float:
        return eval(expression)


class MovieInfoTool:
    def get_movie_info(self, title: str) -> str:
        return f"Movie: {title}\nRelease Year: 2023\nDirector: John Doe\nRating: 8.5/10"

class ArxivSearchTool:
    def search_arxiv(self, query: str) -> str:
        """Computer Science research tool, useful when you need papers from arxiv website."""
        limit = 1
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
# Define all available tools
all_tools = {
    "Age Calculator": calculate_age,
    # "Weather": WeatherTool().get_weather,
    # "Currency Converter": CurrencyConverterTool().convert_currency,
    "Stock Price": StockPriceTool().get_stock_price,
    # "Random Fact": RandomFactTool().get_random_fact,
    # "Quote": QuoteTool().get_quote,
    # "Maths Calculator": MathsCalculatorTool().maths_calculate,
    # "Movie Info": MovieInfoTool().get_movie_info,
    "Arxiv Search": ArxivSearchTool().search_arxiv,
    "Wikipedia": wikipedia_query_run,
    "YouTube Search": youtube_search
}

# Initialize HermesThoth and ToolCallingChain
@st.cache_resource
def initialize_llm(_selected_tools,host_type='Llama-3-8B(Slow but High Accuracy)'):
    if host_type == "Llama-3-8B(Slow but High Accuracy)":
        generate_kwargs = dict(
            host_type="hf",
            # api_url = "http://192.168.162.147:8000",
            model="meta-llama/Meta-Llama-3-8B-Instruct",  # "Amu/supertiny-llama3-0.25B-v0.1"
            # model="qwen2:0.5b",
            temperature=1e-3,
            max_new_tokens=100,
            torch_dtype="float16"
        )
    elif host_type == "Qwen-2(Fast with Medium Accuracy)":
        generate_kwargs = dict(
            model="qwen2:0.5b",
            temperature=1e-3,
            max_new_tokens=100,
            torch_dtype="float16"
        )
    elif host_type == "Own Server (Local)":
        generate_kwargs = dict(
            host_type="hf",
            # api_url = "http://192.168.162.147:8000",
            model="meta-llama/Meta-Llama-3-8B-Instruct",  # "Amu/supertiny-llama3-0.25B-v0.1"
            # model="qwen2:0.5b",
            temperature=1e-3,
            max_new_tokens=100,
            torch_dtype="float16"
        )
    else:
        raise ValueError("Invalid host type")

    llm = HermesThoth(
        **generate_kwargs
    ).bind_tools(_selected_tools)
    return ToolCallingChain(llm=llm, verbose=True)
# Streamlit app
st.set_page_config(page_title="Hermes Thoth Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Hermes Thoth Assistant")
host_type = st.sidebar.selectbox(
    "Select Host Type",

        options=[
            "Llama-3-8B(Slow but High Accuracy)",
            "Qwen-2(Fast with Medium Accuracy)",
            "Own Server (Local)"
        ],
    help="Choose the host type for loading the model"
)

st.sidebar.header("Tool Selection")
use_all_tools = st.sidebar.checkbox("Use All Tools", value=True)

if use_all_tools:
    selected_tool_names = list(all_tools.keys())
else:
    selected_tool_names = st.sidebar.multiselect(
        "Select Tools to Use",
        options=list(all_tools.keys()),
        default=list(all_tools.keys())
    )

selected_tools = [all_tools[name] for name in selected_tool_names]

st.sidebar.header("About")
st.sidebar.info(
    "This app uses the Hermes Thoth model to provide answers and perform various tasks. "
    "You can ask questions, request information, or use any of the available tools."
)

# Initialize the chain with selected tools
chain = initialize_llm(selected_tools,host_type)

# Callback function to update input value
def update_input(question):
    st.session_state.user_input = question

# User input
user_input = st.text_input("Ask me anything:", key="user_input", value=st.session_state.get('user_input', ''))

if st.button("Submit"):
    if user_input:
        start_time = time.perf_counter()

        # Create a placeholder for the streaming output
        output_placeholder = st.empty()

        # Stream the response
        full_response = ""
        for chunk in chain.stream({"input": user_input}):
            full_response += chunk["output"]
            output_placeholder.markdown(full_response + "▌")

        # Update the placeholder with the final response
        output_placeholder.markdown(full_response)

        end_time = time.perf_counter()
        st.info(f"Time taken: {end_time - start_time:.4f} seconds")
    else:
        st.warning("Please enter a question or request.")

# Sample questions
st.sidebar.header("Sample Questions")
sample_questions = [
    "What's the weather like in London?",
    "Convert 100 USD to EUR",
    "Translate 'Hello, world!' to Spanish",
    "What's the current stock price of AAPL?",
    "Tell me a random fact",
    "Give me an inspirational quote",
    "Calculate 15 * 7 + 22",
    "What's the current date and time?",
    "Give me information about the movie 'Inception'",
    "What's the top news headline today?",
    "Search Wikipedia for 'Artificial Intelligence'",
    "Find recent arXiv papers on 'quantum computing'",
    "Give me 5 points about Jawaharlal Nehru you can use wikipedia",
    "Give me lisa rockstar official song link"
]

for i, question in enumerate(sample_questions):
    if st.sidebar.button(question, key=f"sample_question_{i}", on_click=update_input, args=(question,)):
        pass

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit and Hermes Thoth")
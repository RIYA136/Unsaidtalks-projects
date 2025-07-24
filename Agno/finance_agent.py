
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
import os

if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set your OpenAI API key as an environment variable: OPENAI_API_KEY")

instructions = dedent("""
    You are a financial analyst. Your job is to:
    - Give a quick summary of the company’s stock
    - Show the latest price, 52-week high and low
    - Include financials like P/E ratio, market cap, and EPS
    - Include recent news and what analysts say
    - Show risks and end with a future outlook

    Use tables, bullet points, and emojis like 📈 📉 ✅ ⚠️
    Always say: "This is not investment advice."
""")

finance_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=instructions,
    add_datetime_to_instructions=True,
    markdown=True,
    show_tool_calls=True,
)

finance_agent.print_response("Tell me about Apple (AAPL)'s financial performance.", stream=True)

finance_agent.print_response(
    dedent("""
    Compare these semiconductor companies:
    - NVIDIA (NVDA)
    - AMD (AMD)
    - Intel (INTC)
    - TSMC (TSM)
    Look at growth, valuation, and future outlook.
    """),
    stream=True
)

finance_agent.print_response(
    dedent("""
    How are these car companies doing?
    - Tesla (TSLA)
    - Ford (F)
    - General Motors (GM)
    - Toyota (TM)
    Focus on electric vehicles and traditional metrics.
    """),
    stream=True
)

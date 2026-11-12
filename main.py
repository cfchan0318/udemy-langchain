import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch

# from tavily import TavilyClient
#
# tavily = TavilyClient()

# @tool
# def search(query: str) -> dict:
#     """
#     Tool for searching through a query
#     :param query:
#     :return:
#     """
#     print(f"Searching {query}")
#     return tavily.search(query)

llm = ChatDeepSeek(model='deepseek-flash')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage("What is the weather in Hong Kong?")})
    print(result)

if __name__ == "__main__":
    main()
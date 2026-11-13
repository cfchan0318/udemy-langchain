from typing import List
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy

#Scripts
#from agents_under_the_hood.agent_loop_langchain_tool_calling import run_agent_main
#from agents_under_the_hood.agent_loop_raw_function_calling import run_agent_main
from react_prompt_underneath.raw_react_prompt import run_agent_main

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate answer")

llm = ChatDeepSeek(
    model='deepseek-flash',
    extra_body={"thinking": {"type": "disabled"}}
)
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse))

def main():
    result = agent.invoke({"messages":HumanMessage("What is the weather in Hong Kong?")})
    print(result)

if __name__ == "__main__":
    run_agent_main()
## To Learn agent loop raw
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL="gemma4:e4b"

@tool()
def get_product_price(product:str) -> float:
    """Look up the product price in the catalog"""
    prices = {"laptop":123.99, "headphones":149.21, "keyboard":99}
    return prices.get(product,0)

@tool()
def apply_discount(price:float,discount_tier) -> float:
    """Apply the discount to the price and return the final price
    Available Tiers: bronze, silver, gold
    """
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1-discount/100),2)

#---- AGENT LOOP ----------

@traceable(name="LangChain agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name:t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature =0, reasoning=True)
    llm_with_tools = llm.bind_tools(tools)


    print(f"Question: {question}")
    print("=" * 60)

    final_answer = None
    messages: list = [SystemMessage(
        content=( "You are a helpful shopping assistant"
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price\n"
                "You MUST call get_prodcut_price first to get the real price\n"
                "2. Only call apply discount AFTER you have received a price "
                "from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                  "ask them which tier to use - do NOT assume one."
                )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"Iteration: {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            final_answer = ai_message.content
            print(f"Final Answer: {ai_message.content}")
            return final_answer

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Tool Call: {tool_call_id} - {tool_name} - {tool_args}")

        tool_to_use = tools_dict.get(tool_name,None)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_use.invoke(tool_args)

        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    return None



def run_agent_main():
    print("hello langchain agent (.bind tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")

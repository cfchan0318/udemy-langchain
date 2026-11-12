## To Learn agent loop raw
from typing import List

from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the product price in the catalog"""
    prices = {"laptop": 123.99, "headphones": 149.21, "keyboard": 99}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier) -> float:
    """Apply the discount to the price and return the final price
    Available Tiers: bronze, silver, gold
    """
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Diff 2 without @tool, we must manually define json schema for each function

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Return the price of the product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string",
                                "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"},
                },
                "required": ["product"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# ---- Helper: traced ollama call ----
#Diff 3 without langchain, we must manually trace LLM calls for langsmith
@traceable(run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# ---- AGENT LOOP ----------

@traceable(name="LangChain agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    final_answer = None
    messages = [
        {
            "role":"system",
            "content" : ("You are a helpful shopping assistant"
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
        },
        {"role":"user","content":question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration: {iteration}")

        #Diff 5 :ollama chat directly , no llm with tools invoke
        response = ollama_chat_traced(messages=messages)
        ai_message = response.get("message")

        tool_calls = ai_message.get("tool_calls")

        if not tool_calls:
            final_answer = ai_message.content
            print(f"Final Answer: {ai_message.content}")
            return final_answer

        tool_call = tool_calls[0]
        tool_name = tool_call.get("function").get("name")
        tool_args = tool_call.get("function").get("arguments", {})

        print(f"Tool Call - {tool_name} - {tool_args}")

        tool_to_use = tools_dict.get(tool_name, None)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_use(**tool_args)

        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append({
            "role":"tool",
            "content":str(observation),
        })

    return None


def run_agent_main():
    print("hello langchain agent (.bind tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")

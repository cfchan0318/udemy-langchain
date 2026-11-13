## To Learn agent loop raw
from typing import List
import re
import inspect
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
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

#CHANGE 3: Tools live inside prompt as plain text

def get_tool_description(tools_dict):
    description = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        description.append(f"{tool_name}-{signature} - {docstring}")
    return "\n".join(description)

tool_description = get_tool_description(tools)
tool_names = ", ".join((tools.keys()))


react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price
You MUST call get_prodcut_price first to get the real price
2. Only call apply discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one."

Answer the following questions as best you can. You have access to the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:{{scratchpad}}
"""

# ---- Helper: traced ollama call ----
#Diff 3 without langchain, we must manually trace LLM calls for langsmith
@traceable(name='Ollama Chat',run_type="llm")
def ollama_chat_traced(model,messages,options):
    return ollama.chat(model=model, messages=messages, options=options)

# ---- AGENT LOOP ----------

@traceable(name="LangChain agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    final_answer = None
    scratchpad = ""
    prompt = react_prompt.format(question=question,scratchpad=scratchpad)


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration: {iteration}")
        full_prompt = prompt + scratchpad

        #Diff 5 :ollama chat directly , no llm with tools invoke
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role":"user", "content":full_prompt}],
            options={"stop":["\nObservation"],"temperature":0}
        )
        output = response.get("message").get("content",None)

        print("LLM output:",output)

        print(f" [Parsing] Looking for Final Answer in LLM output")
        final_answer_match = re.search(r"Final Answer:\s*(.*)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer: {final_answer}")
            print("=" * 60)
            print("Final Answer:", final_answer)
            return final_answer

        print(f" [Parsing] Looking for Action and Action input in LLM output")
        action_match = re.search(r"Action:\s*(.*)", output)
        action_input_match = re.search(r"Action Input:\s*(.*)", output)

        if not action_match or not action_input_match:
            print(f" [Parsing] ERROR: Action  Action input in LLM output could not be parsed")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"Tool Call - {tool_name} - {tool_input_raw}")

        # Split comma-separated args; strip key prefix if LLM outputs key value form
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("-", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool Executing] {tool_name} ({args}).......")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' is not available. tools we got : {list[str](tools.keys())}"
        else:
            observation = tools[tool_name](*args)


        print(f" [Tool Result]: {observation}")

        scratchpad += f"{output}\n Observation: {observation}\n Thought"


    return None


def run_agent_main():
    print("hello langchain agent (.bind tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")

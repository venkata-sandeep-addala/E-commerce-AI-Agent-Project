from pprint import pprint
from dotenv import load_dotenv
from langsmith import traceable
import ollama

load_dotenv()

MAX_ITERATIONS = 5
MODEL="qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Tool to get the price of a product."""
    # In a real implementation, this would query a database or an API
    print(f"    >> Executing get_product_price(product='{product_name}')\n")
    prices = {
        "laptop": 999,
        "smartphone": 499,
        "headphones": 199
    }
    return prices.get(product_name.lower(), 0)


@traceable(run_type="tool")
def get_final_price_after_discount(product_price: int, discount_code: str) -> float:
    """Tool to get the final price of a product after applying a discount code."""
    # In a real implementation, this would query a database or an API
    print(f"    >> Executing apply_discount(price={product_price}, discount_code='{discount_code}')\n")
    discounts = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discounts.get(discount_code.lower(), 0)
    final_price = product_price * (1 - discount/100)
    
    return final_price


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


tools_for_llm =  [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Tool to get the price of a product.",
        "parameters": {
          "type": "object",
          "required": ["product_name"],
          "properties": {
            "product_name": {"type": "string", "description": "The name of the product e.g. laptop, smartphone, headphones"}
          }
        }
      }
    },
	
	{
      "type": "function",
      "function": {
        "name": "get_final_price_after_discount",
        "description": "Tool to get the final price of a product after applying a discount code.",
        "parameters": {
          "type": "object",
          "required": ["product_price", "discount_code"],
          "properties": {
            "product_price": {"type": "number", "description": "The price of the product"},
			"discount_code": {"type": "string", "description": "The discount code of the customer. e.g. bronze, silver, gold"}
          }
        }
      }
    }
  ]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traceable(messages):
    
    ai_response = ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)
    return ai_response

# Run agent loop
@traceable(name="Ollama Agent Loop")
def run_agent_loop(query: str):
    
    tool_dict = {'get_product_price': get_product_price, 'get_final_price_after_discount': get_final_price_after_discount}
    
    messages = [
        {'role':'system','content': ("You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call get_final_price_after_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the get_final_price_after_discount tool.\n"
                "4. If the user does not specify a discount code, "
                "ask them which code to use — do NOT assume one.")},
        {'role':'user','content': query}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = ollama_chat_traceable(messages)
        
        tool_calls = ai_message.message.tool_calls
        
        if not tool_calls:
            print("No more tool calls. Ending loop.")
            return ai_message.message.content
        
        # Process only the first tool call in this example for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
    
        
        tool_to_use = tool_dict.get(tool_name)
        
        if tool_to_use is None:
            print(f"Tool '{tool_name}' not found. Ending loop.")
            return ai_message.message.content
        
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        
        observation = tool_to_use(**tool_args)
        
        messages.append(ai_message.message)
        messages.append({'role':'tool', 'tool_name': tool_name, 'content': str(observation)})
        
    print("ERROR: Max iterations reached without a final answer")
    return None
            
        
if __name__ == "__main__":
    final_respone = run_agent_loop(query="What is the final price of a smartphone with a silver discount code?")
    print(f"\nFinal Response: {final_respone}")
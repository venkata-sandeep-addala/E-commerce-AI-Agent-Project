import inspect
import re
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
def get_final_price_after_discount(product_price: str, discount_code: str) -> float:
    """Tool to get the final price of a product after applying a discount code."""
    # In a real implementation, this would query a database or an API
    print(f"    >> Executing apply_discount(price={product_price}, discount_code='{discount_code}')\n")
    discounts = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discounts.get(discount_code.lower(), 0)
    final_price = float(product_price) * (1 - discount/100)
    
    return final_price



# --- Helper: traced Ollama call ---
# Difference: Without LangChain, we must manually trace LLM calls for LangSmith.

tools_dict = {
    "get_product_price": get_product_price,
    "get_final_price_after_discount": get_final_price_after_discount
}


# Function to get the tool definitions for the LLM, including argument names and types
def get_tool_descriptions(tools):
    tool_descriptions = []
    for tool_name, tool_func in tools.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)  # Get the original function if it's wrapped by traceable
        
        signature = inspect.signature(original_function)
        doc_string = inspect.getdoc(original_function) or ""
        tool_descriptions.append(f'{tool_name}{signature}: {doc_string}')
        
    return '\n'.join(tool_descriptions)


tool_descriptions = get_tool_descriptions(tools_dict)
tool_names = ', '.join(tools_dict.keys())

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traceable(messages,options=None):
    
    ai_response = ollama.chat(model=MODEL, messages=[messages], options=options)
    return ai_response


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



# Run agent loop
@traceable(name="Ollama Agent Loop")
def run_agent_loop(query: str):
    
    prompt = react_prompt.format(question=query)
    scratchpad = ''
    
    
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        
        full_prompt = prompt + '\n' + scratchpad
        
        print(f"\n--- Iteration {iteration} ---")
        
        ai_message = ollama_chat_traceable(messages = {'role': 'user', 'content': full_prompt}, options = {'stop':['\nObservation:'], 'temperature':0})
        
        response = ai_message.message.content
        
        final_answer = re.search(r'Final Answer:\s*(.+)', response, re.IGNORECASE)
        
        if final_answer:
            print("Final answer found in response.")
            return final_answer.group(1).strip()
        
        
        
        action_match = re.search(r'Action:\s*(\w+)', response)
        action_input_raw = re.search(r'Action Input:\s*([^\n]+)', response)
        
        tool_name = action_match.group(1).strip()
        
        if tool_name not in tools_dict:
            print(f"ERROR: Invalid tool name '{tool_name}' in response. Ending loop.")
            return ai_message.message.content
        
        action_input = action_input_raw.group(1).strip() if action_input_raw else ''
        
        tool_args = {inp.split("=")[0].replace('"','').strip(): inp.split("=")[-1].replace('"', '').strip() for inp in action_input.split(',')}
    
        
        tool_to_use = tools_dict.get(tool_name)
        
        
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        
        observation = tool_to_use(**tool_args)
        
        scratchpad += response + f"\nObservation: {observation}\nThought:"

        
    print("ERROR: Max iterations reached without a final answer")
    return None
            
        
if __name__ == "__main__":
    final_respone = run_agent_loop(query="What is the final price of a smartphone with a silver discount code?")
    print(f"\nFinal Response: {final_respone}")
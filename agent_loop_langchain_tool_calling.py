from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 5
MODEL="qwen3:1.7b"

@tool
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


@tool
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



# Run agent loop
@traceable(name="LangChain Agent Loop")
def run_agent_loop(query: str):
    
    tools = [get_product_price, get_final_price_after_discount]
    tool_dict = {tool.name: tool for tool in tools}
    
    chat_model = init_chat_model(f'ollama:{MODEL}', temperature=0)
    
    llm_with_tools = chat_model.bind_tools(tools)

    messages = [
        SystemMessage(content="You are a helpful shopping assistant. "
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
                "ask them which code to use — do NOT assume one."),
        HumanMessage(content=query),
    ]

    for _ in range(MAX_ITERATIONS):
        
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print("No more tool calls. Ending loop.")
            return ai_message.content
        
        # Process only the first tool call in this example for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        tool_to_use = tool_dict.get(tool_name)
        
        if tool_to_use is None:
            print(f"Tool '{tool_name}' not found. Ending loop.")
            return ai_message.content
        
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        
        observation = tool_to_use.invoke(tool_args)
        
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
            
        
if __name__ == "__main__":
    final_respone = run_agent_loop(query="What is the final price of a smartphone with a silver discount code?")
    print(f"\nFinal Response: {final_respone}")
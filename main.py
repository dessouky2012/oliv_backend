import os
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from typing import Optional

from predict import predict_price
from perplexity_search import find_listings, find_general_commentary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. The application might not work as intended.")
openai.api_key = OPENAI_API_KEY

class UserMessage(BaseModel):
    message: str

app = FastAPI(
    title="Oliv - AI-driven Real Estate Assistant",
    description="Oliv uses OpenAI ChatGPT as the 'brain' and external APIs as tools to handle Dubai real estate queries.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the OpenAI functions (tools) ChatGPT can use
openai_functions = [
    {
        "name": "predict_price",
        "description": "Predicts approximate property price given details.",
        "parameters": {
            "type": "object",
            "properties": {
                "area_en": {"type": "string", "description": "Area name in English (e.g. 'Dubai Marina')."},
                "prop_type_en": {"type": "string", "description": "Property type in English (e.g. 'apartment')."},
                "actual_area": {"type": "number", "description": "The property area in sq. ft."},
                "bedrooms": {"type": "integer", "description": "Number of bedrooms. 0 if studio."},
                "parking": {"type": "integer", "description": "Number of parking spots."}
            },
            "required": ["area_en", "prop_type_en", "actual_area", "bedrooms", "parking"]
        }
    },
    {
        "name": "find_listings",
        "description": "Fetch property listings based on given criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Area or building name (e.g. 'Dubai Marina')."},
                "property_type": {"type": "string", "description": "Property type (e.g. 'apartment')."},
                "bedrooms": {"type": "integer", "description": "Number of bedrooms, 0 if studio."},
                "max_price": {"type": "integer", "description": "Maximum price or budget."},
                "exact_location": {"type": "string", "description": "Exact building or tower name if any.", "nullable": True}
            },
            "required": ["location", "property_type", "bedrooms", "max_price"]
        }
    },
    {
        "name": "find_general_commentary",
        "description": "Get general market commentary if no exact listings are available.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "property_type": {"type": "string"},
                "bedrooms": {"type": "integer"},
                "max_price": {"type": "integer"}
            },
            "required": ["location", "property_type", "bedrooms", "max_price"]
        }
    }
]

def call_tool(function_name: str, arguments: dict):
    """Execute the requested tool function with given arguments."""
    if function_name == "predict_price":
        return predict_price({
            "AREA_EN": arguments['area_en'],
            "PROP_TYPE_EN": arguments['prop_type_en'],
            "ACTUAL_AREA": arguments['actual_area'],
            "BEDROOMS": arguments['bedrooms'],
            "PARKING": arguments['parking']
        })
    elif function_name == "find_listings":
        return find_listings(
            arguments['location'],
            arguments['property_type'],
            arguments['bedrooms'],
            arguments['max_price'],
            exact_location=arguments.get('exact_location')
        )
    elif function_name == "find_general_commentary":
        return find_general_commentary(
            arguments['location'],
            arguments['property_type'],
            arguments['bedrooms'],
            arguments['max_price']
        )
    else:
        return {"error": "Unknown function"}

@app.get("/")
def read_root():
    return {"message": "Oliv backend running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    logger.info(f"User input: {user_input}")

    # System message sets the stage and personality of the assistant
    system_message = (
        "You are Oliv, a British AI real estate assistant specializing in Dubai properties. "
        "Your task is to help the user with Dubai real estate queries. If you need data to answer, "
        "call one of the provided functions. Once you get the data, integrate it into a helpful, "
        "warm, and professional final answer. If uncertain, ask the user for clarification.\n\n"
        "Always provide clear and context-rich answers, and when you have the data from the tools, "
        "incorporate it into your response.\n"
    )

    # First call: Let ChatGPT think and possibly call a function
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        functions=openai_functions,
        function_call="auto"
    )

    assistant_message = response.choices[0].message

    # If ChatGPT wants to call a function
    if assistant_message.get("function_call"):
        function_name = assistant_message["function_call"]["name"]
        function_args = assistant_message["function_call"]["arguments"]

        # Call the tool
        tool_result = call_tool(function_name, function_args)

        # Send the tool result back to ChatGPT
        second_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
                # The assistant message that requested the function call
                {"role": "assistant", "content": assistant_message.get("content", ""), "function_call": assistant_message["function_call"]},  
                {"role": "function", "name": function_name, "content": str(tool_result)}
            ]
        )
        final_msg = second_response.choices[0].message["content"].strip()
        return {"reply": final_msg}
    else:
        # No function call, just return what ChatGPT said
        final_msg = assistant_message["content"].strip()
        return {"reply": final_msg}

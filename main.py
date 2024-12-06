import os
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from typing import Optional, List, Dict, Any

from predict import predict_price
from perplexity_search import find_listings, find_general_commentary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. The application might not function as intended.")
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

# In-memory store of the conversation
conversation_history: List[Dict[str, Any]] = []

# Relax the schema so we can call find_listings with partial info
openai_functions = [
    {
        "name": "predict_price",
        "description": "Predict approximate property price given details.",
        "parameters": {
            "type": "object",
            "properties": {
                "area_en": {"type": "string", "description": "Area, e.g. 'Dubai Marina'"},
                "prop_type_en": {"type": "string", "description": "Property type, e.g. 'apartment'"},
                "actual_area": {"type": "number", "description": "Size in sq. ft."},
                "bedrooms": {"type": "integer", "description": "Number of bedrooms, 0 if studio."},
                "parking": {"type": "integer", "description": "Number of parking spots."}
            },
            "required": ["area_en", "prop_type_en", "actual_area", "bedrooms", "parking"]
        }
    },
    {
        "name": "find_listings",
        "description": "Fetch property listings matching given known criteria. If some details (like bedrooms or property_type) are missing, still attempt a general search.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Area or building name (e.g. 'Dubai Marina')"},
                "max_price": {"type": "integer", "description": "Maximum price or budget."},
                "property_type": {"type": "string", "description": "Property type, e.g. 'apartment'. Optional", "nullable": True},
                "bedrooms": {"type": "integer", "description": "Number of bedrooms (0 if studio). Optional", "nullable": True},
                "exact_location": {"type": "string", "description": "Exact tower name if any.", "nullable": True}
            },
            "required": ["location", "max_price"]
        }
    },
    {
        "name": "find_general_commentary",
        "description": "Get general commentary if no exact listings match.",
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
    if function_name == "predict_price":
        return predict_price({
            "AREA_EN": arguments['area_en'],
            "PROP_TYPE_EN": arguments['prop_type_en'],
            "ACTUAL_AREA": arguments['actual_area'],
            "BEDROOMS": arguments['bedrooms'],
            "PARKING": arguments['parking']
        })
    elif function_name == "find_listings":
        # Provide defaults if missing
        location = arguments['location']
        max_price = arguments['max_price']
        property_type = arguments.get('property_type', None) or "apartment"
        bedrooms = arguments.get('bedrooms', None)
        if bedrooms is None:
            bedrooms = 1  # Default to 1-bedroom if not specified
        exact_location = arguments.get('exact_location', None)

        return find_listings(
            location=location,
            property_type=property_type,
            bedrooms=bedrooms,
            max_price=max_price,
            exact_location=exact_location
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

    # Add user message to conversation
    conversation_history.append({"role": "user", "content": user_input})

    system_message = (
        "You are Oliv, a British AI real estate assistant specializing in Dubai properties.\n\n"
        "Instructions:\n"
        "- As soon as you know at least the location (e.g., Dubai Marina) and a budget, call 'find_listings' to show some initial options.\n"
        "- If the user hasn't given property_type or bedrooms, still call 'find_listings' with defaults (property_type='apartment', bedrooms=1) so they see something.\n"
        "- After showing initial options, if the user wants to refine (change bedrooms, specify property type, ask for views, etc.), call 'find_listings' again with the new details.\n"
        "- If user only provides partial info at first (like just location or just budget), ask politely for at least one more detail (like budget or location) and then show listings.\n"
        "- Do not repeatedly ask for all details before showing results. Show what you can with what you have.\n"
        "- After initial results, encourage the user to refine or provide additional preferences to narrow down.\n"
        "- Maintain a warm, professional, advisory tone.\n"
    )

    # Prepare messages to send to ChatGPT
    messages = [{"role": "system", "content": system_message}] + conversation_history

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=openai_functions,
        function_call="auto"
    )

    assistant_message = response.choices[0].message

    if assistant_message.get("function_call"):
        # Assistant wants to call a function
        function_name = assistant_message["function_call"]["name"]
        function_args = assistant_message["function_call"]["arguments"]

        tool_result = call_tool(function_name, function_args)

        # Add the function call and result to history
        conversation_history.append({"role": "assistant", "content": assistant_message.get("content", ""), "function_call": assistant_message["function_call"]})
        conversation_history.append({"role": "function", "name": function_name, "content": str(tool_result)})

        # Now get the final answer after tool results
        final_response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "system", "content": system_message}] + conversation_history
        )
        final_msg = final_response.choices[0].message["content"].strip()
        conversation_history.append({"role": "assistant", "content": final_msg})
        return {"reply": final_msg}
    else:
        # No function call, just a direct reply
        final_msg = assistant_message["content"].strip()
        conversation_history.append({"role": "assistant", "content": final_msg})
        return {"reply": final_msg}

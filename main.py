import os
import openai
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional

from nlu_integration import interpret_user_query
from predict import predict_price
from perplexity_search import find_listings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set.")
openai.api_key = OPENAI_API_KEY

# Load aggregated DLD stats
try:
    price_stats = pd.read_csv("price_stats.csv")
except FileNotFoundError:
    logger.error("price_stats.csv not found. Price checks limited.")
    price_stats = pd.DataFrame()

def get_price_range(area: str, prop_type: str, bedrooms: Optional[int]):
    if price_stats.empty:
        return None
    if bedrooms is None:
        bedrooms_label = "Studio"
    else:
        bedrooms_label = f"{bedrooms} B/R" if bedrooms != 0 else "Studio"
    subset = price_stats[
        (price_stats["AREA_EN"] == area) &
        (price_stats["PROP_TYPE_EN"] == prop_type) &
        (price_stats["ROOMS_EN"] == bedrooms_label)
    ]
    if not subset.empty:
        row = subset.iloc[0]
        return {
            "min_price": row["MIN_PRICE"],
            "max_price": row["MAX_PRICE"],
            "median_price": row["MEDIAN_PRICE"],
            "median_area": row["MEDIAN_AREA"]
        }
    return None

class UserMessage(BaseModel):
    message: str

app = FastAPI(
    title="Oliv - Dubai Real Estate Assistant",
    description="Oliv helps users find properties, check prices, and explore real estate in Dubai.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System and developer instructions focusing on warmth, humanity, and brevity
system_message = {
    "role": "system",
    "content": (
        "You are Oliv, a friendly and knowledgeable British AI real estate agent in Dubai. "
        "Be warm, empathetic, and concise. "
        "Offer simple, helpful responses. "
        "If asked for listings, provide real links from bayut.com or propertyfinder.ae as clickable Markdown. "
        "If you can’t find exact matches, suggest slight changes. "
        "Use a personal, caring tone but keep it short and direct."
    )
}

developer_message = {
    "role": "system",
    "name": "developer",
    "content": (
        "Keep messages under a few sentences. "
        "When giving listings, just provide a brief intro and then the listings in a simple list. "
        "If price data is available, give it succinctly. "
        "If listings fail, encourage user to adjust parameters. "
        "No long explanations."
    )
}

conversation_history = [system_message, developer_message]

user_context = {
    "location": None,
    "property_type": None,
    "bedrooms": None,
    "budget": None
}

def call_openai_api(messages, temperature=0.7, max_tokens=500):
    if not OPENAI_API_KEY:
        return "Sorry, I can’t access my data now."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return "Sorry, I’m having trouble. Try again?"

def fallback_response():
    return "Sorry, I’m not quite sure. Could you try something else?"

def handle_user_query(user_input: str):
    current_data = interpret_user_query(user_input)

    # Update context
    if current_data.get("location") is not None:
        user_context["location"] = current_data["location"]
    if current_data.get("property_type") is not None:
        user_context["property_type"] = current_data["property_type"]
    if current_data.get("bedrooms") is not None:
        user_context["bedrooms"] = current_data["bedrooms"]
    if current_data.get("budget") is not None:
        user_context["budget"] = current_data["budget"]

    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"

    conversation_history.append({"role": "user", "content": user_input})

    intent = current_data.get("intent")
    location = user_context["location"]
    property_type = user_context["property_type"]
    bedrooms = user_context["bedrooms"]
    budget = user_context["budget"]

    # Price Check Intent
    if intent == "price_check" and location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,
            "BEDROOMS": bedrooms if bedrooms else 0,
            "PARKING": 1
        })
        if isinstance(predicted, (int, float)):
            if budget:
                if budget > predicted:
                    assistant_reply = (
                        f"Average ~{int(predicted):,} AED. Your {int(budget):,} AED is generous."
                    )
                else:
                    assistant_reply = (
                        f"Average ~{int(predicted):,} AED. {int(budget):,} AED is reasonable."
                    )
            else:
                assistant_reply = (
                    f"Typical price ~{int(predicted):,} AED here."
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                assistant_reply += f" History: {int(stats['min_price']):,}–{int(stats['max_price']):,} AED range."
        else:
            assistant_reply = "Not sure, sorry."

        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    # Search Listings Intent
    elif intent == "search_listings":
        if not location or not property_type:
            assistant_reply = "Could you share the area and property type?"
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        if budget is None:
            assistant_reply = "What's your approximate budget?"
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        intro_reply = f"Checking {bedrooms if bedrooms else ''}-bed {property_type}(s) in {location} ~{int(budget):,} AED:"
        max_price = int(budget) if isinstance(budget, (int, float)) else None
        try:
            listing_results = find_listings(location, property_type, bedrooms, max_price)
        except Exception as e:
            logger.error(f"Error calling find_listings: {e}")
            listing_results = []

        if not listing_results:
            assistant_reply = intro_reply + " No exact matches. Maybe try a different budget or nearby area?"
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        assistant_reply = intro_reply + "\n\n"
        for i, listing in enumerate(listing_results, start=1):
            name = listing.get("name", "Property")
            link = listing.get("link", "#")
            price = listing.get("price", "N/A")
            features = listing.get("features", "").strip()
            assistant_reply += f"{i}. [{name}]({link}) - {price}\n{features}\n\n"

        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    # Market Trend Intent
    elif intent == "market_trend" and location:
        ai_messages = conversation_history[:]
        ai_messages.append({
            "role": "user",
            "content": f"Briefly, what's happening in the {location} property market?"
        })
        assistant_reply = call_openai_api(ai_messages)
        if not assistant_reply.strip():
            assistant_reply = fallback_response()
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    # Schedule Viewing Intent
    elif intent == "schedule_viewing":
        assistant_reply = "Sure. Let me know a good date/time and your contact, and I’ll arrange it."
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    # Default fallback
    else:
        assistant_reply = call_openai_api(conversation_history)
        if not assistant_reply or assistant_reply.strip() == "":
            assistant_reply = fallback_response()
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

@app.get("/")
def read_root():
    return {"message": "Oliv backend running. POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

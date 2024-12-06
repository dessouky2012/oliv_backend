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
from perplexity_search import find_listings, find_general_commentary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Oliv may have limited capabilities.")
openai.api_key = OPENAI_API_KEY

try:
    price_stats = pd.read_csv("price_stats.csv")
except FileNotFoundError:
    logger.error("price_stats.csv not found. Historical price checks will be limited.")
    price_stats = pd.DataFrame()

def get_price_range(area: str, prop_type: str, bedrooms: Optional[int]):
    """Retrieve historical pricing from DLD stats."""
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
    description="Oliv helps you find properties, check prices, and understand market trends in Dubai.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple persona instructions (no repeated GPT calls for finalizing)
persona = (
    "You are Oliv, a friendly and knowledgeable British AI real estate agent specialized in Dubai. "
    "You keep it warm and human, but also informative. Use DLD data if available, and supplement with real listings via Perplexity. "
    "If crucial details are missing, politely ask. Always give a concise, friendly response."
)

user_context = {
    "location": None,
    "property_type": None,
    "bedrooms": None,
    "budget": None
}

def get_perplexity_commentary(location, property_type, bedrooms, budget):
    if location and property_type:
        max_price = int(budget) if isinstance(budget, (int, float)) else None
        listings = find_listings(location, property_type, bedrooms, max_price)
        if listings:
            # Format listings in a friendly, human way
            reply = "\n\nI’ve found a few options that might interest you:\n"
            for i, l in enumerate(listings, start=1):
                name = l.get("name", "A property")
                link = l.get("link", "#")
                price = l.get("price", "Price not specified")
                features = l.get("features", "")
                reply += f"\nOption {i}:\nName: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return reply
        else:
            commentary = find_general_commentary(location, property_type, bedrooms, max_price)
            if commentary.strip():
                return "\n\n" + commentary
    return ""

def handle_user_query(user_input: str):
    # Understand the user's intent and details
    parsed = interpret_user_query(user_input)
    intent = parsed.get("intent")
    location = parsed.get("location", user_context["location"])
    property_type = parsed.get("property_type", user_context["property_type"])
    bedrooms = parsed.get("bedrooms", user_context["bedrooms"])
    budget = parsed.get("budget", user_context["budget"])

    # Update global user context
    if location is not None:
        user_context["location"] = location
    if property_type is not None:
        user_context["property_type"] = property_type
    if bedrooms is not None:
        user_context["bedrooms"] = bedrooms
    if budget is not None:
        user_context["budget"] = budget

    # If property_type is still missing but we have location and bedrooms, assume apartment
    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"
        property_type = "apartment"

    # Now respond based on intent:
    # PRICE CHECK
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
                    reply = (
                        f"Currently, a {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location} averages around {int(predicted):,} AED. "
                        f"Your budget of {int(budget):,} AED is above that typical range, giving you flexibility to pick a premium unit."
                    )
                else:
                    reply = (
                        f"For a {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location}, "
                        f"the going rate is about {int(predicted):,} AED, so {int(budget):,} AED fits nicely within the market range."
                    )
            else:
                reply = (
                    f"A {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location} "
                    f"often costs around {int(predicted):,} AED."
                )

            # Include historical DLD data
            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, units like these ranged from about {int(stats['min_price']):,} to {int(stats['max_price']):,} AED, "
                    f"with a median near {int(stats['median_price']):,} AED."
                )
        else:
            reply = "I’m sorry, I don’t have enough data to estimate a price range for that at the moment."

        # Add perplexity commentary (listings or suggestions)
        reply += get_perplexity_commentary(location, property_type, bedrooms, budget)
        return reply.strip()

    # SEARCH LISTINGS
    elif intent == "search_listings":
        if not location or not property_type:
            return (
                "Could you clarify the area of Dubai and the type of property? For example, 'a 2-bedroom apartment in Dubai Marina'."
            )

        if budget is None:
            return (
                f"So you're interested in a {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location}? "
                "What’s your approximate budget? That’ll help me find suitable options."
            )

        # Attempt to find listings directly
        reply_intro = (
            f"Alright, let's look for a {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location} "
            f"with a budget around {int(budget):,} AED."
        )
        commentary = get_perplexity_commentary(location, property_type, bedrooms, budget)
        if commentary.strip() == "":
            # No listings found, offer suggestions
            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                price_hint = (
                    f" Historically, similar places ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
            else:
                price_hint = ""
            reply = (
                reply_intro + "\n\nI’m not seeing exact matches at that price. You might consider adjusting your budget "
                f"or exploring nearby neighborhoods. {price_hint} Interested in looking at similar areas?"
            )
        else:
            # We got some listings or commentary
            reply = reply_intro + commentary
        return reply.strip()

    # MARKET TREND
    elif intent == "market_trend" and location:
        # Provide a simple market trend commentary from DLD data if possible
        # If DLD data is limited, just rely on Perplexity commentary
        stats = get_price_range(location, property_type if property_type else "apartment", bedrooms)
        if stats:
            base_trend = (
                f"In {location}, historically, prices ranged from about {int(stats['min_price']):,} to {int(stats['max_price']):,} AED, "
                f"with a median near {int(stats['median_price']):,} AED. "
                "Lately, buyers have shown steady interest, especially in well-known areas."
            )
        else:
            base_trend = f"In {location}, the market can vary, but let's see what’s currently happening."

        commentary = get_perplexity_commentary(location, property_type if property_type else "apartment", bedrooms, budget=None)
        reply = base_trend + commentary
        return reply.strip()

    # SCHEDULE VIEWING
    elif intent == "schedule_viewing":
        return (
            "Certainly! Let me know your preferred date and time for the viewing, "
            "and the best way to reach you, and I’ll arrange it."
        )

    # DEFAULT / NO CLEAR INTENT
    else:
        # If we have enough info, try to help anyway
        if location and property_type:
            commentary = get_perplexity_commentary(location, property_type, bedrooms, budget)
            if commentary.strip():
                return (
                    f"I see you’re interested in a {bedrooms if bedrooms else 'studio'}-bedroom {property_type} in {location}. "
                    "How else can I help you today?" + commentary
                )
            else:
                return (
                    f"Could you share a bit more about what you're looking for in {location}? "
                    "I’d love to help find the perfect place."
                )
        else:
            return (
                "Hi there! How can I help you with your Dubai property search today? "
                "For example, let me know the area, property type, and budget."
            )

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

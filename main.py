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
    description="Oliv helps users find properties, check prices, trends, and schedule viewings in Dubai.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_context = {
    "location": None,
    "property_type": None,
    "bedrooms": None,
    "budget": None
}

def get_perplexity_commentary(location, property_type, bedrooms, budget, exact_location=None):
    """Fetch listings or commentary ONLY if intent relates to property data."""
    if location and property_type:
        return get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location)
    return ""

def get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=None):
    """Fetch actual listings if available, otherwise general commentary."""
    max_price = int(budget) if isinstance(budget, (int, float)) else None
    listings = find_listings(location, property_type, bedrooms, max_price, exact_location=exact_location)
    if listings:
        reply = "\n\nHere are some options that match your request:\n"
        for i, l in enumerate(listings, start=1):
            name = l.get("name", "A lovely place")
            link = l.get("link", "#")
            price = l.get("price", "Not specified")
            features = l.get("features", "")
            reply += f"\nOption {i}:\nName: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
        return reply
    else:
        # If no direct listings found, try general commentary
        commentary = find_general_commentary(location, property_type, bedrooms, max_price)
        if commentary.strip():
            return "\n\n" + commentary
    return ""

def handle_user_query(user_input: str):
    # Interpret the user query
    parsed = interpret_user_query(user_input)
    intent = parsed.get("intent")
    location = parsed.get("location", user_context["location"])
    property_type = parsed.get("property_type", user_context["property_type"])
    bedrooms = parsed.get("bedrooms", user_context["bedrooms"])
    budget = parsed.get("budget", user_context["budget"])

    logger.info(f"User Input: {user_input}")
    logger.info(f"Extracted Intent: {intent}")
    logger.info(f"Location: {location}, Property Type: {property_type}, Bedrooms: {bedrooms}, Budget: {budget}")

    # Update context
    if location is not None:
        user_context["location"] = location
    if property_type is not None:
        user_context["property_type"] = property_type
    if bedrooms is not None:
        user_context["bedrooms"] = bedrooms
    if budget is not None:
        user_context["budget"] = budget

    # If property type missing but we have location and bedrooms, assume apartment
    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"
        property_type = "apartment"

    # Check if user specified an exact building location (like "Marina View Tower")
    exact_location = None
    if location and "tower" in location.lower():
        exact_location = location

    # Handle intents
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
                        f"For a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}, "
                        f"the average is around {int(predicted):,} AED. With your budget of {int(budget):,} AED, you have good flexibility."
                    )
                else:
                    reply = (
                        f"A {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} is about {int(predicted):,} AED, "
                        f"so {int(budget):,} AED fits well in the local range."
                    )
            else:
                reply = (
                    f"For a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}, "
                    f"you’re looking at roughly {int(predicted):,} AED."
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, similar units range {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )

            # Since this is a price check, user likely cares about listings, so call perplexity
            reply += get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
            return reply.strip()
        else:
            return (
                "I’m sorry, I don’t have enough data to give a reliable price estimate right now. "
                "Could you share more details or consider another area?"
            )

    elif intent == "search_listings":
        if not location or not property_type:
            return (
                "Could you share the area and property type you’re considering? Also, is this for a family home or investment?"
            )
        if budget is None:
            return (
                f"A {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}, got it. "
                "What’s your budget, and is this place for you or for investment?"
            )

        intro_reply = (
            f"Let’s see what’s available for a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} "
            f"around {int(budget):,} AED."
        )
        # Since user explicitly wants listings, call perplexity
        commentary = get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        if commentary.strip():
            return intro_reply + commentary
        else:
            # No matches found
            stats = get_price_range(location, property_type, bedrooms)
            price_hint = ""
            if stats:
                price_hint = (
                    f" Historically, similar units were {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
            return (
                intro_reply + "\n\nI can’t find exact matches right now. "
                f"{price_hint} Would you consider adjusting your criteria or exploring a nearby option?"
            )

    elif intent == "market_trend" and location:
        stats = get_price_range(location, property_type if property_type else "apartment", bedrooms)
        if stats:
            base_trend = (
                f"In {location}, properties historically ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                f"(median ~{int(stats['median_price']):,} AED). "
                "Would you like to see listings or discuss potential ROI?"
            )
        else:
            base_trend = (
                f"In {location}, market trends vary. "
                "Are you looking for long-term growth or rental income?"
            )

        # Market trends might involve seeing what’s out there
        commentary = get_listings_or_commentary(location, property_type if property_type else "apartment", bedrooms, budget=None, exact_location=exact_location)
        return (base_trend + commentary).strip()

    elif intent == "schedule_viewing":
        # No need to fetch listings here unless user specifically asks
        return (
            "Sure! Can you share your preferred date and time, and how I can reach you to confirm the appointment?"
        )

    else:
        # Non property-related queries or no recognized intent
        # For example, if user says "tell me more about you":
        # Just respond naturally, do not call perplexity commentary
        # Make Oliv sound personable and professional
        return (
            "I’m Oliv, your AI real estate assistant specializing in Dubai properties. "
            "I’m here to help guide you through areas, prices, and listings. "
            "Feel free to tell me more about what you’re hoping to find, or if you just want to understand more about the Dubai market!"
        )

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

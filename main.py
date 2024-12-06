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
    """Retrieve historical pricing range from DLD stats based on area, property type, and bedrooms."""
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
    """
    Fetch listings or commentary from Perplexity. If `exact_location` is provided (e.g., a building name),
    we request only listings that match that exact name.
    """
    if location and property_type:
        max_price = int(budget) if isinstance(budget, (int, float)) else None
        listings = find_listings(location, property_type, bedrooms, max_price, exact_location=exact_location)
        if listings:
            reply = "\n\nHere are some options that closely match your request:\n"
            for i, l in enumerate(listings, start=1):
                name = l.get("name", "A lovely place")
                link = l.get("link", "#")
                price = l.get("price", "Not specified")
                features = l.get("features", "")
                reply += f"\nOption {i}:\nName: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return reply
        else:
            # If no direct listings found
            if exact_location:
                return (
                    f"\n\nI couldn’t find any current listings explicitly in {exact_location}. "
                    "Sometimes availability can be limited. Would you consider a nearby building or adjusting your criteria?"
                )
            else:
                # No exact building, just general commentary
                commentary = find_general_commentary(location, property_type, bedrooms, max_price)
                if commentary.strip():
                    return "\n\n" + commentary
    return ""

def handle_user_query(user_input: str):
    # Interpret user query
    parsed = interpret_user_query(user_input)
    intent = parsed.get("intent")
    location = parsed.get("location", user_context["location"])
    property_type = parsed.get("property_type", user_context["property_type"])
    bedrooms = parsed.get("bedrooms", user_context["bedrooms"])
    budget = parsed.get("budget", user_context["budget"])

    logger.info(f"User Input: {user_input}")
    logger.info(f"Extracted Intent: {intent}")
    logger.info(f"Location: {location}, Property Type: {property_type}, Bedrooms: {bedrooms}, Budget: {budget}")

    # Update global context
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

    # Determine if the user mentioned a specific building by checking location details
    # For example, if user says "Marina View Tower" we treat that as exact_location
    exact_location = None
    # Simple heuristic: If user says something like "in Marina View Tower"
    # You might refine your NLU or do a substring check
    # For demonstration, let's assume that if the location string contains "Tower" or "Marina View Tower" specifically, 
    # we treat it as exact_location.
    if location and "tower" in location.lower():
        exact_location = location

    intent = parsed.get("intent")
    location = user_context["location"]
    property_type = user_context["property_type"]
    bedrooms = user_context["bedrooms"]
    budget = user_context["budget"]

    if not intent and not location and not property_type:
        return (
            "Hello there, I’m Oliv. I’d love to help with your Dubai property search. "
            "Are you looking for a home or an investment property? "
            "Please share the area, property type, and your approximate budget, and I’ll do my best."
        )

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
                        f"the market average is around {int(predicted):,} AED. "
                        f"Your budget of {int(budget):,} AED gives you nice flexibility."
                    )
                else:
                    reply = (
                        f"A {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} averages about {int(predicted):,} AED, "
                        f"and {int(budget):,} AED fits the local range."
                    )
            else:
                reply = (
                    f"A {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} "
                    f"often costs around {int(predicted):,} AED."
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, similar units ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
        else:
            reply = (
                f"Sorry, I don’t have exact historical data for a {property_type} in {location}. "
                "Could you tell me more about your priorities (e.g., return on investment, family-friendly, etc.)? "
                "This helps me guide you better."
            )

        reply += get_perplexity_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        return reply.strip()

    elif intent == "search_listings":
        if not location or not property_type:
            return (
                "Could you share the area and type of property you’re interested in? "
                "And is this for personal use or investment?"
            )

        if budget is None:
            return (
                f"So, a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}? "
                "What’s your budget range, and is this more for family living or as an investment?"
            )

        intro_reply = (
            f"Let’s see what’s currently available for a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} "
            f"around {int(budget):,} AED."
        )
        commentary = get_perplexity_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        if commentary.strip() == "":
            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                price_hint = (
                    f" Historically, similar units ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
            else:
                price_hint = ""
            reply = (
                intro_reply + "\n\nI’m not seeing exact matches at the moment. "
                "Would you consider tweaking your criteria or exploring a nearby building? "
                f"{price_hint}"
            )
        else:
            reply = intro_reply + commentary
        return reply.strip()

    elif intent == "market_trend" and location:
        stats = get_price_range(location, property_type if property_type else "apartment", bedrooms)
        if stats:
            base_trend = (
                f"In {location}, prices historically ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED, "
                f"(median ~{int(stats['median_price']):,} AED). "
                "Is your focus more on stable capital appreciation or rental yield?"
            )
        else:
            base_trend = (
                f"In {location}, trends vary. "
                "Are you leaning towards a family-friendly community or a high-yield investment location?"
            )

        commentary = get_perplexity_commentary(location, property_type if property_type else "apartment", bedrooms, None, exact_location=exact_location)
        reply = base_trend + commentary
        return reply.strip()

    elif intent == "schedule_viewing":
        return (
            "Great! When would be a convenient time for you, and how shall I get in touch? "
            "Also, are you viewing for yourself or on behalf of someone else?"
        )

    else:
        # If we have partial info but no clear intent, try to nudge the user
        if location and property_type:
            commentary = get_perplexity_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
            if commentary.strip():
                return (
                    f"I see you’re interested in a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}. "
                    "Could you tell me a bit more about your goals—family home or an investment?"
                    + commentary
                )
            else:
                return (
                    f"Could you clarify what you’re aiming for in {location}? "
                    "Is it a permanent home, a short-term residence, or purely an investment opportunity?"
                )

        return (
            "Hello! Could you share more details about what you’re looking for? "
            "For example: 'I’d like a 2-bedroom apartment in Dubai Marina for investment under 2 million AED.'"
        )

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

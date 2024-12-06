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

def get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=None):
    """Fetch listings or fallback to commentary if no listings found."""
    max_price = int(budget) if isinstance(budget, (int, float)) else None
    listings = find_listings(location, property_type, bedrooms, max_price, exact_location=exact_location)
    if listings:
        reply = "\n\nHere are some options that match your request:\n"
        for i, l in enumerate(listings, start=1):
            name = l.get("name", "A property")
            link = l.get("link", "#")
            price = l.get("price", "Not specified")
            features = l.get("features", "")
            reply += f"\nOption {i}:\nName: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
        return reply
    else:
        commentary = find_general_commentary(location, property_type, bedrooms, max_price)
        if commentary.strip():
            return "\n\n" + commentary
    return "\n\nI’m not finding any listings that match your criteria at the moment."

def format_bedroom_label(bedrooms):
    if bedrooms is None:
        return "studio"
    elif bedrooms == 1:
        return "1-bedroom"
    else:
        return f"{bedrooms}-bedroom"

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

    # Infer property type if missing but we have location and bedrooms
    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"
        property_type = "apartment"

    # If we have all details but no explicit intent, assume search_listings
    if intent is None and location and property_type and bedrooms is not None and budget:
        intent = "search_listings"

    # Check if user specified an exact building (like "Marina View Tower")
    exact_location = None
    if location and "tower" in location.lower():
        exact_location = location

    # Handle different intents
    if intent == "search_listings":
        if not location or not property_type or bedrooms is None or not budget:
            # Missing info, ask for details
            return (
                "Could you clarify your requirements? For instance, which area, how many bedrooms, and your budget?"
            )
        # We have everything, so perform the search
        bed_label = format_bedroom_label(bedrooms)
        intro_reply = (
            f"Right, you’re interested in a {bed_label} {property_type} in {location} "
            f"with a budget around {int(budget):,} AED. Let’s have a look at what’s available."
        )
        commentary = get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        return intro_reply + commentary

    elif intent == "price_check" and location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,  # Assuming area
            "BEDROOMS": bedrooms if bedrooms else 0,
            "PARKING": 1
        })
        bed_label = format_bedroom_label(bedrooms)

        if isinstance(predicted, (int, float)):
            predicted_str = f"{int(predicted):,} AED"
            if budget:
                if budget > predicted:
                    reply = (
                        f"Typically, a {bed_label} {property_type} in {location} averages around {predicted_str}. "
                        f"Your budget of {int(budget):,} AED gives you good flexibility."
                    )
                else:
                    reply = (
                        f"Typically, a {bed_label} {property_type} in {location} is around {predicted_str}, "
                        f"so your budget of {int(budget):,} AED fits well within typical market prices."
                    )
            else:
                reply = (
                    f"A {bed_label} {property_type} in {location} generally averages around {predicted_str}."
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, similar units range from {int(stats['min_price']):,} to "
                    f"{int(stats['max_price']):,} AED, with a median near {int(stats['median_price']):,} AED."
                )

            # Provide listings or commentary as well
            reply += get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
            return reply.strip()
        else:
            return (
                "I’m sorry, I don’t have enough data to provide a reliable estimate. "
                "Could you consider another similar area or provide more details?"
            )

    elif intent == "market_trend" and location:
        # Provide market insights
        bed_label = format_bedroom_label(bedrooms)
        pt = property_type if property_type else "apartment"
        stats = get_price_range(location, pt, bedrooms)
        if stats:
            base_trend = (
                f"In {location}, {bed_label} {pt}s have historically ranged from {int(stats['min_price']):,} "
                f"to {int(stats['max_price']):,} AED, with a median around {int(stats['median_price']):,} AED."
            )
        else:
            base_trend = f"In {location}, market prices vary based on exact location and quality."

        commentary = get_listings_or_commentary(location, pt, bedrooms, budget, exact_location=exact_location)
        return base_trend + commentary

    elif intent == "schedule_viewing":
        return (
            "Certainly! Please provide your preferred date, time, and contact details, and I’ll arrange a viewing."
        )

    else:
        # Fallback: Introduce Oliv again or try to prompt user for more info
        return (
            "I’m Oliv, your AI real estate assistant specializing in Dubai’s property market. "
            "I can help you find listings, check if a certain price is fair, or understand market trends. "
            "How can I assist you today?"
        )

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

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
    """Fetch actual listings if available, otherwise provide general commentary."""
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

    # If property type missing but we have location and bedrooms, assume apartment
    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"
        property_type = "apartment"

    # Check if user specified an exact building location (e.g., "Marina View Tower")
    exact_location = None
    if location and "tower" in location.lower():
        exact_location = location

    # If we have location, property_type, bedrooms, and budget, let's proactively show listings
    # This covers the scenario where user just gave these details and expects immediate results.
    if (location and property_type and (bedrooms is not None) and budget and intent in ["search_listings", None]):
        # Provide an introductory statement
        bed_label = format_bedroom_label(bedrooms)
        intro_reply = (
            f"Right, you’re looking for a {bed_label} {property_type} in {location}, "
            f"with a budget around {int(budget):,} AED. Let’s see what we can find.\n\n"
        )
        commentary = get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        if commentary.strip():
            return intro_reply + commentary.strip()
        else:
            # If no listings found, provide guidance and market context
            stats = get_price_range(location, property_type, bedrooms)
            price_hint = ""
            if stats:
                price_hint = (
                    f" Historically, similar units in {location} have ranged between {int(stats['min_price']):,} and "
                    f"{int(stats['max_price']):,} AED, with a median around {int(stats['median_price']):,} AED."
                )
            return (
                intro_reply +
                "I’m not seeing any exact matches at the moment. "
                f"{price_hint} Would you consider adjusting your criteria or exploring a nearby area in Dubai Marina?"
            )

    # Handle specific intents if above conditions not fully met

    if intent == "price_check" and location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,
            "BEDROOMS": bedrooms if bedrooms else 0,
            "PARKING": 1
        })
        bed_label = format_bedroom_label(bedrooms)

        if isinstance(predicted, (int, float)):
            predicted_str = f"{int(predicted):,} AED"
            if budget:
                if budget > predicted:
                    reply = (
                        f"For a {bed_label} {property_type} in {location}, the average often hovers around {predicted_str}. "
                        f"Your budget of {int(budget):,} AED gives you comfortable flexibility to choose from several options."
                    )
                else:
                    reply = (
                        f"For a {bed_label} {property_type} in {location}, prices average about {predicted_str}. "
                        f"Your budget of {int(budget):,} AED fits well into the local range."
                    )
            else:
                reply = (
                    f"For a {bed_label} {property_type} in {location}, you’re generally looking at around {predicted_str}."
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, similar units ranged from {int(stats['min_price']):,} to "
                    f"{int(stats['max_price']):,} AED, with a median near {int(stats['median_price']):,} AED."
                )

            # Provide listings or commentary
            reply += get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
            return reply.strip()
        else:
            return (
                "I’m sorry, I don’t have enough data to provide a reliable price estimate at the moment. "
                "Could you share more details or consider a slightly different area?"
            )

    elif intent == "search_listings":
        if not location or not property_type:
            return (
                "Could you share the area and property type you’re considering? Also, is this for a personal home or an investment?"
            )
        if budget is None:
            return (
                f"Understood, a {format_bedroom_label(bedrooms)} {property_type} in {location}. "
                "What’s your budget, and is this place for you or for investment?"
            )

        # If we reach here, it means we have location, property_type, and budget but it didn't trigger the early return above:
        # Just call listings again
        bed_label = format_bedroom_label(bedrooms)
        intro_reply = (
            f"Let’s see what’s available for a {bed_label} {property_type} in {location} "
            f"around {int(budget):,} AED.\n\n"
        )
        commentary = get_listings_or_commentary(location, property_type, bedrooms, budget, exact_location=exact_location)
        if commentary.strip():
            return intro_reply + commentary
        else:
            stats = get_price_range(location, property_type, bedrooms)
            price_hint = ""
            if stats:
                price_hint = (
                    f" Historically, similar units ranged from {int(stats['min_price']):,} to {int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
            return (
                intro_reply +
                "I’m not seeing any exact matches at the moment. " +
                price_hint + 
                " Would you consider adjusting your search parameters or looking at a nearby building?"
            )

    elif intent == "market_trend" and location:
        bed_label = format_bedroom_label(bedrooms)
        pt = property_type if property_type else "apartment"
        stats = get_price_range(location, pt, bedrooms)
        if stats:
            base_trend = (
                f"In {location}, {bed_label} {pt}s historically range between {int(stats['min_price']):,} and "
                f"{int(stats['max_price']):,} AED, with a median around {int(stats['median_price']):,} AED. "
                "This suggests a stable market segment with reasonable demand."
            )
        else:
            base_trend = (
                f"In {location}, the market can vary depending on exact location and quality. "
                "Would you like more specifics on recent trends or to see current listings?"
            )

        commentary = get_listings_or_commentary(location, pt, bedrooms, budget, exact_location=exact_location)
        return (base_trend + commentary).strip()

    elif intent == "schedule_viewing":
        return (
            "Certainly! Please let me know your preferred date, time, and contact details, and I’ll arrange the viewing at your convenience."
        )

    else:
        # Generic fallback
        return (
            "I’m Oliv, your AI real estate assistant specializing in Dubai’s property market. "
            "I can help you find listings, understand market trends, or check if a certain price is fair. "
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

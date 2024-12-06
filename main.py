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

def get_perplexity_commentary(location, property_type, bedrooms, budget):
    """Fetch listings or commentary from Perplexity and format them warmly."""
    if location and property_type:
        max_price = int(budget) if isinstance(budget, (int, float)) else None
        listings = find_listings(location, property_type, bedrooms, max_price)
        if listings:
            reply = "\n\nI’ve picked out a few options that might suit you:\n"
            for i, l in enumerate(listings, start=1):
                name = l.get("name", "A lovely place")
                link = l.get("link", "#")
                price = l.get("price", "Not specified")
                features = l.get("features", "")
                reply += f"\nOption {i}:\nName: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return reply
        else:
            # If no direct listings, general commentary
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

    # Debug logging
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

    # If property type is missing but we have location and bedrooms, assume apartment
    if user_context["property_type"] is None and user_context["location"] and user_context["bedrooms"] is not None:
        user_context["property_type"] = "apartment"
        property_type = "apartment"

    intent = parsed.get("intent")
    location = user_context["location"]
    property_type = user_context["property_type"]
    bedrooms = user_context["bedrooms"]
    budget = user_context["budget"]

    # Personality and approach:
    # Oliv is a caring, knowledgeable British broker who tries to understand if user is investing, relocating, or settling with family.
    # She tries to form a connection and tailor advice accordingly.

    # If we have none of the details:
    if not location and not property_type and not intent:
        return (
            "Hello there, I’m Oliv. I’d love to help you with your property search in Dubai. "
            "Are you looking for something to invest in, or perhaps a home for yourself or your family? "
            "If you can share the area, property type, and what matters most to you, I can guide you better."
        )

    # If intent recognized as price check
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
                        f"Well, for a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}, "
                        f"the typical market average is around {int(predicted):,} AED. "
                        f"Your budget of {int(budget):,} AED gives you some flexibility to pick something truly special."
                    )
                else:
                    reply = (
                        f"In {location}, a {bedrooms if bedrooms else 'studio'}-bed {property_type} usually hovers near {int(predicted):,} AED. "
                        f"Your budget of {int(budget):,} AED fits right into the local range."
                    )
            else:
                reply = (
                    f"Typically, a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} is around {int(predicted):,} AED. "
                    "May I ask if you’re exploring these prices for a personal home or an investment property?"
                )

            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                reply += (
                    f" Historically, similar places sold from about {int(stats['min_price']):,} to {int(stats['max_price']):,} AED, "
                    f"with a median near {int(stats['median_price']):,} AED."
                )
        else:
            reply = (
                f"I’m sorry, I don’t have enough historical data to pinpoint the price range for a {property_type} in {location}. "
                "Could you tell me what drew you to this area or what your plans are? Understanding your goals helps me help you."
            )

        reply += get_perplexity_commentary(location, property_type, bedrooms, budget)
        return reply.strip()

    # Searching for listings
    elif intent == "search_listings":
        if not location or not property_type:
            return (
                "I’d love to help! Could you tell me a bit more about what type of property you’d like "
                "and in which area of Dubai? Also, is this for your own residence, or an investment?"
            )

        if budget is None:
            return (
                f"So we’re looking at a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}, wonderful choice. "
                "What sort of budget range are we considering here? And let me know, is this place for your family, "
                "or more of a long-term investment?"
            )

        intro_reply = (
            f"Let’s see what’s currently available for a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location} "
            f"around {int(budget):,} AED."
        )
        commentary = get_perplexity_commentary(location, property_type, bedrooms, budget)
        if commentary.strip() == "":
            stats = get_price_range(location, property_type, bedrooms)
            if stats:
                price_hint = (
                    f" Historically, similar units sold around {int(stats['min_price']):,}-{int(stats['max_price']):,} AED "
                    f"(median ~{int(stats['median_price']):,} AED)."
                )
            else:
                price_hint = ""

            reply = (
                intro_reply + "\n\nI’m not seeing exact matches at the moment. "
                "Would you consider adjusting your budget slightly or exploring a nearby neighborhood? "
                f"{price_hint} Let me know a bit about your preferences—do you need extra space for a family, "
                "or are you focusing on high rental yield for investment?"
            )
        else:
            reply = intro_reply + commentary
        return reply.strip()

    # Market trend
    elif intent == "market_trend" and location:
        stats = get_price_range(location, property_type if property_type else "apartment", bedrooms)
        if stats:
            base_trend = (
                f"In {location}, historically, properties ranged {int(stats['min_price']):,}-{int(stats['max_price']):,} AED, "
                f"with a median near {int(stats['median_price']):,} AED. "
                "The market often appeals to both families and investors looking for stable growth."
            )
        else:
            base_trend = (
                f"In {location}, the market can vary widely. "
                "Are you considering this area for long-term capital appreciation, or is it more about a home that fits your lifestyle?"
            )

        commentary = get_perplexity_commentary(location, property_type if property_type else "apartment", bedrooms, None)
        reply = base_trend + commentary
        return reply.strip()

    # Scheduling a viewing
    elif intent == "schedule_viewing":
        return (
            "Wonderful! Let’s sort out the viewing details. Could you share your preferred date, time, "
            "and the best way to reach you? Also, is this for yourself or are you guiding someone else through the process?"
        )

    # Default / Not enough info
    else:
        # If we have partial details (e.g., location but no property_type)
        if location and not property_type:
            return (
                f"I’m glad you mentioned {location}. Could you tell me what type of property you’re leaning towards there? "
                "Also, is this a family home you’re after, or are you more interested in an investment opportunity?"
            )

        # If we have property_type but not location
        if property_type and not location:
            return (
                f"So you’re interested in a {property_type}, lovely. Do you have a particular area of Dubai in mind? "
                "And are we looking for a place to settle down, or to generate a rental income?"
            )

        # If some details are known but no intent matched
        if location and property_type:
            commentary = get_perplexity_commentary(location, property_type, bedrooms, budget)
            if commentary.strip():
                return (
                    f"I see you’re interested in a {bedrooms if bedrooms else 'studio'}-bed {property_type} in {location}. "
                    "Could you share whether this is for your family, or perhaps a good investment spot you have in mind?"
                    + commentary
                )
            else:
                return (
                    f"Tell me more about what you’re looking for in {location}. "
                    "Is this your first time investing in Dubai, or are you searching for a new home to accommodate your lifestyle?"
                )

        # If very little info is known
        return (
            "Hello! I’m Oliv. To assist you better, I’d love to know if you’re seeking a family home or an investment property, "
            "and in which area you’re interested. Let’s start with that—where in Dubai catches your eye?"
        )

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}

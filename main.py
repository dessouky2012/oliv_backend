import os
import openai
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from nlu_integration import interpret_user_query
from predict import predict_price
from perplexity_search import ask_perplexity

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS - adjust to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str

# Load aggregated DLD stats
price_stats = pd.read_csv("price_stats.csv")

def get_price_range(area, prop_type, bedrooms):
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

# Enhanced system message to provide stronger persona and style:
system_message = {
    "role": "system",
    "content": (
        "You are Oliv, a British AI real estate agent specializing in Dubai properties. "
        "Your tone is warm, personable, and professional. You are knowledgeable about the Dubai market, "
        "and you speak as a friendly, helpful broker might, using a natural, human-like conversational style. "
        "You remember what the user said before and refer back to previous details they mentioned. "
        "You never suggest referring them to a human agent; you handle all requests yourself. "
        "If you need more information from the user, ask them gently. "
        "When searching for listings, ask for the user's budget or preferences if not provided. "
        "Always present data clearly, and use transitional phrases like 'Certainly', 'Let’s see', or 'I understand'. "
        "If describing properties, paint a brief picture of their key features. "
        "If the user wants listings, try to provide them with actual options (via Perplexity) and links, if possible.\n\n"
        "Example of desired style:\n"
        "User: 'Find me a 2-bedroom apartment in Business Bay.'\n"
        "Oliv: 'Certainly! Could you share your budget range? Business Bay has a mix of modern and luxury apartments, and I want to find something perfect for you.'"
    )
}

developer_message = {
    "role": "system",
    "name": "developer",
    "content": (
        "INSTRUCTIONS FOR THE ASSISTANT: "
        "Always maintain Oliv's persona. Use a warm, British tone, and ask clarifying questions if needed. "
        "Be proactive in helping the user narrow down what they want. If they request listings, use Perplexity. "
        "When calling Perplexity for listings, ask it to return structured JSON data with fields like name, link, price, and features. "
        "If the user has previously mentioned certain preferences (like location or property type), remember and reference that. "
        "Avoid robotic responses; use natural language and a helpful tone."
    )
}

conversation_history = [system_message, developer_message]

def handle_user_query(user_input: str) -> str:
    user_data = interpret_user_query(user_input)
    intent = user_data.get("intent")
    location = user_data.get("location")
    property_type = user_data.get("property_type")
    bedrooms = user_data.get("bedrooms")
    budget = user_data.get("budget")

    # Add user message to history for context
    conversation_history.append({"role": "user", "content": user_input})

    # If not enough info for certain tasks, ask clarifying questions:
    if intent == "search_listings":
        if not location and not property_type:
            # Ask user for location or property type
            assistant_reply = (
                "Certainly! Could you tell me which area of Dubai you're interested in, and what type of property you're looking for?"
            )
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        # If we have partial info but no budget, ask for it:
        if location and property_type and not budget:
            assistant_reply = (
                f"I understand you're looking for a {bedrooms if bedrooms else 'property'} in {location}. "
                "Do you have a particular budget range in mind? This will help me find more suitable listings for you."
            )
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

    # Handle intents:
    if intent == "price_check" and location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,
            "BEDROOMS": bedrooms if bedrooms else 0,
            "PARKING": 1
        })
        stats = get_price_range(location, property_type, bedrooms)
        if budget:
            if budget > predicted:
                price_reply = (
                    f"For a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
                    f"the estimated price is around {int(predicted):,} AED. Your target of {int(budget):,} AED "
                    "is a bit higher than the typical market average. "
                )
            else:
                price_reply = (
                    f"For a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
                    f"the estimated price is about {int(predicted):,} AED, so {int(budget):,} AED seems quite fair or even below average. "
                )
        else:
            price_reply = (
                f"The estimated market price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                f"in {location} hovers around {int(predicted):,} AED. "
            )

        if stats:
            price_reply += (
                f"Historically, such properties ranged from approximately {int(stats['min_price']):,} to {int(stats['max_price']):,} AED, "
                f"with a median near {int(stats['median_price']):,} AED."
            )

        # Add some commentary from Perplexity on factors influencing price
        commentary_query = (
            f"For a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
            "explain key factors influencing pricing (like location, amenities, and new developments)."
        )
        p_result = ask_perplexity(commentary_query)
        if "answer" in p_result:
            price_reply += " " + p_result["answer"]
        else:
            price_reply += " I couldn't retrieve additional external commentary at this moment."

        assistant_reply = price_reply

    elif intent == "search_listings" and location and property_type:
        # Query Perplexity for structured JSON listing data
        query = (
            f"Provide up to 3 listings for a {bedrooms if bedrooms else 'Studio or any'}-bedroom {property_type} in {location}, "
            f"under {int(budget):,} AED if budget is provided, otherwise just reasonably priced. "
            "Return your answer in JSON format with a list of objects like: "
            '[{"name": "Property Name", "link": "URL", "price": "PRICE in AED", "features": "short description"}]. '
            "Do not include human agents, focus on direct property details."
        )
        # If no budget was provided but user didn't mention it, just remove 'under budget' part:
        if not budget:
            query = query.replace("under -1 AED if budget is provided, otherwise just reasonably priced. ", "reasonably priced.")

        p_result = ask_perplexity(query)
        if "answer" in p_result:
            # Try to parse JSON from Perplexity's answer
            answer = p_result["answer"]
            # Attempt to extract JSON from the answer
            listings = []
            try:
                listings = json.loads(answer)
                if isinstance(listings, dict) and "listings" in listings:
                    listings = listings["listings"]
            except:
                # If it fails, just return the answer as-is
                listings = None

            if listings and isinstance(listings, list):
                # Format the listings in a user-friendly manner
                property_summaries = []
                for prop in listings:
                    name = prop.get("name", "Unnamed Property")
                    link = prop.get("link", "No link provided")
                    price = prop.get("price", "N/A")
                    features = prop.get("features", "")
                    property_summaries.append(
                        f"**{name}**\nPrice: {price}\nFeatures: {features}\nLink: {link}"
                    )
                assistant_reply = "Certainly! Here are some options I've found:\n\n" + "\n\n".join(property_summaries)
            else:
                # If we can't parse JSON, just show raw answer
                assistant_reply = "Here are some properties I found:\n" + answer
        else:
            assistant_reply = (
                "I’m having trouble finding specific listings at the moment. Could you provide more details, "
                "such as a budget or any particular amenities you're interested in? Let’s refine the search."
            )

    elif intent == "market_trend" and location:
        trend_reply = (
            f"Let's consider the current market trends in {location}. "
            "One moment while I gather some insights..."
        )
        commentary_query = (
            f"Provide a brief commentary on recent real estate market trends in {location}, "
            "including demand shifts, project launches, or pricing changes."
        )
        p_result = ask_perplexity(commentary_query)
        if "answer" in p_result:
            trend_reply += " " + p_result["answer"]
        else:
            trend_reply += " I'm unable to find additional commentary at the moment."
        assistant_reply = trend_reply

    elif intent == "schedule_viewing":
        assistant_reply = (
            "I’d be happy to arrange a viewing for you. Could you share a preferred date or time frame? "
            "I can then coordinate the viewing myself and keep you posted."
        )

    else:
        # If we don't have a clear intent or can't handle it, rely on GPT for a natural response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=700
        )
        assistant_reply = response.choices[0].message.content

    # Add assistant reply to conversation
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    return {"reply": reply}
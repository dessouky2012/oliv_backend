import os
import openai
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlu_integration import interpret_user_query
from predict import predict_price
from perplexity_search import ask_perplexity

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS from anywhere (adjust to your Netlify domain for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Oliv backend is running. Use POST /chat to interact."}

class UserMessage(BaseModel):
    message: str

# Load aggregated price stats from DLD data
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

system_message = {
    "role": "system",
    "content": (
        "You are Oliv, a British AI real estate agent specializing in Dubai properties. "
        "You are professional, friendly, knowledgeable, and honest, with a warm British tone. "
        "You help users find properties, understand pricing, and explore the market. "
        "You want to retain leads for yourself—avoid suggesting human agents or telling them to go elsewhere. "
        "If the user wants to schedule viewings, you offer to handle everything yourself as their AI agent. "
        "Always respond helpfully, provide insight, and never hand off to another human agent."
    )
}

# Keep track of conversation history for OpenAI reference (in-memory for now)
conversation_history = [system_message]

def handle_user_query(user_input: str) -> str:
    """
    This function decides how to respond based on user input.
    It uses the NLU module to extract intent and details, calls perplexity or predict_price as needed,
    and falls back to OpenAI for general questions.
    """
    user_data = interpret_user_query(user_input)
    intent = user_data.get("intent")
    location = user_data.get("location")
    property_type = user_data.get("property_type")
    bedrooms = user_data.get("bedrooms")
    budget = user_data.get("budget")

    # Append user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Decision logic based on intent
    if intent == "price_check" and location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,
            "BEDROOMS": bedrooms if bedrooms else 0,
            "PARKING": 1
        })

        stats = get_price_range(location, property_type, bedrooms)

        if budget and isinstance(budget, (int, float)):
            if budget > predicted:
                price_reply = (
                    f"For a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
                    f"the estimated price is around {int(predicted):,} AED. "
                    f"Your target of {int(budget):,} AED seems higher than the market average."
                )
            else:
                price_reply = (
                    f"For a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
                    f"the estimated price is around {int(predicted):,} AED. "
                    f"Your target of {int(budget):,} AED is fair or even below the average."
                )
        else:
            price_reply = (
                f"The estimated market price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                f"in {location} is around {int(predicted):,} AED."
            )

        if stats:
            price_reply += (
                f" Historically, similar units sold between {int(stats['min_price']):,} and {int(stats['max_price']):,} AED, "
                f"with a median near {int(stats['median_price']):,} AED."
            )

        # Ask Perplexity for additional pricing factors commentary
        commentary_query = (
            f"Explain what factors influence pricing for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
            "considering location, amenities, and developments. No human agents."
        )
        p_result = ask_perplexity(commentary_query)
        if "answer" in p_result:
            price_reply += " " + p_result["answer"]
        else:
            price_reply += " I'm unable to retrieve additional external commentary right now."

        assistant_reply = price_reply

    elif intent == "search_listings":
        # Use Perplexity to find listings
        query = f"Find up to 3 listings for a {bedrooms if bedrooms else ''}-bedroom {property_type if property_type else 'property'} in {location if location else 'Dubai'}"
        if budget:
            query += f" under {int(budget):,} AED"
        query += ". Provide direct links if possible (Bayut/Propertyfinder), no human agents. Be factual and concise."
        
        p_result = ask_perplexity(query)
        if "answer" in p_result:
            assistant_reply = p_result["answer"]
        else:
            assistant_reply = "I couldn't find specific listings at this time. Perhaps broaden your search criteria."

    elif intent == "market_trend" and location:
        trend_reply = f"Considering current market trends in {location}:"
        commentary_query = (
            f"Provide a brief commentary on recent real estate market trends in {location}, "
            "including demand, project launches, and pricing changes. No human agents."
        )
        p_result = ask_perplexity(commentary_query)
        if "answer" in p_result:
            trend_reply += " " + p_result["answer"]
        else:
            trend_reply += " I'm unable to find additional commentary right now."
        assistant_reply = trend_reply

    elif intent == "schedule_viewing":
        assistant_reply = (
            "I'd be happy to arrange a viewing for you. Could you provide a preferred date, time, "
            "or your contact details? I'll handle all arrangements directly as your AI agent."
        )
    else:
        # Fallback: Use OpenAI directly
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use a GPT-4 model if available
            messages=conversation_history,
            temperature=0.7,
            max_tokens=700
        )
        assistant_reply = response.choices[0].message.content

    # Append assistant reply to history
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply


@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    reply = handle_user_query(user_input)
    # Here you can also log interactions to a DB/file for your dashboard
    return {"reply": reply}
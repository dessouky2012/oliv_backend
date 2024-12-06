import os
import openai
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlu_integration import interpret_user_query
from predict import predict_price
from perplexity_search import ask_perplexity

# Read your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS from everywhere for now. Ideally restrict this to your Netlify URL.
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

# Load aggregated price stats
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

# Oliv's Personality and Instructions
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

conversation_history = [system_message]

@app.post("/chat")
def chat_with_oliv(user_msg: UserMessage):
    user_input = user_msg.message.strip()
    user_data = interpret_user_query(user_input)
    intent = user_data.get("intent")
    location = user_data.get("location")
    property_type = user_data.get("property_type")
    bedrooms = user_data.get("bedrooms")
    budget = user_data.get("budget")

    conversation_history.append({"role": "user", "content": user_input})
    assistant_reply = ""

    if intent == "price_check" and location and property_type:
        # Price Check:
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
                    f"The estimated price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                    f"in {location} is around {int(predicted):,} AED. Your mentioned price of {int(budget):,} AED "
                    "seems higher than the average."
                )
            else:
                price_reply = (
                    f"The estimated price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                    f"in {location} is around {int(predicted):,} AED. Your mentioned price of {int(budget):,} AED "
                    "is fair or below the average."
                )
        else:
            price_reply = (
                f"The estimated market price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                f"in {location} is around {int(predicted):,} AED."
            )

        if stats:
            price_reply += (
                f" Historically, similar units have sold between {int(stats['min_price']):,} and {int(stats['max_price']):,} AED, "
                f"with a median of about {int(stats['median_price']):,} AED."
            )

        # Use Perplexity to get external commentary about pricing factors
        commentary_query = (
            f"Given a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} in {location}, "
            "explain what factors influence pricing (location, amenities, new developments) "
            "without referring to human agents."
        )
        perplexity_result = ask_perplexity(commentary_query)
        if "answer" in perplexity_result:
            price_reply += " " + perplexity_result["answer"]
        else:
            price_reply += " I couldn't find more details at the moment from external sources."

        assistant_reply = price_reply

    elif intent == "search_listings":
        # Searching for listings. Use Perplexity to find actual examples if possible
        query = (
            f"Find up to 3 listings for a {bedrooms if bedrooms else ''}-bedroom {property_type if property_type else 'property'} "
            f"in {location if location else 'Dubai'}"
        )
        if budget:
            query += f" under {int(budget):,} AED"
        query += (
            ". Provide direct links if possible (e.g., Bayut/Propertyfinder), and do not mention human agents. "
            "Keep it factual and concise."
        )

        perplexity_result = ask_perplexity(query)
        if "answer" in perplexity_result:
            assistant_reply = perplexity_result["answer"]
        else:
            assistant_reply = "I couldn't find specific listings at this time. Consider adjusting your criteria."

    elif intent == "market_trend" and location:
        # Market Trend:
        trend_reply = f"Let’s consider the current market trends in {location}."
        commentary_query = (
            f"Provide a brief commentary on recent real estate market trends in {location}, "
            "including demand shifts, project launches, or pricing changes. No human agents."
        )
        perplexity_result = ask_perplexity(commentary_query)
        if "answer" in perplexity_result:
            trend_reply += " " + perplexity_result["answer"]
        else:
            trend_reply += " I'm unable to find additional commentary right now."
        assistant_reply = trend_reply

    elif intent == "schedule_viewing":
        # Scheduling a viewing:
        assistant_reply = (
            "I can schedule the viewing for you myself. Could you provide a preferred date, time, or contact details? "
            "I'll manage all arrangements directly."
        )

    else:
        # Fallback: Use GPT directly if no specialized intent is found
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=700
        )
        assistant_reply = response.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": assistant_reply})
    return {"reply": assistant_reply}
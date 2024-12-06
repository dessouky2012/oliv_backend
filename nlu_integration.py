# nlu_integration.py
import os
import openai

# Read the OpenAI API key from an environment variable instead of hardcoding it
openai.api_key = os.getenv("OPENAI_API_KEY")

def interpret_user_query(user_query: str) -> dict:
    prompt = f"""
    You are Oliv's NLU module. Extract structured data from the user's query about real estate in Dubai.

    Identify:
      - intent: "search_listings", "price_check", "market_trend", "schedule_viewing", or None
      - location: str or None
      - property_type: str or None
      - bedrooms: int or None
      - budget: float or None
      - timeframe: str or None

    Return strictly in JSON format. No extra text.

    User query: "{user_query}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Keeping gpt-4o as requested
        messages=[
            {"role": "system", "content": "You only extract structured data in JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()
    import json
    try:
        data = json.loads(content)
    except:
        data = {
            "intent": None,
            "location": None,
            "property_type": None,
            "bedrooms": None,
            "budget": None,
            "timeframe": None
        }

    return data
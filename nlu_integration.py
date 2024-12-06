import os
import openai
import json
import logging

logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

def interpret_user_query(user_query: str) -> dict:
    if not OPENAI_API_KEY:
        return {
            "intent": None,
            "location": None,
            "property_type": None,
            "bedrooms": None,
            "budget": None,
            "timeframe": None
        }

    prompt = f"""
    Extract structured data in JSON from this user query about Dubai real estate:
    - intent: "search_listings", "price_check", "market_trend", "schedule_viewing", or None
    - location: str or None
    - property_type: str or None
    - bedrooms: int or None
    - budget: float or None
    - timeframe: str or None

    User query: "{user_query}"
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception as e:
        logger.error(f"Failed to interpret user query: {e}")
        return {
            "intent": None,
            "location": None,
            "property_type": None,
            "bedrooms": None,
            "budget": None,
            "timeframe": None
        }

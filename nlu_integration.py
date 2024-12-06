import os
import openai
import json
import logging

logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def interpret_user_query(user_query: str) -> dict:
    """Use OpenAI to interpret user query into structured data:
    intent: "search_listings", "price_check", "market_trend", "schedule_viewing", or None
    location, property_type, bedrooms, budget, timeframe
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. Returning empty interpretation.")
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
    - property_type: str or None (e.g., 'apartment', 'villa')
    - bedrooms: int or None
    - budget: float or None
    - timeframe: str or None (e.g., 'next month', 'immediately')

    User query: "{user_query}"
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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

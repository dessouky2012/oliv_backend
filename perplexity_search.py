import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int):
    """Call Perplexity API to find listings. Returns a list of JSON objects with name, link, price, features."""
    if not PERPLEXITY_API_KEY:
        logger.warning("PERPLEXITY_API_KEY not set. Returning empty listings.")
        return []

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    bed_text = f"{bedrooms}-bedroom" if bedrooms else ""
    budget_text = f"around {price_max} AED" if price_max else "a reasonable price range"

    query = (
        f"Find up to 3 listings for a {bed_text} {property_type} in {location} {budget_text}. "
        "Return ONLY JSON as a list of objects: "
        '[{"name":"Property Name","link":"URL","price":"XXX AED","features":"Short description"}].'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a data retrieval assistant. Return ONLY JSON."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content","").strip()
            try:
                listings = json.loads(content)
                if isinstance(listings, list):
                    return listings
                else:
                    return []
            except json.JSONDecodeError:
                logger.error("Failed to parse listings JSON from Perplexity.")
                return []
        else:
            logger.error(f"Perplexity returned status {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Exception calling Perplexity: {e}")
        return []

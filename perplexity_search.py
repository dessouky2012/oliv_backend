import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Choose a model known to work or widely supported. Adjust if Perplexity docs require a specific model.
MODEL_NAME = "llama-3.1-sonar-huge-128k-online"

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

    # Strict instructions to return ONLY JSON
    user_prompt = (
        f"Find a list of current property listings for a {bed_text} {property_type} in {location} {budget_text}. "
        "Return ONLY valid JSON as a list of objects with keys: name, link, price, features. "
        "No extra text. For example:\n\n"
        '[{"name":"Luxury Apartment","link":"https://example.com","price":"1,500,000 AED","features":"3 beds, 2 baths"}]'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Return ONLY JSON."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 700,
        "temperature": 0.0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            # Try direct JSON parsing first
            try:
                listings = json.loads(content)
                if isinstance(listings, list):
                    return listings
                else:
                    logger.error("Perplexity response did not return a list.")
                    return []
            except json.JSONDecodeError:
                # Attempt to extract JSON if there's extra formatting
                logger.error("Failed to parse JSON directly. Attempting fallback parsing.")
                # Fallback: try to locate JSON in content
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1:
                    maybe_json = content[start:end+1]
                    try:
                        listings = json.loads(maybe_json)
                        if isinstance(listings, list):
                            return listings
                    except json.JSONDecodeError:
                        pass
                logger.error("Failed to parse listings JSON from Perplexity after fallback.")
                return []
        else:
            logger.error(f"Perplexity returned status {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Exception calling Perplexity: {e}")
        return []

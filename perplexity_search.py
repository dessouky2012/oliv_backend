import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"

# If Perplexity has specific models, adjust accordingly
MODEL_NAME = "llama-3.1-sonar-huge-128k-online"

def call_perplexity(query: str) -> str:
    """Call Perplexity API with given query and return raw response content."""
    if not PERPLEXITY_API_KEY:
        logger.warning("PERPLEXITY_API_KEY not set. Returning empty response.")
        return "[]"

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Return ONLY JSON. If no results, return []"},
            {"role": "user", "content": query}
        ],
        "max_tokens": 700,
        "temperature": 0.0
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        logger.info(f"Perplexity API status: {response.status_code}")
        logger.info("Perplexity raw response: " + response.text)

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            logger.info("Perplexity extracted content: " + content)
            return content
        else:
            logger.error(f"Perplexity returned status {response.status_code}: {response.text}")
            return "[]"
    except Exception as e:
        logger.error(f"Exception calling Perplexity: {e}")
        return "[]"

def parse_listings(content: str):
    """Parse JSON listings from content."""
    try:
        listings = json.loads(content)
        if isinstance(listings, list):
            return listings
        else:
            logger.error("Parsed JSON is not a list. Content was: " + content)
            return []
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON. Content: " + content)
        return []

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int):
    bed_text = f"{bedrooms}-bedroom" if bedrooms else ""
    budget_text = f"around {price_max} AED" if price_max else "within a reasonable price range"

    user_prompt = (
        f"Find available {bed_text} {property_type}(s) in {location} {budget_text}. "
        "Return ONLY JSON as an array of objects, each with keys: name, link, price, features. "
        "If no listings found, return []."
    )
    content = call_perplexity(user_prompt)
    return parse_listings(content)

def find_general_commentary(location: str, property_type: str, bedrooms: int, price_max: int):
    bed_text = f"{bedrooms}-bedroom" if bedrooms else ""
    budget_text = f"around {price_max} AED" if price_max else "a given price range"

    user_prompt = (
        f"Provide a JSON array with a few recommended {bed_text} {property_type}(s) in {location} {budget_text}, "
        "or general commentary if listings are unavailable. Each element: {\"name\":\"...\",\"link\":\"...\",\"price\":\"...\",\"features\":\"...\"}. "
        "If no specific listings found, return a single-element array with a short commentary in 'features'."
    )
    content = call_perplexity(user_prompt)
    # Try parsing as listings; if empty or fail, return commentary as string
    try:
        results = json.loads(content)
        if isinstance(results, list) and len(results) > 0:
            # Format as a short commentary block
            commentary = "\nFrom my online search:\n"
            for i, r in enumerate(results, start=1):
                name = r.get("name", "A property")
                link = r.get("link", "#")
                price = r.get("price", "N/A")
                features = r.get("features", "")
                commentary += f"\nOption {i}: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return commentary
        else:
            return "\nI couldn’t find listings, but online sources suggest exploring different platforms for more options."
    except json.JSONDecodeError:
        return "\nI couldn’t find relevant listings online at the moment."

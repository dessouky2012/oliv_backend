import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int):
    if not PERPLEXITY_API_KEY:
        logger.warning("PERPLEXITY_API_KEY not set.")
        return []

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    bed_text = f"{bedrooms}-bedroom" if bedrooms else ""
    budget_text = f"around {price_max} AED" if price_max else "a reasonable price"
    query = (
        f"Find 3 current, real, and active listings for a {bed_text} {property_type} in {location} {budget_text}. "
        "Focus on recent listings from bayut.com or propertyfinder.ae only. Links must be directly to the listing pages. "
        "If no suitable matches found, return an empty JSON list. "
        "Return ONLY JSON like: "
        '[{"name":"Property Name","link":"https://...","price":"XXX AED","features":"Short description"}].'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You fetch real property listings. Return ONLY JSON, no extra text."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            try:
                listings = json.loads(content)
                # Validate domain
                valid_listings = []
                for lst in listings:
                    link = lst.get("link", "").lower()
                    if "bayut.com" in link or "propertyfinder.ae" in link:
                        valid_listings.append(lst)
                return valid_listings
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from Perplexity.")
                return []
        else:
            logger.error(f"Perplexity returned {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Exception calling Perplexity: {e}")
        return []

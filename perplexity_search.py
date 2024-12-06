import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "llama-3.1-sonar-small-128k-online"  # or another permitted model per Perplexity's docs

def call_perplexity(query: str) -> str:
    """Call the Perplexity API with the given query and return raw response content (possibly with code fences)."""
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
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant. Return ONLY JSON as requested. "
                    "No commentary or explanations. If no listings match, return []."
                )
            },
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

def clean_json_content(content: str) -> str:
    """Remove code fences and extra formatting from the returned content to ensure valid JSON."""
    cleaned = content.replace("```json", "").replace("```", "").strip()
    return cleaned

def parse_listings(content: str):
    """Parse JSON listings from content."""
    cleaned_content = clean_json_content(content)
    try:
        listings = json.loads(cleaned_content)
        if isinstance(listings, list):
            return listings
        else:
            logger.error("Parsed JSON is not a list. Content was: " + cleaned_content)
            return []
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON. Content: " + cleaned_content)
        return []

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int, exact_location: str = None):
    bed_text = f"{bedrooms}-bedroom" if bedrooms else "studio"
    budget_text = f"around {price_max} AED" if price_max else "within a reasonable price range"

    if exact_location:
        user_prompt = (
            f"Find currently available {bed_text} {property_type}(s) specifically in '{exact_location}', Dubai {budget_text}. "
            "Return ONLY listings that clearly state the property is located in this exact building/area name. "
            "If no exact matches, return []. "
            "Format as a JSON array of objects with keys: name, link, price, features. No extra text."
        )
    else:
        user_prompt = (
            f"Find currently available {bed_text} {property_type}(s) in {location}, Dubai {budget_text}. "
            "Return ONLY JSON as an array of objects: name, link, price, features. If none found, return []."
        )

    content = call_perplexity(user_prompt)
    return parse_listings(content)

def find_general_commentary(location: str, property_type: str, bedrooms: int, price_max: int):
    bed_text = f"{bedrooms}-bedroom" if bedrooms else "studio"
    budget_text = f"around {price_max} AED" if price_max else "a given price range"

    user_prompt = (
        f"Provide a JSON array of {bed_text} {property_type}(s) in {location}, Dubai {budget_text}, "
        "or a single-element array with general commentary in 'features' if no listings are found. "
        "Format as [{\"name\":\"...\",\"link\":\"...\",\"price\":\"...\",\"features\":\"...\"}]. If no listings, return a single-element array describing the situation. No extra text."
    )

    content = call_perplexity(user_prompt)
    # Try parsing as listings; if empty or fail, return commentary as a string
    cleaned_content = clean_json_content(content)
    try:
        results = json.loads(cleaned_content)
        if isinstance(results, list) and len(results) > 0:
            # Format as a short commentary block
            commentary = "\nFrom my online lookup:\n"
            for i, r in enumerate(results, start=1):
                name = r.get("name", "A property")
                link = r.get("link", "#")
                price = r.get("price", "N/A")
                features = r.get("features", "")
                commentary += f"\nOption {i}: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return commentary
        else:
            return "\nI couldn’t find relevant listings at the moment."
    except json.JSONDecodeError:
        return "\nI couldn’t find relevant listings at the moment."

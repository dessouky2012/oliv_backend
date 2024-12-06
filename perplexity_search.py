import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "llama-3.1-sonar-large-128k-online"  # Adjust if needed per Perplexity's available models

def call_perplexity(query: str) -> str:
    """Call the Perplexity API with the given query and return raw response content."""
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
                    "You are a helpful real estate assistant. "
                    "Return ONLY a JSON array of listings with no extra text. "
                    "If no matches, return an empty array []."
                )
            },
            {"role": "user", "content": query}
        ],
        "max_tokens": 900,
        "temperature": 0.0
    }

    try:
        logger.info(f"Sending request to Perplexity with query: {query}")
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
    # If any code fences appear, remove them
    content = content.replace("```json", "").replace("```", "").strip()
    return content

def parse_listings(content: str):
    cleaned_content = clean_json_content(content)
    try:
        listings = json.loads(cleaned_content)
        if isinstance(listings, list):
            return listings
        else:
            logger.error("JSON parsed is not a list. Content was: " + cleaned_content)
            return []
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON. Content: " + cleaned_content)
        return []

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int, exact_location: str = None):
    """
    Query Perplexity for listings. 
    If exact_location is given (e.g., "Marina View Tower"), we explicitly ask for listings in that building.
    Otherwise, just query by area.
    """
    bed_text = f"{bedrooms}-bedroom" if bedrooms else "studio"
    if exact_location:
        # Natural prompt focusing on the exact building
        user_prompt = (
            f"Please find current {bed_text} {property_type}(s) for sale in '{exact_location}' in Dubai. "
            "Only return listings actually in this specific building. "
            "Each listing should be an object with keys: name, link, price, features. "
            "If no exact matches, return []."
        )
    else:
        # Query by general area
        budget_text = f"around {price_max} AED" if price_max else "a suitable price range"
        user_prompt = (
            f"Find currently available {bed_text} {property_type}(s) in {location} Dubai {budget_text}. "
            "Return a JSON array of objects with: name, link, price, features. "
            "If none found, return []."
        )

    content = call_perplexity(user_prompt)
    return parse_listings(content)

def find_general_commentary(location: str, property_type: str, bedrooms: int, price_max: int):
    """
    Provide general commentary if no direct listings are found, in JSON form.
    We'll try a friendly prompt that encourages Perplexity to give some indicative options or a single commentary object.
    """
    bed_text = f"{bedrooms}-bedroom" if bedrooms else "studio"
    budget_text = f"around {price_max} AED" if price_max else "an affordable range"
    user_prompt = (
        f"Find a few {bed_text} {property_type}(s) in {location}, Dubai {budget_text}, "
        "or if none available, return a single object with a 'features' key summarizing the situation. "
        "Return as a JSON array, no extra text."
    )

    content = call_perplexity(user_prompt)
    # Try parsing as listings
    cleaned_content = clean_json_content(content)
    try:
        results = json.loads(cleaned_content)
        if isinstance(results, list) and len(results) > 0:
            # Format them into commentary
            commentary = "\nFrom my lookup:\n"
            for i, r in enumerate(results, start=1):
                name = r.get("name", "A property")
                link = r.get("link", "#")
                price = r.get("price", "N/A")
                features = r.get("features", "")
                commentary += f"\nOption {i}: {name}\nPrice: {price}\nFeatures: {features}\nLink: {link}\n"
            return commentary
        else:
            return "\nIt seems I couldn’t locate suitable listings at the moment."
    except json.JSONDecodeError:
        return "\nI’m sorry, I couldn’t parse the listings right now."

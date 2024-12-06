import os
import requests

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def find_listings(location: str, property_type: str, bedrooms: int, price_max: int):
    """
    Call Perplexity to find listings.
    We instruct Perplexity to return JSON with a list of properties.
    Each property: {"name": "...", "link": "...", "price": "...", "features": "..."}
    If none found, return an empty list.
    """

    if not PERPLEXITY_API_KEY:
        return []

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Prompt Perplexity for listings in JSON format
    query = (
        f"Find up to 3 actual listings for a {bedrooms if bedrooms else ''}-bedroom {property_type} "
        f"in {location} around {price_max if price_max else 'a reasonable'} AED. "
        "Return the response as a JSON array of objects like this: "
        '[{"name":"Property Name","link":"URL","price":"XXX AED","features":"Short description"}]. '
        "No extra text, only JSON."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a data retrieval assistant. Return only the requested JSON."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Extract content
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"].get("content","").strip()
                # Attempt to parse JSON
                try:
                    listings = json.loads(content)
                    if isinstance(listings, list):
                        return listings
                except:
                    # If parsing fails, return empty
                    return []
        return []
    except:
        return []

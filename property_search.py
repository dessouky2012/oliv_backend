# property_search.py
import os
from serpapi import GoogleSearch

# Load the API key from environment variables
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "YOUR_SERPAPI_KEY")

def search_properties(location: str, property_type: str, bedrooms: int, budget: float):
    # Build a query string from the parameters
    query = f"{bedrooms} bedroom {property_type} in {location} under {int(budget)} AED property for sale"

    params = {
        "engine": "google",
        "q": query,
        "hl": "en",        # language
        "gl": "ae",        # geo location (ae = United Arab Emirates)
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    listings = []
    # SerpAPI returns a structured JSON. "organic_results" are the main search results.
    if "organic_results" in results:
        for item in results["organic_results"]:
            listing = {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "displayed_link": item.get("displayed_link")
            }
            listings.append(listing)

    return listings

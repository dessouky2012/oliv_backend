# perplexity_search.py
import os
import requests

# Load your Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")

# Define the Perplexity model you want to use (check Perplexity’s docs for available models)
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def ask_perplexity(query: str) -> dict:
    """
    Sends the user's query to Perplexity's Chat Completions endpoint and returns the assistant's reply.
    We have improved the system prompt to encourage structured, actionable responses (e.g., listings),
    and to not direct the user to other human agents. We ask for helpful, link-rich output if possible.
    """
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    system_prompt = (
        "You are Oliv's research assistant, specializing in Dubai real estate. "
        "You help find listings, market data, and property information. "
        "If the user requests listings, provide structured details (e.g., bullet points) and direct links "
        "to listings on Bayut or Propertyfinder if found. If no exact links are found, "
        "give the closest available helpful data. The user should not be directed to human agents—Oliv wants "
        "to handle leads herself. You can mention other online platforms, but not hand the user off to a human."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Assuming at least one choice is returned
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer}
        else:
            # If not 200, return error info
            return {"error": f"Non-200 status code: {response.status_code}", "full_response": response.text}
    except Exception as e:
        return {"error": str(e)}

# Remove the example usage block in production
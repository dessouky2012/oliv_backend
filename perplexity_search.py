# perplexity_search.py
import os
import requests

# Load your Perplexity API key from environment variables
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Define the Perplexity model you want to use (check Perplexity’s docs for available models)
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def ask_perplexity(query: str) -> dict:
    """
    Sends the user's query to Perplexity's Chat Completions endpoint and returns the assistant's reply.
    
    Requirements:
    - Provide structured details if listings are requested.
    - Provide links if possible.
    - Avoid directing to human agents. Oliv handles leads herself.
    - If no listings or data found, return a helpful fallback message.
    
    Returns a dict:
    - On success: {"answer": "..."} containing Perplexity's answer.
    - On error: {"error": "..."} containing error details.
    """

    if not PERPLEXITY_API_KEY:
        return {"error": "Perplexity API key not set. Please set PERPLEXITY_API_KEY in environment variables."}

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    system_prompt = (
        "You are Oliv, a research assistant, specializing in Dubai real estate. "
        "The user may ask for listings, pricing details, or market info. "
        "If the user requests listings, provide factual details and direct Bayut or Propertyfinder links if found. "
        "Never direct the user to human agents, Oliv handles leads herself. "
        "Focus on Dubai properties, return relevant listings or data. "
        "If no exact listings found, provide best possible related info."
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
            # Check if 'choices' and 'message' and 'content' are present
            if "choices" in data and len(data["choices"]) > 0 and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
                answer = data["choices"][0]["message"]["content"].strip()
                if answer:
                    return {"answer": answer}
                else:
                    return {"answer": "I couldn't find specific details at this moment."}
            else:
                return {"answer": "No usable answer returned by Perplexity."}
        else:
            return {"error": f"Non-200 status code from Perplexity: {response.status_code}", "full_response": response.text}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
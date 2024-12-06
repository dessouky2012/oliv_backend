import os
import requests

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
MODEL_NAME = "llama-3.1-sonar-large-128k-online"

def ask_perplexity(query: str) -> dict:
    if not PERPLEXITY_API_KEY:
        return {"error": "Perplexity API key not set."}

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    system_prompt = (
        "You are Oliv, a research assistant specializing in Dubai real estate. "
        "User may ask for listings, pricing details, or market info. "
        "If user requests listings, provide factual details and direct links if possible. "
        "No human agents. Focus on Dubai properties. If no listings, give best possible related info."
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
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"].get("content", "").strip()
                if content:
                    return {"answer": content}
            return {"answer": "No usable answer returned by Perplexity."}
        else:
            return {"error": f"Non-200 status: {response.status_code}", "full_response": response.text}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}
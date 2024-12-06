import os
import openai
from nlu_integration import interpret_user_query
from predict import predict_price

openai.api_key = os.getenv("OPENAI_API_KEY")

system_message = {
    "role": "system",
    "content": (
        "You are Oliv, an AI real-estate assistant for Dubai. "
        "If asked about price, you can call predict_price. Otherwise, respond naturally."
    )
}

messages = [system_message]

print("Oliv is ready to chat! Type your message below. Type 'quit' to exit.\n")

while True:
    user_input = input("User: ")
    if user_input.strip().lower() == "quit":
        print("Exiting.")
        break

    user_data = interpret_user_query(user_input)
    location = user_data.get("location")
    property_type = user_data.get("property_type")
    bedrooms = user_data.get("bedrooms")
    budget = user_data.get("budget")

    messages.append({"role": "user", "content": user_input})

    if location and property_type:
        predicted = predict_price({
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,
            "BEDROOMS": bedrooms if bedrooms else 1,
            "PARKING": 1
        })

        if predicted:
            assistant_reply = f"The estimated price for a {bedrooms if bedrooms else 'studio'} {property_type} in {location} is about {int(predicted):,} AED."
        else:
            assistant_reply = "I’m sorry, I don’t have enough data to estimate that price range."
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        assistant_reply = response.choices[0].message.content

    print("Oliv:", assistant_reply, "\n")
    messages.append({"role": "assistant", "content": assistant_reply})

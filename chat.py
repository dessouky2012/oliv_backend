# chat.py
import os
import openai
from nlu_integration import interpret_user_query
from predict import predict_price

# Instead of hardcoding the key, read it from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a system message that sets the role of Oliv
system_message = {
    "role": "system",
    "content": (
        "You are Oliv, an AI real-estate assistant for Dubai. "
        "You can use historical Dubai Land Department data to tell the user if a property is overpriced or underpriced. "
        "You can also provide price predictions for properties based on location, property type, and other details. "
        "If the user asks about a property price or wants to know if something is overpriced, use the predict_price function (already integrated) to help. "
        "If the user asks for something else, respond naturally. "
        "Be helpful, polite, and knowledgeable."
    )
}

# Initialize conversation
messages = [system_message]

print("Oliv is ready to chat! Type your message below. Type 'quit' to exit.\n")

while True:
    user_input = input("User: ")
    if user_input.strip().lower() == "quit":
        print("Exiting conversation. Goodbye!")
        break

    # Interpret the user query to get structured data
    user_data = interpret_user_query(user_input)
    # Expected structure: { "intent": ..., "location": ..., "property_type": ..., "bedrooms": ..., "budget": ... }

    # Add the user input to the conversation
    messages.append({"role": "user", "content": user_input})

    # Extract parameters
    location = user_data.get("location")
    property_type = user_data.get("property_type")
    bedrooms = user_data.get("bedrooms")
    budget = user_data.get("budget")

    # Decide whether we can do a price prediction
    if location and property_type:
        # Prepare data for predict_price()
        new_data = {
            "AREA_EN": location,
            "PROP_TYPE_EN": property_type,
            "ACTUAL_AREA": 100,  # default area, adjust if needed
            "BEDROOMS": bedrooms if bedrooms else 1,
            "PARKING": 1
        }

        predicted = predict_price(new_data)

        # If user provided a budget, compare with predicted price
        if budget and isinstance(budget, (int, float)):
            if budget > predicted:
                assistant_reply = (
                    f"The estimated price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                    f"in {location} is around {int(predicted):,} AED. "
                    f"Your mentioned price of {int(budget):,} AED seems higher than the average."
                )
            else:
                assistant_reply = (
                    f"The estimated price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                    f"in {location} is around {int(predicted):,} AED. "
                    f"Your mentioned price of {int(budget):,} AED is fair or even below the average."
                )
        else:
            # No budget given, just provide predicted price
            assistant_reply = (
                f"The estimated market price for a {bedrooms if bedrooms else 'Studio'}-bedroom {property_type} "
                f"in {location} is around {int(predicted):,} AED."
            )
    else:
        # Not enough info for a price prediction, just respond generally using gpt-4o
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using gpt-4o as requested
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        assistant_reply = response.choices[0].message.content

    # Print the assistant's reply
    print("Oliv:", assistant_reply, "\n")

    # Add assistant reply to conversation
    messages.append({"role": "assistant", "content": assistant_reply})
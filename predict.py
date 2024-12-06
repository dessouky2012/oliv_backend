import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Load model and columns
try:
    with open("pricing_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    logger.error("pricing_model.pkl not found. Price predictions won't work.")
    model = None

try:
    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
except FileNotFoundError:
    logger.error("training_columns.pkl not found. Price predictions won't work.")
    training_columns = []

def predict_price(new_data: dict):
    """Predict property price based on input features."""
    if model is None or not training_columns:
        return None

    # Default values for missing fields
    new_data['AREA_EN'] = new_data.get('AREA_EN', 'UNKNOWN_AREA')
    new_data['PROP_TYPE_EN'] = new_data.get('PROP_TYPE_EN', 'UNKNOWN_TYPE')
    new_data['ACTUAL_AREA'] = new_data.get('ACTUAL_AREA', 80)
    new_data['BEDROOMS'] = new_data.get('BEDROOMS', 1)
    new_data['PARKING'] = new_data.get('PARKING', 1)

    input_df = pd.DataFrame([new_data])
    input_df = pd.get_dummies(input_df, columns=["AREA_EN","PROP_TYPE_EN"], dummy_na=True)

    # Add missing columns
    for col in set(training_columns) - set(input_df.columns):
        input_df[col] = 0

    # Reorder columns
    input_df = input_df[training_columns]

    try:
        predicted_price = model.predict(input_df)[0]
        return predicted_price
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/', methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            lead_time = int(request.form['lead_time'])
            no_of_special_request = int(request.form['no_of_special_request'])
            avg_price_per_room = float(request.form['avg_price_per_room'])
            arrival_month = int(request.form['arrival_month'])
            arrival_date = int(request.form['arrival_date'])
            market_segment_type = int(request.form['market_segment_type'])
            no_of_week_nights = int(request.form['no_of_week_nights'])
            no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
            type_of_meal_plan = int(request.form['type_of_meal_plan'])
            room_type_reserved = int(request.form['room_type_reserved'])

            # Arrange features in the expected order and reshape for prediction
            features = np.array([
                lead_time,
                no_of_special_request,
                avg_price_per_room,
                arrival_month,
                arrival_date,
                market_segment_type,
                no_of_week_nights,
                no_of_weekend_nights,
                type_of_meal_plan,
                room_type_reserved
            ]).reshape(1, -1)

            prediction = loaded_model.predict(features)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
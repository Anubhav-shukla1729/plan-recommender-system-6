from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("indian_mobile_plans_classified.csv")

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Recommend route
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    message = ""
    plans = []

    if request.method == 'POST':
        try:
            # Get inputs from form
            price = request.form.get('price', type=float)
            validity = request.form.get('validity', type=int)
            data_per_day = request.form.get('data', type=float)
            selected_category = request.form.get('category')

            # Start with the full dataset
            filtered = df.copy()

            # Filter by category if selected
            if selected_category:
                filtered = filtered[filtered['plan_class'] == selected_category]

            # Apply price filter if provided
            if price is not None:
                filtered = filtered[
                    (filtered['price'] >= price * 0.8) & (filtered['price'] <= price * 1.2)
                ]
            
            # Apply validity filter if provided
            if validity is not None:
                filtered = filtered[
                    (filtered['validity_days'] >= validity * 0.8) & (filtered['validity_days'] <= validity * 1.2)
                ]
            
            # Apply data per day filter if provided
            if data_per_day is not None:
                filtered = filtered[
                    (filtered['data_per_day'] >= data_per_day * 0.8) & (filtered['data_per_day'] <= data_per_day * 1.2)
                ]

            # If category was selected, use sorted plans
            if selected_category:
                plans = filtered.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                message = f"Showing best {selected_category} plans"
            else:
                # If no category selected, use model prediction
                if price is not None and validity is not None and data_per_day is not None:
                    input_features = [[price, validity, data_per_day]]
                    prediction = model.predict(input_features)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]

                    matching_plans = filtered[filtered['plan_class'] == predicted_label]
                    plans = matching_plans.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                    message = f"Predicted Category: {predicted_label}"
                else:
                    # If some inputs are missing, just show the best available plans
                    plans = filtered.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                    message = "Showing best available plans based on input."

            # If no plans match, show a message
            if not plans:
                message = "No plans found matching your criteria."

        except Exception as e:
            message = f"Error: {e}"

    return render_template("recommend.html", message=message, plans=plans)

# About route
@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)

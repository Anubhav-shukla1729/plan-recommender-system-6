from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("indian_mobile_plans_classified.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    plans = []

    if request.method == 'POST':
        try:
            price = request.form.get('price', type=float)
            validity = request.form.get('validity', type=int)
            data_per_day = request.form.get('data', type=float)
            selected_category = request.form.get('category')

            # If a category is selected, prioritize it
            if selected_category:
                filtered = df[df['plan_class'] == selected_category]

                # Allow a 20% fluctuation around input price, validity, and data
                if price:
                    filtered = filtered[
                        (filtered['price'] >= price * 0.8) & (filtered['price'] <= price * 1.2)
                    ]
                if validity:
                    filtered = filtered[
                        (filtered['validity_days'] >= validity * 0.8) & (filtered['validity_days'] <= validity * 1.2)
                    ]
                if data_per_day:
                    filtered = filtered[
                        (filtered['data_per_day'] >= data_per_day * 0.8) & (filtered['data_per_day'] <= data_per_day * 1.2)
                    ]

                plans = filtered.sort_values(by='price_per_GB').head(4).to_dict(orient='records')

                message = f"Showing best {selected_category} plans"
            else:
                # Use model for prediction
                input_features = [[price, validity, data_per_day]]
                prediction = model.predict(input_features)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Filter top matching plans
                matching_plans = df[df['plan_class'] == predicted_label]
                matching_plans = matching_plans.sort_values(by='price_per_GB').head(4)

                plans = matching_plans.to_dict(orient='records')
                message = f"Predicted Category: {predicted_label}"
        except Exception as e:
            message = f"Error: {e}"

    return render_template("index.html", message=message, plans=plans)

if __name__ == '__main__':
    app.run(debug=True)

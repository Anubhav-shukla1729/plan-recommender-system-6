from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("indian_mobile_plans_classified.csv")

# Predefined speeds for ISPs (this is a sample, you can modify as needed)
isp_speeds = {
    'Airtel': {'download': 25, 'upload': 10},   # Example: 25 Mbps download and 10 Mbps upload
    'Jio': {'download': 20, 'upload': 8},
    'BSNL': {'download': 15, 'upload': 5},
    'Vodafone': {'download': 18, 'upload': 6}
}

# Best ISP based on speed
best_isp = max(isp_speeds, key=lambda k: isp_speeds[k]['download'])

# Home route
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/speedtest', methods=['GET', 'POST'])
def speedtest_view():
    message = ""
    best_isp_speed = isp_speeds[best_isp]

    if request.method == 'POST':
        # Get the selected ISP and pincode from the form
        selected_isp = request.form.get('isp')
        pincode = request.form.get('pincode')

        if selected_isp in isp_speeds:
            # Get the speed for the selected ISP
            selected_speed = isp_speeds[selected_isp]

            # Compare with the best ISP
            if selected_isp == best_isp:
                message = f"Congratulations! Your ISP, {selected_isp}, is the best in your area with {selected_speed['download']} Mbps download speed."
            else:
                message = f"Your ISP, {selected_isp}, has {selected_speed['download']} Mbps download speed. " \
                          f"However, the best ISP is {best_isp} with {best_isp_speed['download']} Mbps download speed. Consider switching to {best_isp} for faster speeds."
        else:
            message = "Invalid ISP selection. Please choose a valid ISP."

    return render_template("speed.html", message=message, best_isp=best_isp, best_isp_speed=best_isp_speed, isp_speeds=isp_speeds)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    message = ""
    plans = []
    isp_speed_info = None
    recommended_isp = None
    selected_isp = None

    if request.method == 'POST':
        try:
            # Get inputs from form
            price = request.form.get('price', type=float)
            validity = request.form.get('validity', type=int)
            data_per_day = request.form.get('data', type=float)
            selected_category = request.form.get('category')
            pincode = request.form.get('pincode')
            selected_isp = request.form.get('isp')

            # Get ISP speed info
            if selected_isp in isp_speeds:
                isp_speed_info = isp_speeds[selected_isp]
                # Recommend the best ISP if the selected ISP is not the fastest
                if selected_isp != best_isp:
                    recommended_isp = best_isp

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

    return render_template("recommend.html", message=message, plans=plans, isp_speed_info=isp_speed_info,
                           selected_isp=selected_isp, recommended_isp=recommended_isp)

@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)

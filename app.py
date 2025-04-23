from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objs as go

app = Flask(__name__)

# === MOBILE PLAN SETUP ===
model_mobile = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
df_mobile = pd.read_csv("indian_mobile_plans_classified.csv")

isp_speeds = {
    'Airtel': {'download': 25, 'upload': 10},
    'Jio': {'download': 20, 'upload': 8},
    'BSNL': {'download': 15, 'upload': 5},
    'Vodafone': {'download': 18, 'upload': 6}
}
best_isp = max(isp_speeds, key=lambda k: isp_speeds[k]['download'])

# === BROADBAND SETUP ===
model_bb = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
features_bb = joblib.load('features.pkl')
df_bb = pd.read_csv('cleaned_broadband_plans.csv')
region_cols = [f for f in features_bb if f.startswith('Region_')]

# ================= ROUTES ================= #

@app.route('/')
def home():
    return render_template("index.html")

# ---------- ISP Speed Comparison ----------
@app.route('/speedtest', methods=['GET', 'POST'])
def speedtest_view():
    message = ""
    best_isp_speed = isp_speeds[best_isp]

    if request.method == 'POST':
        selected_isp = request.form.get('isp')
        pincode = request.form.get('pincode')

        if selected_isp in isp_speeds:
            selected_speed = isp_speeds[selected_isp]
            if selected_isp == best_isp:
                message = f"Congratulations! Your ISP, {selected_isp}, is the best in your area with {selected_speed['download']} Mbps download speed."
            else:
                message = f"Your ISP, {selected_isp}, has {selected_speed['download']} Mbps download speed. " \
                          f"However, the best ISP is {best_isp} with {best_isp_speed['download']} Mbps. Consider switching."
        else:
            message = "Invalid ISP selection. Please choose a valid ISP."

    return render_template("speed.html", message=message, best_isp=best_isp, best_isp_speed=best_isp_speed, isp_speeds=isp_speeds)

# ---------- Mobile Plan Recommendation ----------
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    message = ""
    plans = []
    isp_speed_info = None
    recommended_isp = None
    selected_isp = None

    if request.method == 'POST':
        try:
            price = request.form.get('price', type=float)
            validity = request.form.get('validity', type=int)
            data_per_day = request.form.get('data', type=float)
            selected_category = request.form.get('category')
            pincode = request.form.get('pincode')
            selected_isp = request.form.get('isp')

            if selected_isp in isp_speeds:
                isp_speed_info = isp_speeds[selected_isp]
                if selected_isp != best_isp:
                    recommended_isp = best_isp

            filtered = df_mobile.copy()

            if selected_category:
                filtered = filtered[filtered['plan_class'] == selected_category]

            if price:
                filtered = filtered[(filtered['price'] >= price * 0.8) & (filtered['price'] <= price * 1.2)]

            if validity:
                filtered = filtered[(filtered['validity_days'] >= validity * 0.8) & (filtered['validity_days'] <= validity * 1.2)]

            if data_per_day:
                filtered = filtered[(filtered['data_per_day'] >= data_per_day * 0.8) & (filtered['data_per_day'] <= data_per_day * 1.2)]

            if selected_category:
                plans = filtered.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                message = f"Showing best {selected_category} plans"
            else:
                if price and validity and data_per_day:
                    input_features = [[price, validity, data_per_day]]
                    prediction = model_mobile.predict(input_features)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    matching_plans = filtered[filtered['plan_class'] == predicted_label]
                    plans = matching_plans.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                    message = f"Predicted Category: {predicted_label}"
                else:
                    plans = filtered.sort_values(by='price_per_GB').head(4).to_dict(orient='records')
                    message = "Showing best available plans."

            if not plans:
                message = "No plans found matching your criteria."

        except Exception as e:
            message = f"Error: {e}"

    return render_template("recommend.html", message=message, plans=plans, isp_speed_info=isp_speed_info,
                           selected_isp=selected_isp, recommended_isp=recommended_isp)




# ---------- Broadband Plan Recommendation ----------
@app.route('/broadband', methods=['GET', 'POST'])
def broadband():
    plans = []
    if request.method == 'POST':
        try:
            # Extract input values from the form, defaults to empty string if not provided
            price = request.form.get('price')
            validity = request.form.get('validity')
            speed = request.form.get('speed')
            region = request.form.get('region')

            # Prepare the input data with default values for missing fields
            input_data = {
                'Price (₹)': float(price) if price else 0,
                'Validity (days)': float(validity) if validity else 0,
                'Speed (Mbps)': float(speed) if speed else 0,
            }

            # Create a region vector, where 1 is for the selected region, and 0 for others
            region_vector = [1 if f'Region_{region}' == col else 0 for col in region_cols] if region else [0] * len(region_cols)

            # Combine input data and region vector
            input_vector = list(input_data.values()) + region_vector

            # If no input data is provided, provide the default recommendation
            if not any(input_vector):
                message = "No input provided. Showing all broadband plans."
                plans = df_bb.to_dict(orient='records')
            else:
                # Normalize the input vector
                scaled_input = scaler.transform([input_vector])

                # Get the closest broadband plans based on the model's prediction
                distances, indices = model_bb.kneighbors(scaled_input)

                # Retrieve the corresponding broadband plans
                plans = df_bb.iloc[indices[0]].to_dict(orient='records')
                message = "Showing recommended broadband plans based on your input."

        except Exception as e:
            print(f"Error: {e}")
            plans = []
            message = f"An error occurred: {e}"

    else:
        message = "Fill in any combination of inputs to get recommendations."

    return render_template("broadband.html", plans=plans, message=message, regions=sorted(df_bb['Region'].unique()))



# ---------- About ----------
@app.route('/about')
def about():
    return render_template("about.html")

# ================= RUN APP ================= #
if __name__ == '__main__':
    app.run(debug=True)

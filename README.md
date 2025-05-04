
# 📶 Telecom Plan Recommender System

**Live Website**: [https://plan-recommender-system-6.onrender.com]

This project is a comprehensive telecom solution that integrates:
- ✅ Mobile Recharge Plan Recommender
- ✅ Broadband Plan Recommender
- ✅ Internet Speed Checker
- ✅ Churn Prediction via Machine Learning (XGBoost)

It is built using a **Flask** backend, a **Bootstrap**-powered responsive frontend, and ML-powered intelligence. The system is trained on real-world Indian ISP data (Jio, Airtel, Vi, BSNL) and is designed to assist users in selecting the most suitable telecom plans based on personalized preferences.

---

## 🌐 Features

### 🔹 1. Mobile Plan Recommender
- Predicts user needs using ML or lets the user select a category.
- Categories: Heavy Data, Only Calls, Calls + Data, SMS Only, Long Term, Short Term.
- Filters plans by user’s expected price, data usage, and validity.
- Supports multiple ISPs (Jio, Airtel, Vi, BSNL).

### 🔹 2. Broadband Plan Recommender
- Suggests broadband plans based on:
  - Monthly budget
  - Preferred speed (Mbps)
  - Usage pattern (Streaming, Gaming, Office Work, etc.)
- Supports providers like JioFiber, Airtel Xstream, BSNL Bharat Fiber, and ACT.

### 🔹 3. Speed Checker
- Measures real-time internet speed using Python’s `speedtest` module.
- Displays download and upload speeds in Mbps.
- Helps users make informed decisions about switching plans.

### 🔹 4. Churn Prediction (Internal Module)
- Trained XGBoost model analyzes:
  - Usage behavior
  - Complaint history
  - Recharge patterns
- Helps ISPs predict customer churn and offer retention plans.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (Bootstrap), JS
- **ML Models**: XGBoost, KMeans (for clustering)
- **Hosting**: Render (for website), Colab (for initial model training)

---

## 🧪 How to Run Locally

1. **Clone the Repo**  
   ```bash
   git clone https://github.com/Anubhav-shukla1729/plan-recommender-system-6.git
   cd plan-recommender-system-6
2. **Install dependecies**
   ```bash
   pip install -r requirements.txt
3. **Run Flask app**
   ```bash
   python app.py
**Open app on localhost**
  Open http://localhost:5000 in your browser.

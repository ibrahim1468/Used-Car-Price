🚗 Car Price Prediction App
This is a simple and interactive Streamlit app that predicts the price of a car based on user inputs. It uses a trained machine learning model (Gradient Boosting) and handles both numerical and categorical features.

📦 Features
Predicts used car prices based on:

Brand & Model

Model Year, Mileage (km), Engine HP & Litres

Transmission, Fuel Type, Accidents

Exterior and Interior Colors

Pre-trained ML model (Gradient Boosting Regressor)

Categorical encoding & feature scaling

Clean user interface (no need for file upload)

🧠 Model Details
Best Model: GradientBoostingRegressor

R² Score: ~0.88

Training Data: Real-world car data with preprocessing

Libraries Used: scikit-learn, xgboost, lightgbm, pandas, streamlit

📁 Project Structure
bash
Copy
Edit
📦 Car Price Prediction
├── Car_price.py              # Main Streamlit app
├── car_price_model.pkl       # Trained ML model
├── encoded_columns.pkl       # One-hot encoder column structure
├── model_features.pkl        # Final feature list used in the model
├── scaler.pkl                # Scaler object for numerical features
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
▶️ How to Run
Locally:

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run Car_price.py
☁️ Deploying on Streamlit Cloud
To deploy:

Push all files to a GitHub repository.

Go to https://streamlit.io/cloud.

Link your repo and click Deploy.

Done!

Make sure your repo includes:

Car_price.py

requirements.txt

All .pkl files

📌 Notes
This app doesn't require users to upload data.

All preprocessing is handled internally.

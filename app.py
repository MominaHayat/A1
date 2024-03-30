from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LinearRegression


app = Flask(_name_)


# Load dataset into a Pandas DataFrame
csv_file = "./dataset/LR.csv"
df = pd.read_csv(csv_file)

# Extract features (X) and target variable (y) from dataset
X = df[['X']]  # Ensure column name is 'X'
y = df['Y']    # Ensure column name is 'Y'

# Initialize and fit linear regression model
model = LinearRegression()
model.fit(X, y)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request and prepare for prediction
    data = request.json
    X_new = pd.DataFrame(data['X'], columns=['X'])

    # Make predictions
    y_pred = model.predict(X_new)

    # Return predictions
    return jsonify({'predictions': y_pred.tolist()})


if _name_ == '_main_':
    app.run(debug=True)

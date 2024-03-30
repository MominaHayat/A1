from flask import Flask, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset into a Pandas DataFrame
csv_file = "./dataset/LR.csv"
df = pd.read_csv(csv_file)

# Check the structure and contents of the dataset
print(df.head())  # Print the first few rows to verify the column names and data

# Extract the features (X) and target variable (y) from the dataset
X = df[['X']]  # Correct column name to 'X'
y = df['Y']    # Correct column name to 'Y'

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions using the trained model
y_pred = model.predict(X)

# Plot the dataset and the linear regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression on LR Dataset')
plt.legend()
plt.show()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    X_new = pd.DataFrame(data['X'], columns=['X'])  # Assuming input data is in JSON format with 'X' key

    # Make predictions
    y_pred = model.predict(X_new)

    # Return predictions
    return jsonify({'predictions': y_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

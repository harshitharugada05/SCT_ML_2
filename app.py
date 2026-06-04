from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(
    open("customer_segmentation.pkl", "rb")
)

cluster_names = {
    0: "Premium Customers",
    1: "Regular Customers",
    2: "Budget Customers",
    3: "High Potential Customers",
    4: "Luxury Shoppers"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():

    income = float(request.form['income'])
    spending = float(request.form['spending'])

    cluster = model.predict(
        np.array([[income, spending]])
    )[0]

    result = cluster_names.get(
        cluster,
        f"Cluster {cluster}"
    )

    return render_template(
    'results.html',
    cluster=result
)

if __name__ == '__main__':
    app.run(debug=True)
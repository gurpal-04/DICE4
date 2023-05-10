from flask import Flask, render_template, request

model = pickle.load(open('plant_detection_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
  return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
  pred = model.predict(request.form['pic'])
  return render_template('output.html', data=pred)

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=81, debug=True)

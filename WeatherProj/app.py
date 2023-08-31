from flask import Flask, render_template
import request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('weather.html')

@app.route('/search', methods=['POST'])
def search():
    city = request.form.get('city')
    # Fetch weather data for the entered city
    # ...
    return render_template('results.html')

if __name__ == '__main__':
    app.run()
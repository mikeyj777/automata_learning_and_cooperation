from flask import Flask, jsonify
from flask_cors import CORS

from ps import ps_001

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)
CORS(app, resources={
    r"/*": {  # This specifically matches your API routes
        "origins": ["http://localhost:3000", "http://emo.riskspace.net"],
        "methods": ["GET", "POST", "OPTIONS"],  # Explicitly allow methods
        "allow_headers": ["Content-Type"]  # Allow common headers
    }
})

@app.route('/api/initialize_001', methods=['POST'])
def init_001():
    return ps_001.initialize()

@app.route("/")
def home():
    return jsonify({"message": "This is the API."})

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)

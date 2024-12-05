import os
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO
from utils import preprocess_image, predict_and_format_result

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Running Flask App in environment: {os.environ}")

# Flask application
app = Flask(__name__, template_folder='.')  # Use current directory for templates

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    return "App is running successfully!", 200

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_filename = None
    error_message = None

    if request.method == 'POST':
        try:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Read file into memory
                file_content = BytesIO(file.read())

                # Predict and format result
                result, probability = predict_and_format_result(file_content)
                image_filename = filename
                return render_template('index.html', result=(result, probability), image=image_filename)
            else:
                error_message = "Invalid file type. Please upload a valid image."
                logging.error(error_message)
                raise ValueError(error_message)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('index.html', result=None, error=str(e))
    return render_template('index.html', result=result, image=image_filename, error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Read file into memory
            file_content = BytesIO(file.read())

            # Predict and format result
            result = predict_and_format_result(file_content)
            if result == "Anomalous":
                return jsonify({'error': 'Image is anomalous and cannot be classified.'}), 400
            return jsonify({'class': result[0], 'probability': result[1]})
        else:
            return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400
    except Exception as e:
        logging.error(f"API Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)  # Set to False for production

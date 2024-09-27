from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import easyocr
import numpy as np
from PIL import Image
import io
import base64
from PIL import ImageEnhance

app = Flask(__name__, static_url_path='/assets', static_folder='assets')
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize multiple EasyOCR readers
readers = {
    'en': easyocr.Reader(['en']),
    'es': easyocr.Reader(['es']),
    'fr': easyocr.Reader(['fr']),
    'tr': easyocr.Reader(['tr']),
    'tl': easyocr.Reader(['tl'])
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    # Get the image data from the request
    if request.json and 'image' in request.json:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    else:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Get the language from the request (default to English if not specified)
    language = request.json.get('language', 'en')
    
    # Perform OCR with the specified language
    reader = readers.get(language, readers['en'])
    result = reader.readtext(image_array, detail=0, paragraph=True)
    
    # Join the results into a single string
    text = ' '.join(result)
    
    return jsonify({'text': text})

@app.route('/commands.html')
def commands():
    return render_template('commands.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    if request.json and 'image' in request.json:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]
    else:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
    
    # Apply preprocessing (e.g., contrast enhancement)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    buffered = io.BytesIO()
    enhanced_image.save(buffered, format="PNG")
    enhanced_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({'image': f'data:image/png;base64,{enhanced_image_data}'})

if __name__ == '__main__':
    app.run(ssl_context=('localhost.pem', 'localhost-key.pem'), host='localhost', port=5000, debug=True)
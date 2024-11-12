from flask import Flask, request, jsonify, render_template, Response, send_file
from flask_cors import CORS
from PIL import Image
import io
import base64
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import easyocr
from typing import Dict, Any
import numpy as np
import time
from pyngrok import ngrok
import os
from langdetect import detect
import traceback
import sys
from docx2pdf import convert
import tempfile
import re
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes

app = Flask(__name__, static_url_path='/assets', static_folder='assets')
CORS(app, resources={r"/*": {"origins": "*"}})

# Load M2M100 model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
except ImportError as e:
    print(f"Error importing PaddleOCR: {e}")
    print("Please make sure PaddlePaddle and PaddleOCR are properly installed")
    sys.exit(1)

# Start ngrok
#ngrok.set_auth_token("2mvdbKaN0WGhsWJLNR6dja75qDb_4C2an6GPPfpZEmZHQ91sW")  #hide this sht
#public_url = ngrok.connect("5000")  # Expose your Flask app on port 5000 as a string
#print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    data: Dict[str, Any] = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Perform OCR with PaddleOCR
    result = ocr.ocr(np.array(image), cls=True)
    text = ' '.join([line[1][0] for line in result[0]])
    words = text.split()

    # Detect language
    try:
        detected_lang = detect(text)
    except:
        detected_lang = 'unknown'

    return jsonify({'text': text, 'words': words, 'detected_language': detected_lang})

@app.route('/translate', methods=['POST'])
def translate_text():
    data: Dict[str, Any] = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text = data.get('text')
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')

    if not all([text, source_lang, target_lang]):
        return jsonify({"error": "Missing required fields"}), 400

    translated_text = translate(text, source_lang, target_lang)
    translated_words = translated_text.split()
    
    return jsonify({'translated_text': translated_text, 'words': translated_words})

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

@app.route('/help.html')
def help_page():
    return render_template('help.html')

@app.errorhandler(500)
def internal_server_error(error):
    return f"500 Internal Server Error: {str(error)}\n\n{traceback.format_exc()}", 500

@app.route('/convert-to-pdf', methods=['POST'])
def convert_to_pdf():
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
        
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file.filename and (file.filename.endswith('.docx') or file.filename.endswith('.doc')):
        try:
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
                file.save(temp_docx.name)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                convert(temp_docx.name, temp_pdf.name)
                
            return send_file(
                temp_pdf.name,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='converted.pdf'
            )
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Unsupported file type"}), 400

@app.route('/detect_dates', methods=['POST'])
def detect_dates():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400
        
    try:
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    # Date patterns (existing)
    date_patterns = [
        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
        r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,\s+\d{4}\b'
    ]
    
    # Label patterns
    label_patterns = [
        r'(?:DATE\s+OF\s+ISSUE|ISSUE\s+DATE|DUE\s+DATE|DATE|DELIVERY\s+DATE)',
        r'(?:DATE\s*:)',
        r'(?:INVOICE\s*(?:NO|NUMBER|#|№)?)',
        r'(?:INV\s*(?:NO|NUMBER|#|№)?)'
    ]
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    detected_items = []
    result = ocr.ocr(np.array(image), cls=True)
    
    if result and result[0]:
        for i, line in enumerate(result[0]):
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence < 0.5:
                continue

            # Check for dates
            for pattern in date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detected_items.append({
                        "type": "date",
                        "text": match.group(),
                        "bbox": {
                            "x": int(bbox[0][0]),
                            "y": int(bbox[0][1]),
                            "width": int(bbox[2][0] - bbox[0][0]),
                            "height": int(bbox[2][1] - bbox[0][1])
                        }
                    })

            # Check for emails
            matches = re.finditer(email_pattern, text, re.IGNORECASE)
            for match in matches:
                detected_items.append({
                    "type": "email",
                    "text": match.group(),
                    "bbox": {
                        "x": int(bbox[0][0]),
                        "y": int(bbox[0][1]),
                        "width": int(bbox[2][0] - bbox[0][0]),
                        "height": int(bbox[2][1] - bbox[0][1])
                    }
                })

    return jsonify({"detected_items": detected_items})

@app.route('/analyze_layout', methods=['POST'])
def analyze_layout():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400
        
    try:
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    result = ocr.ocr(np.array(image), cls=True)
    layout_regions = []
    
    if result and result[0]:
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence < 0.5:
                continue
                
            # Determine region type based on content
            region_type = "text"  # default type
            
            # Check for headers (uppercase text)
            if text.isupper():
                region_type = "header"
            
            # Check for amounts/numbers
            elif re.match(r'^[\d,.]+$', text.strip()):
                region_type = "amount"
            
            # Check for table-like content
            elif '|' in text or '\t' in text:
                region_type = "table"
            
            layout_regions.append({
                "type": region_type,
                "text": text,
                "bbox": {
                    "x": int(bbox[0][0]),
                    "y": int(bbox[0][1]),
                    "width": int(bbox[2][0] - bbox[0][0]),
                    "height": int(bbox[2][1] - bbox[0][1])
                }
            })
    
    return jsonify({"layout_regions": layout_regions})

def process_document(file_data, file_type):
    if file_type == 'pdf':
        # Decode base64 PDF data
        pdf_bytes = base64.b64decode(file_data.split(',')[1])
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)
        text_content = []
        
        for image in images:
            # Convert PIL Image to numpy array for OCR
            img_array = np.array(image)
            
            # Perform OCR on the image
            result = ocr.ocr(img_array, cls=True)
            if result and result[0]:
                page_text = ' '.join([line[1][0] for line in result[0]])
                text_content.append(page_text)
        
        return '\n\n'.join(text_content)
    else:
        # Handle regular images
        image_data = base64.b64decode(file_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        result = ocr.ocr(np.array(image), cls=True)
        if result and result[0]:
            return ' '.join([line[1][0] for line in result[0]])
    return ''

@app.route('/document_ocr', methods=['POST'])
def document_ocr():
    data = request.get_json()
    if not data or 'document' not in data or 'fileType' not in data:
        return jsonify({"error": "Missing document data or file type"}), 400
        
    try:
        text = process_document(data['document'], data['fileType'])
        detected_lang = detect(text) if text else 'unknown'
        return jsonify({
            'text': text,
            'detected_language': detected_lang
        })
    except Exception as e:
        return jsonify({"error": f"Error processing document: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
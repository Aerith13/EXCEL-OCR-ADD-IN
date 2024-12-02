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
import cv2
from paddleocr import PaddleOCR
from paddleocr import PPStructure
from pathlib import Path
import json
from werkzeug.utils import secure_filename
import logging
import wget  # Commented out as it was causing an unresolved import error

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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get the base directory of paddleocr package
paddleocr_path = Path(__file__).parent / '.venv' / 'Lib' / 'site-packages' / 'paddleocr'

# Initialize PPStructure for table detection
table_engine = PPStructure(
    show_log=True,
    lang='en',
    det_model_dir='inference/en_PP-OCRv3_det_infer',
    rec_model_dir='inference/en_PP-OCRv3_rec_infer',
    table_model_dir='inference/en_ppstructure_mobile_v2.0_SLANet_infer',
    layout=True,
    table=True,
    ocr=True
)


def download_models():
    if not os.path.exists('inference'):
        os.makedirs('inference')
    
    models = {
        'det': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
        'rec': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
        'table': 'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar'
    }
    
    for model_type, url in models.items():
        tar_file = os.path.join('inference', os.path.basename(url))
        if not os.path.exists(tar_file):
            wget.download(url, tar_file)
            os.system(f'cd inference && tar xf {os.path.basename(url)}')

# Call this before initializing the table engine
download_models()

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
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400
        
        # Decode and process image
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_array = np.array(image)
        
        # TODO: Implement your own layout analysis here
        results = []
        
        return jsonify({'tables': results})
        
    except Exception as e:
        return jsonify({"error": f"Error analyzing layout: {str(e)}"}), 500

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

@app.route('/extract-table', methods=['POST'])
def extract_table():
    if not table_engine:
        return jsonify({'error': 'Table detection is not available. Please check server logs.'}), 503
        
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode and process image
        try:
            image_data = data['image'].split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Convert to numpy array for processing
            image_np = np.array(image)
            
            # Detect table boundaries
            table_regions = detect_table_boundaries(image_np)
            
            if not table_regions:
                return jsonify({
                    'error': 'No tables detected',
                    'suggestion': 'Please ensure the image contains clear table boundaries'
                }), 400
            
            # Process each detected table
            processed_tables = []
            for region in table_regions:
                # Extract table region
                x1, y1, x2, y2 = region['bbox']
                table_image = image_np[y1:y2, x1:x2]
                
                # Save temporary image
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_table_{len(processed_tables)}.jpg')
                Image.fromarray(table_image).save(temp_path)
                
                # Process with table engine
                result = table_engine(temp_path)
                
                # Extract structured data
                table_data = extract_table_structure(result)
                if table_data:
                    processed_tables.append({
                        'html': table_data['html'],
                        'structure': table_data['structure'],
                        'bbox': region['bbox'],
                        'confidence': region['confidence']
                    })
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if not processed_tables:
                return jsonify({
                    'error': 'Failed to extract table structure',
                    'suggestion': 'Try adjusting the image clarity or table alignment'
                }), 400
                
            return jsonify({
                'success': True,
                'tables': processed_tables,
                'message': f'Successfully extracted {len(processed_tables)} table(s)'
            })
            
        except Exception as e:
            return jsonify({
                'error': 'Table processing failed',
                'details': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}")
        return jsonify({'error': str(e)}), 500

def detect_table_boundaries(image):
    """
    Enhanced table detection for both regular and irregular tables
    """
    try:
        # First attempt: Use PPStructure for standard table detection
        result = table_engine(image)
        
        table_regions = []
        for region in result:
            if isinstance(region, dict) and region.get('type') == 'table':
                bbox = region.get('bbox', [])
                if len(bbox) == 4:
                    confidence = region.get('score', 0.0)
                    table_regions.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'type': 'structured'
                    })
        
        # If no tables found, try detecting aligned text patterns
        if not table_regions:
            # Get all text blocks with positions
            ocr_result = ocr.ocr(image, cls=True)
            if ocr_result and ocr_result[0]:
                text_blocks = []
                for line in ocr_result[0]:
                    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), (text, conf) = line
                    text_blocks.append({
                        'text': text,
                        'bbox': [min(x1, x2, x3, x4), min(y1, y2, y3, y4),
                                max(x1, x2, x3, x4), max(y1, y2, y3, y4)],
                        'confidence': conf
                    })
                
                # Group text blocks by vertical alignment
                aligned_blocks = detect_aligned_text_blocks(text_blocks)
                if aligned_blocks:
                    table_regions.append({
                        'bbox': get_table_bounds(aligned_blocks),
                        'confidence': calculate_alignment_confidence(aligned_blocks),
                        'type': 'aligned'
                    })
        
        return table_regions
    except Exception as e:
        logger.error(f"Error in enhanced table detection: {str(e)}")
        return []

def detect_aligned_text_blocks(text_blocks):
    """
    Detect text blocks that form table-like structures through alignment
    """
    # Sort blocks by vertical position
    sorted_blocks = sorted(text_blocks, key=lambda x: x['bbox'][1])
    
    # Group blocks into rows based on y-coordinate proximity
    rows = []
    current_row = [sorted_blocks[0]]
    y_threshold = 10  # Adjust based on your needs
    
    for block in sorted_blocks[1:]:
        if abs(block['bbox'][1] - current_row[0]['bbox'][1]) < y_threshold:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)
    
    # Check if rows have consistent column alignment
    if len(rows) >= 2:  # At least 2 rows needed for a table
        # Sort blocks in each row by x-coordinate
        aligned_rows = [sorted(row, key=lambda x: x['bbox'][0]) for row in rows]
        
        # Check column alignment
        if is_columnar_alignment(aligned_rows):
            return aligned_rows
    
    return None

def get_table_bounds(aligned_blocks):
    """
    Calculate the bounding box of a table from aligned text blocks
    """
    min_x = min([block['bbox'][0] for block in aligned_blocks])
    max_x = max([block['bbox'][2] for block in aligned_blocks])
    min_y = min([block['bbox'][1] for block in aligned_blocks])
    max_y = max([block['bbox'][3] for block in aligned_blocks])
    return [min_x, min_y, max_x, max_y]

def calculate_alignment_confidence(aligned_blocks):
    """
    Calculate the confidence of a table based on alignment
    """
    # Calculate the average confidence of the blocks
    avg_confidence = sum([block['confidence'] for block in aligned_blocks]) / len(aligned_blocks)
    
    # Calculate the standard deviation of the block positions
    std_dev = np.std([block['bbox'][0] for block in aligned_blocks])
    
    # Calculate the alignment confidence
    alignment_confidence = 1 - (std_dev / (max([block['bbox'][2] for block in aligned_blocks]) - min([block['bbox'][0] for block in aligned_blocks])))
    
    # Combine the two confidences
    return avg_confidence * alignment_confidence

def is_columnar_alignment(aligned_rows):
    """
    Check if the blocks in each row are aligned in columns
    """
    for row in aligned_rows:
        if len(set([block['bbox'][0] for block in row])) > 1:
            return False
    return True

def extract_table_structure(result):
    """
    Extract structured table data from PPStructure result
    """
    try:
        if not result or not isinstance(result, list):
            return None
            
        for region in result:
            if isinstance(region, dict) and region.get('type') == 'table':
                if 'res' not in region:
                    continue
                    
                table_res = region['res']
                if not isinstance(table_res, dict):
                    continue
                    
                # Extract HTML and cell structure
                html = table_res.get('html', '')
                cells = table_res.get('cells', [])
                
                # Build structured representation
                rows = {}
                for cell in cells:
                    row_idx = cell.get('row_idx', 0)
                    col_idx = cell.get('col_idx', 0)
                    text = cell.get('text', '')
                    
                    if row_idx not in rows:
                        rows[row_idx] = {}
                    rows[row_idx][col_idx] = text
                
                # Convert to ordered structure
                structured_data = []
                for row_idx in sorted(rows.keys()):
                    row = rows[row_idx]
                    structured_data.append([row.get(col_idx, '') for col_idx in sorted(row.keys())])
                
                return {
                    'html': html,
                    'structure': structured_data
                }
                
        return None
    except Exception as e:
        logger.error(f"Error extracting table structure: {str(e)}")
        return None
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
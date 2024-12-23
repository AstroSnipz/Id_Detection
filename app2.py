import cv2  # for image processing
import numpy as np
import pytesseract  # python library for OCR
from flask import Flask, request, jsonify  # web-application instance
from werkzeug.utils import secure_filename  # creating a folder to store the uploaded files temporarily
import os
import re  # for classification of gov-id type
from PIL import Image  # for image processing
from pdf2image import convert_from_path  # for pdf type uploaded gov-id
from ultralytics import YOLO  # for YOLO model
from config import MODEL_PATH  # path to YOLO model
import time  # for generating unique filenames

# Create a Flask web application instance
app = Flask(__name__)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}

# Creating the upload folder to store uploaded files and cropped faces
UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'uploads/cropped_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

# Clarity score threshold for uploaded gov-id images
CLARITY_THRESHOLD = 100

# Load the YOLO model for ID card detection
model = YOLO(MODEL_PATH)

# Haar cascade for face detection
HAAR_CASCADE_PATH = 'haar_face.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess images for better OCR performance
def preprocess_image(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Image must be either a file path or a numpy array")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to calculate the clarity score of an image
def calculate_clarity_score(image):
    if isinstance(image, np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Image must be a numpy array in BGR format")
    
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

# Function to clean up the OCR text
def clean_ocr_text(text):
    return text.strip()

# Function to identify the type of government ID from the image
def identify_gov_id(gov_id_image):
    preprocessed_img = preprocess_image(gov_id_image)
    h, w = preprocessed_img.shape
    x_start, y_start, x_end, y_end = int(w * 0.05), int(h * 0.1), int(w * 0.95), int(h * 0.8)
    cropped_img = preprocessed_img[y_start:y_end, x_start:x_end]
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(cropped_img, config=custom_config)
    extracted_text = clean_ocr_text(extracted_text)

    print(f"Extracted Text: {extracted_text}")  

    extracted_text_lower = extracted_text.lower()

    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'
    aadhaar_pattern = r'\d{4} \d{4} \d{4}'
    driving_license_pattern = r'[A-Z]{2}\d{2} \d{4} \d{7}'
    passport_pattern = r'\b[A-Z]{1}[0-9]{7}\b'

    if "permanent account number" in extracted_text_lower or re.search(pan_pattern, extracted_text):
        return "PAN Card"
    elif "aadhaar" in extracted_text_lower or re.search(aadhaar_pattern, extracted_text):
        return "Aadhaar Card"
    elif "driving licence" in extracted_text_lower or re.search(driving_license_pattern, extracted_text):
        return "Driving Licence"
    elif re.search(passport_pattern, extracted_text):
        return "Passport"
    elif "p<ind" in extracted_text_lower:
        return "Passport (MRZ Detected)"
    else:
        return "Unknown ID Type"

# Function to detect and crop face using Haar cascade
def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, "No face detected in the image."

    # Assume the first detected face is the most relevant
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]

    return cropped_face, "Face detected successfully."

# Function to generate a unique filename for the cropped face
def generate_unique_filename(prefix='cropped_face'):
    timestamp = int(time.time())  # Get the current time as a unique identifier
    return f"{prefix}_{timestamp}.jpg"

# Endpoint to upload government ID images
@app.route('/upload_id', methods=['POST'])
def upload_id():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Detect and crop the ID card
        results = model(file_path)
        img = cv2.imread(file_path)

        for result in results:
            boxes = result.boxes.xyxy
            class_ids = result.boxes.cls
            confidences = result.boxes.conf

            if len(boxes) > 0:
                # Assuming the first detection is the most relevant
                box = boxes[0]
                class_id = int(class_ids[0])
                confidence = float(confidences[0])
                class_name = model.names[class_id]

                # Check confidence score threshold
                if confidence < 0.7:
                    return jsonify({
                        "id_type": "Unknown ID",
                        "confidence_score": confidence,
                        "message": "ID uploaded but classified as Unknown due to low confidence."
                    }), 200

                x1, y1, x2, y2 = map(int, box)
                cropped_image = img[y1:y2, x1:x2]

                # Detect and crop face in the image
                cropped_face, face_message = detect_and_crop_face(file_path)

                # Save cropped face if detected
                cropped_face_path = None
                if cropped_face is not None:
                    cropped_face_filename = generate_unique_filename('cropped_face_id')
                    cropped_face_path = os.path.join(app.config['CROPPED_FOLDER'], cropped_face_filename)
                    cv2.imwrite(cropped_face_path, cropped_face)

                # Return the classified ID type and confidence score
                return jsonify({
                    "id_type": class_name,
                    "confidence_score": confidence,
                    "message": face_message,
                }), 200

        return jsonify({"error": "No ID card detected"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400



# Endpoint to upload selfie images
@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open the image using PIL and convert to RGB
        selfie_image = Image.open(file_path).convert('RGB')

        # Convert the image to a NumPy array for OpenCV
        selfie_cv_image = np.array(selfie_image)
        selfie_cv_image = cv2.cvtColor(selfie_cv_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format

        # Calculate clarity score for the selfie
        clarity_score = calculate_clarity_score(selfie_cv_image)

        # Check clarity score against threshold
        if clarity_score < CLARITY_THRESHOLD:
            return jsonify({
                "clarity_score": clarity_score,
                "message": "Clarity score is below the threshold. Please upload a clearer selfie."
            }), 400
        else:
            return jsonify({
                "clarity_score": clarity_score,
                "message": "Selfie uploaded successfully."
            }), 200

    return jsonify({'error': 'Invalid file type'}), 400

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

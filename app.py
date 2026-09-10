import os
import cv2
import requests
import base64
from flask import Flask, request, jsonify, render_template
import numpy as np
import math
import onnxruntime as ort

# --- Model and Configuration Setup ---
# Define paths to the model files within the 'models' directory
FACE_PROTO = os.path.join('models', 'opencv_face_detector.pbtxt')
FACE_MODEL = os.path.join('models', 'opencv_face_detector_uint8.pb')
AGE_GENDER_MODEL = os.path.join('models', 'age_gender', 'model.onnx')

# Load the face detector and the new ONNX age + gender model
try:
    faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    # Download the ONNX model automatically if it is not available locally.
    MODEL_URL = "https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx"

    if not os.path.exists(AGE_GENDER_MODEL):
        print("Age-gender model not found. Downloading from Hugging Face...")

        os.makedirs(os.path.dirname(AGE_GENDER_MODEL), exist_ok=True)

        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(AGE_GENDER_MODEL, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)

        print("Age-gender model downloaded successfully.")

    ageGenderSession = ort.InferenceSession(
        AGE_GENDER_MODEL,
        providers=["CPUExecutionProvider"]
    )
except Exception as e:
    print(f"Error loading one or more models: {e}")
    faceNet = None
    ageGenderSession = None

# The ONNX model expects 224x224 RGB images normalized with ImageNet values.
MODEL_INPUT_NAME = "pixel_values"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PADDING = 0

# Initialize Flask app
app = Flask(__name__)

def get_health_metrics(age_range, gender):
    """Generates health metrics based on age and gender."""
    # This function remains the same as before
    metrics = {
        "insurance_premium": "Standard Plan",
        "blood_donation": "Not Determined",
        "checkup_recommendation": "N/A",
        "lifestyle_notes": "N/A",
        "recommended_screenings": "N/A"
    }
    if age_range in ['(0-3)', '(4-7)', '(8-14)']:
        metrics["insurance_premium"] = "Family Floater Coverage (Child)"
        metrics["checkup_recommendation"] = "Regular Pediatric Checkups"
        metrics["recommended_screenings"] = "Developmental milestones, vision, and hearing screenings."
        metrics["lifestyle_notes"] = "Ensure adequate sleep, balanced nutrition, and daily physical activity."
        metrics["blood_donation"] = "Not Eligible (due to age restrictions)."
    elif age_range in ['(15-21)', '(22-34)']:
        metrics["insurance_premium"] = "Individual Health Plan (Starter)"
        metrics["checkup_recommendation"] = "Health checkup every 2-3 years."
        metrics["recommended_screenings"] = "Cholesterol check every 4-6 years. Blood pressure screening every 2 years."
        metrics["lifestyle_notes"] = "Maintain a balanced diet, regular exercise, and manage stress."
        metrics["blood_donation"] = "Potentially Eligible from age 18 (subject to screening)."
    else:
        metrics["checkup_recommendation"] = "Annual Health Checkups Recommended"
        metrics["lifestyle_notes"] = "Focus on heart health, strength training, and maintaining a healthy weight."
        screening = "Annual cholesterol & blood pressure checks. "
        if age_range == '(35-47)':
            metrics["insurance_premium"] = "Comprehensive Health Cover (Enhanced)"
            screening += "Discuss mammogram/prostate cancer screenings with your doctor."
        elif age_range == '(48-59)':
            metrics["insurance_premium"] = "Comprehensive Family Floater (Gold)"
            screening += "Mammogram/prostate cancer screenings recommended."
        elif age_range == '(60-100)':
            metrics["insurance_premium"] = "Senior Citizen Health Insurance"
            screening += "Colon cancer screening, bone density scans recommended."
        metrics["recommended_screenings"] = screening
        metrics["blood_donation"] = "Potentially Eligible up to age 65 (subject to screening)."
    return metrics

def get_security_metrics(age_range, gender):
    """
    UPDATED: Generates security-related assessments including checks for pubs, voting, and movies.
    """
    metrics = {
        "access_level": "Not Determined",
        "pub_entry": "Not Determined",
        "voting_eligibility": "Not Determined",
        "movie_rating_access": "Not Determined"
    }
    if age_range in ['(0-3)', '(4-7)', '(8-14)']:
        metrics["access_level"] = "Restricted Access. Requires guardian."
        metrics["pub_entry"] = "Denied (Underage)."
        metrics["voting_eligibility"] = "Not Eligible (Under 18)."
        metrics["movie_rating_access"] = "G-rated (General Audiences) content recommended."
    elif age_range == '(15-20)':
        metrics["access_level"] = "Limited Access to Age-Restricted Venues/Products."
        metrics["pub_entry"] = "Potentially granted from age 18+ (ID verification required)."
        metrics["voting_eligibility"] = "Eligible from age 18."
        metrics["movie_rating_access"] = "Eligible for PG-13/Teen rated content. Restricted from R-rated content until age 17/18."
    else: # This covers all age ranges from 22 and up
        metrics["access_level"] = "General Access."
        metrics["pub_entry"] = "Permitted (Subject to local laws and establishment policies)."
        metrics["voting_eligibility"] = "Eligible."
        metrics["movie_rating_access"] = "Unrestricted access to all movie ratings."
    return metrics

def get_marketing_metrics(age_range, gender):
    """
    UPDATED: Generates more detailed marketing insights.
    """
    metrics = {
        "target_demographic": "General Audience",
        "product_category_focus": "General consumer goods.",
        "campaign_strategy": "Broad-reach digital and traditional media.",
        "preferred_platforms": "Varied."
    }
    if age_range in ['(0-3)', '(4-7)', '(8-14)']:
        metrics["target_demographic"] = "Parents, Guardians, and Family Members."
        metrics["product_category_focus"] = "Toys, educational apps, children's clothing, pediatric health, family snacks."
        metrics["campaign_strategy"] = "Focus on family-oriented content, parenting blogs, and in-store promotions."
        metrics["preferred_platforms"] = "YouTube Kids, Parenting Forums, Facebook Groups."
    elif age_range in ['(15-21)', '(22-34)']:
        metrics["target_demographic"] = "Students and Young Professionals ('Gen Z' & 'Millennials')."
        metrics["product_category_focus"] = "Tech gadgets, fast fashion, streaming services, video games, travel experiences."
        metrics["campaign_strategy"] = "Digital-first: Social media influencers, short-form video content, and targeted online ads."
        metrics["preferred_platforms"] = "TikTok, Instagram, YouTube, Twitch."
    elif age_range in ['(35-47)', '(48-59)']:
        metrics["target_demographic"] = "Established Professionals and Homeowners ('Gen X')."
        metrics["product_category_focus"] = "Financial services (investments), family vehicles, home improvement, and premium brands."
        metrics["campaign_strategy"] = "Content marketing focused on value and reliability, email marketing, and targeted ads on professional networks."
        metrics["preferred_platforms"] = "Facebook, LinkedIn, News Websites, YouTube."
    elif age_range == '(60-100)':
        metrics["target_demographic"] = "Seniors, Retirees, and 'Baby Boomers'."
        metrics["product_category_focus"] = "Healthcare services, retirement planning, comfortable living products, and travel packages."
        metrics["campaign_strategy"] = "Use traditional media (TV, print) and clear, easy-to-read digital ads."
        metrics["preferred_platforms"] = "Facebook, News Websites, Email Newsletters."
    return metrics

def analyze_image_with_models(frame, application_type):
    """Detects faces, predicts age/gender, and gets metrics for the chosen application."""
    frame_height, frame_width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            
    if not face_boxes:
        return None, []

    final_frame = frame.copy()
    detailed_results = []
    
    for i, face_box in enumerate(face_boxes):
        face = frame[max(0, face_box[1] - PADDING):min(face_box[3] + PADDING, frame.shape[0] - 1),
                     max(0, face_box[0] - PADDING):min(face_box[2] + PADDING, frame.shape[1] - 1)]

        try:
            # OpenCV reads BGR; the ONNX model expects RGB.
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))

            # Convert to float32, scale to [0, 1], then apply ImageNet normalization.
            face_rgb = face_rgb.astype(np.float32) / 255.0
            face_rgb = (face_rgb - IMAGE_MEAN) / IMAGE_STD

            # HWC -> CHW -> NCHW
            model_input = np.transpose(face_rgb, (2, 0, 1))
            model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

            outputs = ageGenderSession.run(
                None,
                {MODEL_INPUT_NAME: model_input}
            )
            age_value, gender_probability_female = outputs[0][0]

            # Convert NumPy values to normal Python types
            age_value = int(np.clip(np.rint(float(age_value)), 0, 100))

            gender_probability_female = float(gender_probability_female)

            gender = "Female" if gender_probability_female >= 0.5 else "Male"

            gender_confidence = float(
                gender_probability_female
                if gender == "Female"
                else 1.0 - gender_probability_female
            )

            # Keep the existing application logic based on age groups.
            if age_value <= 3:
                age_range = "(0-3)"
            elif age_value <= 7:
                age_range = "(4-7)"
            elif age_value <= 14:
                age_range = "(8-14)"
            elif age_value <= 21:
                age_range = "(15-21)"
            elif age_value <= 34:
                age_range = "(22-34)"
            elif age_value <= 47:
                age_range = "(35-47)"
            elif age_value <= 59:
                age_range = "(48-59)"
            else:
                age_range = "(60-100)"

        except Exception as e:
            print(f"ONNX prediction error for face {i}: {e}")
            continue

        if application_type == 'health':
            metrics = get_health_metrics(age_range, gender)
        elif application_type == 'security':
            metrics = get_security_metrics(age_range, gender)
        elif application_type == 'marketing':
            metrics = get_marketing_metrics(age_range, gender)
        else:
            metrics = {"error": "Invalid application type"}

        detailed_results.append({
            "id": i,
            "age": age_value,
            "age_range": age_range,
            "gender": gender,
            "gender_confidence": round(gender_confidence, 4),
            "metrics": metrics,
            "application": application_type
        })

        # UPDATED: Removed gender from the display text on the image
        label = f"Age: {age_range}"
        cv2.putText(final_frame, label, (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(final_frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
        
    _, buffer = cv2.imencode('.jpg', final_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{img_str}"
    
    return image_data, detailed_results

# --- Flask Routes ---
@app.route('/')
def index():
    if faceNet is None or ageGenderSession is None:
        return "Server Error: Models could not be loaded.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    filepath = None
    application_type = None

    try:
        if 'image' in request.files:
            file = request.files['image']
            application_type = request.form.get('application')
            if file.filename == '': return jsonify({"error": "No selected file."}), 400
            filepath = os.path.join(UPLOAD_FOLDER, "upload.jpg")
            file.save(filepath)
        elif request.is_json:
            data = request.get_json()
            application_type = data.get('application')
            image_url = data.get('url')
            if not image_url: return jsonify({"error": "No URL provided."}), 400
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            filepath = os.path.join(UPLOAD_FOLDER, "url_upload.jpg")
            with open(filepath, 'wb') as f: f.write(response.content)
        else:
            return jsonify({"error": "Invalid request."}), 400
        
        if not application_type:
            return jsonify({"error": "Application type not specified."}), 400

        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "Could not read image file."}), 400

        image_data, detailed_results = analyze_image_with_models(frame, application_type)
        
        if image_data is None:
            return jsonify({"error": "No face detected in the image."}), 400

        return jsonify({"image_data": image_data, "results": detailed_results})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=True
    )
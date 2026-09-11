import os
import cv2
import requests
import base64
from flask import Flask, request, jsonify, render_template
import numpy as np
import math

# --- Model and Configuration Setup ---
# Define paths to the model files within the 'models' directory
FACE_PROTO = os.path.join('models', 'opencv_face_detector.pbtxt')
FACE_MODEL = os.path.join('models', 'opencv_face_detector_uint8.pb')
AGE_PROTO = os.path.join('models', 'age_deploy.prototxt')
AGE_MODEL = os.path.join('models', 'age_net.caffemodel')
GENDER_PROTO = os.path.join('models', 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join('models', 'gender_net.caffemodel')

# Load the networks from disk
try:
    faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
except cv2.error as e:
    print("Error loading one or more models. Make sure the model files are in the 'models' directory.")
    faceNet = ageNet = genderNet = None

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(22-29)', '(35-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
PADDING = 20

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
    if age_range in ['(0-2)', '(4-6)', '(8-12)']:
        metrics["insurance_premium"] = "Family Floater Coverage (Child)"
        metrics["checkup_recommendation"] = "Regular Pediatric Checkups"
        metrics["recommended_screenings"] = "Developmental milestones, vision, and hearing screenings."
        metrics["lifestyle_notes"] = "Ensure adequate sleep, balanced nutrition, and daily physical activity."
        metrics["blood_donation"] = "Not Eligible (due to age restrictions)."
    elif age_range in ['(15-20)', '(22-29)']:
        metrics["insurance_premium"] = "Individual Health Plan (Starter)"
        metrics["checkup_recommendation"] = "Health checkup every 2-3 years."
        metrics["recommended_screenings"] = "Cholesterol check every 4-6 years. Blood pressure screening every 2 years."
        metrics["lifestyle_notes"] = "Maintain a balanced diet, regular exercise, and manage stress."
        metrics["blood_donation"] = "Potentially Eligible from age 18 (subject to screening)."
    else:
        metrics["checkup_recommendation"] = "Annual Health Checkups Recommended"
        metrics["lifestyle_notes"] = "Focus on heart health, strength training, and maintaining a healthy weight."
        screening = "Annual cholesterol & blood pressure checks. "
        if age_range == '(35-43)':
            metrics["insurance_premium"] = "Comprehensive Health Cover (Enhanced)"
            screening += "Discuss mammogram/prostate cancer screenings with your doctor."
        elif age_range == '(48-53)':
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
    if age_range in ['(0-2)', '(4-6)', '(8-12)']:
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
    if age_range in ['(0-2)', '(4-6)', '(8-12)']:
        metrics["target_demographic"] = "Parents, Guardians, and Family Members."
        metrics["product_category_focus"] = "Toys, educational apps, children's clothing, pediatric health, family snacks."
        metrics["campaign_strategy"] = "Focus on family-oriented content, parenting blogs, and in-store promotions."
        metrics["preferred_platforms"] = "YouTube Kids, Parenting Forums, Facebook Groups."
    elif age_range in ['(15-20)', '(22-29)']:
        metrics["target_demographic"] = "Students and Young Professionals ('Gen Z' & 'Millennials')."
        metrics["product_category_focus"] = "Tech gadgets, fast fashion, streaming services, video games, travel experiences."
        metrics["campaign_strategy"] = "Digital-first: Social media influencers, short-form video content, and targeted online ads."
        metrics["preferred_platforms"] = "TikTok, Instagram, YouTube, Twitch."
    elif age_range in ['(35-43)', '(48-53)']:
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
            model_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
        except Exception:
            continue
        
        genderNet.setInput(model_blob)
        gender_preds = genderNet.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        ageNet.setInput(model_blob)
        age_preds = ageNet.forward()
        age_range = AGE_LIST[age_preds[0].argmax()]
        
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
            "age_range": age_range,
            "gender": gender,
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
    if not all([faceNet, ageNet, genderNet]):
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
    app.run(debug=True)
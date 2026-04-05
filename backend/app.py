from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from groq import Groq
from dotenv import load_dotenv
import time
from datetime import datetime
from pymongo import MongoClient
import math
import requests as http_requests

# -----------------------------
# 🔥 PATH CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

# -----------------------------
# 🔥 MONGODB CONNECTION
# -----------------------------
try:
    mongo_url = os.environ.get("mongodb_url", "")
    mongo_client = MongoClient(mongo_url)
    db = mongo_client["complaint_system"]
    complaints_collection = db["complaints"]
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

app = Flask(__name__)
app.secret_key = "super_secret_key_for_complaint_workflow"

# Initialize Groq Client
client = Groq(api_key=os.environ.get("grok_api"))


# Step-specific extraction hints for Groq
STEP_HINTS = {
    "Vehicle Details":    "Extract the vehicle registration number, make and model. Format: 'TN 01 AB 1234 - Vehicle Type'. If not in complaint, return NOT_FOUND.",
    "Customer Details":   "Extract the customer/complainant name and account or card number if mentioned. Format: 'Name: X, Account: Y'. If not found, return NOT_FOUND.",
    "Transaction Details":"Extract the transaction amount, date and transaction/reference ID. Format: 'Amount: X, Date: Y, Ref: Z'. If not found, return NOT_FOUND.",
    "Time of Incident":   "Extract the exact time or date of the incident. If not explicitly mentioned, infer from context (e.g. 'this morning', 'yesterday'). Return NOT_FOUND only if truly absent.",
    "Severity":           "Based on the complaint text, return exactly one of: High / Medium / Low — no other text.",
    "Risk Level":         "Based on the complaint text, return exactly one of: High / Medium / Low — no other text.",
    "Frequency":          "How often does this issue occur? Return e.g. 'Daily', 'Weekly', 'Ongoing', 'First occurrence'. If unclear return NOT_FOUND.",
    "Surface Type":       "What surface or area is affected? (e.g. Road, Pavement, Wall, Building). Extract from complaint.",
    "Damage Type":        "What type of damage is described? (e.g. Crack, Collapse, Pothole, Leakage). Extract from complaint.",
    "Structure Type":     "What structure or infrastructure is involved? (e.g. Bridge, Building, Road, Pipeline). Extract from complaint.",
    "Complaint Category": "Summarise the main category of this complaint in 3-5 words (e.g. 'Road Damage', 'Unauthorised Transaction').",
    "Issue Type":         "In 3-5 words, state the type of issue. E.g. 'Pothole', 'Garbage Overflow', 'Card Blocked'.",
    "Violation Type":     "What traffic or civic violation is described? E.g. 'Signal Jumping', 'Wrong Parking', 'Over-Speeding'.",
    "Complaint Type":     "What type of complaint is this? E.g. 'Garbage Not Collected', 'Drainage Block'.",
    "Description":        "Write a concise, formal 2-3 sentence description of the complaint suitable for a government form.",
    "Details":            "Write a concise, formal 2-3 sentence description of the complaint suitable for a government form.",
    "Location":           "Extract the specific area, street or landmark. Include any mentioned locality or city.",
}

NOT_ANSWER_PHRASES = {
    "not_found", "not specified", "not mentioned", "unknown", "n/a",
    "not available", "not provided", "none", "no information",
    "not stated", "not given", "unspecified", "information not provided"
}


def autofill_with_grok(step, complaint_text, location_context=""):
    print(f"🔍 AI attempting to auto-fill step: {step}")
    try:
        api_key = os.environ.get("grok_api")
        if not api_key:
            print("⚠️ No API key found for Grok (grok_api)")
            return ""

        hint = STEP_HINTS.get(step, f"Extract the '{step}' information from the complaint. If not found, return NOT_FOUND.")

        # Special handling: inject location context into Location step hint
        if "Location" in step and location_context:
            hint = f"{hint} Additionally, the user's captured location context is: {location_context}. Use this if no location is in the complaint text."

        prompt = (
            f"You are an assistant for a Tamil Nadu government complaint management system.\n"
            f"Complaint text: '{complaint_text}'\n"
            f"{('Additional context: ' + location_context) if location_context else ''}\n"
            f"Task: {hint}\n"
            f"Rules:\n"
            f"- Return ONLY the answer value. No labels, no explanations, no extra text.\n"
            f"- If the information is truly absent, return exactly: NOT_FOUND\n"
            f"- Keep it short and professional (suitable for a government form field)."
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        suggestion = chat_completion.choices[0].message.content.strip().replace('"', '').strip()
        print(f"🤖 AI suggestion for {step}: {suggestion}")

        # Filter out non-answers — return '' so the form shows required manual entry
        if suggestion.lower() in NOT_ANSWER_PHRASES or suggestion.lower().startswith("not_found"):
            print(f"⚠️ AI returned non-answer for {step}, showing manual entry field")
            return ""

        return suggestion
    except Exception as e:
        print(f"❌ AI Auto-fill Error: {e}")
        return ""

# -----------------------------
# 🔥 DEPARTMENT WORKFLOWS
# -----------------------------
DEPARTMENT_WORKFLOWS = {
    "🏦 Banking Department": ["Customer Details", "Issue Type", "Transaction Details", "Description", "Proof Upload"],
    "🚧 Road & Traffic Department": ["Issue Type", "Location", "Severity", "Description", "Image Upload"],
    "🗑 Municipal Department": ["Complaint Type", "Location", "Frequency", "Description", "Image Upload"],
    "🚓 Traffic Police Department": ["Violation Type", "Vehicle Details", "Location", "Time of Incident", "Image Proof"],
    "🏙 Municipal Cleaning Department": ["Issue Type", "Location", "Surface Type", "Description", "Image Upload"],
    "🏗 Infrastructure Department": ["Structure Type", "Damage Type", "Location", "Risk Level", "Image Upload"],
    "📌 General Department": ["Complaint Category", "Description", "Location", "Upload"]
}

# -----------------------------
# LOAD MODELS
# -----------------------------
text_model = joblib.load(os.path.join(MODELS_DIR, "text_model.pkl"))    
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
image_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "image_model.h5"))

# Image classes
class_labels = ['Damaged concrete', 'Graffiti', 'IllegalParking', 'Potholes', 'Garbage']

# -----------------------------
# 🔥 DEPARTMENT SIMULATION
# -----------------------------
def get_department_info(label):

    # Banking related
    if "Account" in label or "Transaction" in label:
        return "🏦 Banking Department", "Upload bank statement / transaction proof", "3-5 days"

    # Civic related
    elif label == "Potholes":
        return "🚧 Road & Traffic Department", "Provide location details", "2-3 days"

    elif label == "Garbage":
        return "🗑 Municipal Department", "Upload area photo / location", "1-2 days"

    elif label == "IllegalParking":
        return "🚓 Traffic Police Department", "Provide vehicle details / location", "2-3 days"

    elif label == "Graffiti":
        return "🏙 Municipal Cleaning Department", "Upload clear image of area", "2-4 days"

    elif label == "Damaged concrete":
        return "🏗 Infrastructure Department", "Provide location and damage details", "4-6 days"

    else:
        return "📌 General Department", "Provide more details", "3 days"


# -----------------------------
# 🔥 PRIORITY DETECTION
# -----------------------------
def get_priority_info(text):
    text = text.lower()
    high_keywords = ["danger", "fire", "accident", "electric shock", "urgent", "leakage", "risk", "fraud", "emergency"]
    if any(word in text for word in high_keywords):
        return "High", "#d93025"
    medium_keywords = ["delay", "not working", "issue", "problem", "broken", "malfunction"]
    if any(word in text for word in medium_keywords):
        return "Medium", "#f9ab00"
    return "Low", "#188038"


# -----------------------------
# 📍 GOVERNMENT OFFICE DATASET (Chennai / Tamil Nadu)
# -----------------------------
GOVT_OFFICES = [
    # 🙇 Banking
    {"department": "🏦 Banking Department",      "name": "RBI Chennai Regional Office",          "lat": 13.0688, "lon": 80.2781},
    {"department": "🏦 Banking Department",      "name": "State Bank Main Branch, Chennai",         "lat": 13.0827, "lon": 80.2707},
    # 🙇 Road & Traffic
    {"department": "🚧 Road & Traffic Department", "name": "PWD Highways Office, Chennai",            "lat": 13.0612, "lon": 80.2558},
    {"department": "🚧 Road & Traffic Department", "name": "PWD Division - Tambaram",                  "lat": 12.9249, "lon": 80.1000},
    {"department": "🚧 Road & Traffic Department", "name": "TNRDC Anna Nagar Office",                  "lat": 13.0850, "lon": 80.2101},
    # 🙇 Municipal
    {"department": "🗑 Municipal Department",      "name": "GCC T Nagar Zone Office",                 "lat": 13.0418, "lon": 80.2341},
    {"department": "🗑 Municipal Department",      "name": "GCC Anna Nagar Zone Office",               "lat": 13.0850, "lon": 80.2101},
    {"department": "🗑 Municipal Department",      "name": "GCC Adyar Zone Office",                    "lat": 13.0012, "lon": 80.2565},
    {"department": "🗑 Municipal Department",      "name": "GCC Sholinganallur Zone Office",            "lat": 12.9000, "lon": 80.2273},
    # 🙇 Traffic Police
    {"department": "🚓 Traffic Police Department", "name": "Chennai Traffic Police HQ",               "lat": 13.0827, "lon": 80.2707},
    {"department": "🚓 Traffic Police Department", "name": "Anna Nagar Traffic Police",                "lat": 13.0910, "lon": 80.2120},
    {"department": "🚓 Traffic Police Department", "name": "Tambaram Traffic Police Station",           "lat": 12.9249, "lon": 80.1234},
    # 🙇 Municipal Cleaning (Graffiti)
    {"department": "🏙 Municipal Cleaning Department", "name": "GCC Sanitation - North Zone",           "lat": 13.1100, "lon": 80.2850},
    {"department": "🏙 Municipal Cleaning Department", "name": "GCC Sanitation - South Zone",           "lat": 12.9850, "lon": 80.2200},
    # 🙇 Infrastructure
    {"department": "🏗 Infrastructure Department", "name": "CMDA Chennai HQ",                          "lat": 13.0524, "lon": 80.2496},
    {"department": "🏗 Infrastructure Department", "name": "TNEB Southern Regional Office",              "lat": 13.0012, "lon": 80.2565},
    # 🙇 General
    {"department": "📌 General Department",       "name": "Chennai Collectorate",                    "lat": 13.0827, "lon": 80.2707},
    {"department": "📌 General Department",       "name": "Tambaram Revenue Office",                  "lat": 12.9249, "lon": 80.1000},
]


# -----------------------------
# 📍 HAVERSINE DISTANCE + NEAREST OFFICE
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2  = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_office(user_lat, user_lon, department):
    """Return nearest office for the given department. Falls back to all offices."""
    candidates = [o for o in GOVT_OFFICES if o["department"] == department]
    if not candidates:
        candidates = GOVT_OFFICES          # broad fallback
    best = min(candidates, key=lambda o: haversine_km(user_lat, user_lon, o["lat"], o["lon"]))
    best["distance_km"] = round(haversine_km(user_lat, user_lon, best["lat"], best["lon"]), 2)
    return best

# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------
# TEXT PREDICTION
# -----------------------------
@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form['text']

    # 🔥 Predict with Confidence
    vec = vectorizer.transform([text])
    probs = text_model.predict_proba(vec)[0]
    confidence = np.max(probs)
    prediction = text_model.predict(vec)[0]

    # 🕵️ AUTOMATED ROUTING (Threshold 50%)
    if confidence < 0.5:
        dept, suggestion, time = "📌 General Department", "Provide more details", "3 days"
    else:
        dept, suggestion, time = get_department_info(prediction)

    # 📂 Handle optional supporting documents
    docs_msg = ""
    if 'docs' in request.files:
        doc_file = request.files['docs']
        if doc_file.filename != '':
            doc_path = os.path.join(UPLOADS_DIR, "doc_" + doc_file.filename)
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            doc_file.save(doc_path)
            docs_msg = "<br><small>✅ <i>Supporting document attached</i></small>"

    # 🔥 Get Priority and Color
    priority, p_color = get_priority_info(text)

    result = f"""
    <b>📝 Text Prediction:</b> {prediction} <br>
    <b>🎯 Confidence:</b> {confidence:.1%} <br><br>
    <b>🏢 Detected Department:</b> {dept} <br>
    <b style="color: {p_color};">🔥 Priority: {priority}</b> <br>
    <b>💡 Suggestion:</b> {suggestion} <br>
    <b>⏳ Estimated Time:</b> {time} <br>
    {docs_msg}
    <br><br>
    <form action="/start" method="POST" id="startFormText">
        <input type="hidden" name="dept" value="{dept}">
        <input type="hidden" name="complaint" value="{text}">
        <input type="hidden" name="priority" value="{priority}">
        <input type="hidden" name="p_color" value="{p_color}">
        <input type="hidden" name="user_lat" id="userLatText" value="">
        <input type="hidden" name="user_lon" id="userLonText" value="">
        <div id="locBoxText" style="background:#f0f8ff;border:1px solid #1a73e8;border-radius:10px;padding:12px;margin-bottom:12px;font-size:13px;">
            <b>📍 Capture Your Location (Optional)</b><br><br>
            <button type="button" onclick="getGPS('Text')" style="background:#1a73e8;color:white;border:none;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:12px;margin-right:6px;">📡 Use My GPS</button>
            <span style="color:#888;font-size:11px;">or enter manually:</span><br><br>
            <input type="text" id="manualLocText" placeholder="e.g. T Nagar, Chennai or 600017" style="width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;font-size:13px;margin-bottom:6px;">
            <button type="button" onclick="geocodeAddr('Text')" style="background:#34a853;color:white;border:none;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:12px;">🔍 Locate on Map</button>
            <div id="locStatusText" style="margin-top:8px;font-size:12px;color:#188038;"></div>
        </div>
        <button type="submit" style="background:green;width:100%;padding:11px;border:none;border-radius:8px;color:white;font-size:15px;font-weight:700;cursor:pointer;">Start Resolution Process →</button>
    </form>
    """

    return render_template('index.html', result=result)


# -----------------------------
# IMAGE PREDICTION
# -----------------------------
@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']

    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # 🔥 Predict with Confidence
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_array = image_model.predict(img_array)
    confidence = np.max(prediction_array)
    label = class_labels[np.argmax(prediction_array)]

    # 🕵️ AUTOMATED ROUTING (Threshold 50%)
    if confidence < 0.5:
        dept, suggestion, time = "📌 General Department", "Provide more details", "3 days"
    else:
        dept, suggestion, time = get_department_info(label)

    # 📂 Handle optional supporting documents
    docs_msg = ""
    if 'docs' in request.files:
        doc_file = request.files['docs']
        if doc_file.filename != '':
            doc_path = os.path.join(UPLOADS_DIR, "doc_img_" + doc_file.filename)
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            doc_file.save(doc_path)
            docs_msg = "<br><small>✅ <i>Supporting document attached</i></small>"

    # 🔥 Get department info
    dept, suggestion, time = get_department_info(label)

    # 🔥 Get Priority and Color
    priority, p_color = get_priority_info(label)

    result = f"""
    <b>📷 Image Prediction:</b> {label} <br>
    <b>🎯 Confidence:</b> {confidence:.1%} <br><br>
    <b>🏢 Detected Department:</b> {dept} <br>
    <b style="color: {p_color};">🔥 Priority: {priority}</b> <br>
    <b>💡 Suggestion:</b> {suggestion} <br>
    <b>⏳ Estimated Time:</b> {time} <br>
    {docs_msg}
    <br><br>
    <form action="/start" method="POST" id="startFormImg">
        <input type="hidden" name="dept" value="{dept}">
        <input type="hidden" name="complaint" value="{label}">
        <input type="hidden" name="priority" value="{priority}">
        <input type="hidden" name="p_color" value="{p_color}">
        <input type="hidden" name="user_lat" id="userLatImg" value="">
        <input type="hidden" name="user_lon" id="userLonImg" value="">
        <div id="locBoxImg" style="background:#f0f8ff;border:1px solid #1a73e8;border-radius:10px;padding:12px;margin-bottom:12px;font-size:13px;">
            <b>📍 Capture Your Location (Optional)</b><br><br>
            <button type="button" onclick="getGPS('Img')" style="background:#1a73e8;color:white;border:none;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:12px;margin-right:6px;">📡 Use My GPS</button>
            <span style="color:#888;font-size:11px;">or enter manually:</span><br><br>
            <input type="text" id="manualLocImg" placeholder="e.g. Velachery, Chennai" style="width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;font-size:13px;margin-bottom:6px;">
            <button type="button" onclick="geocodeAddr('Img')" style="background:#34a853;color:white;border:none;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:12px;">🔍 Locate on Map</button>
            <div id="locStatusImg" style="margin-top:8px;font-size:12px;color:#188038;"></div>
        </div>
        <button type="submit" style="background:green;width:100%;padding:11px;border:none;border-radius:8px;color:white;font-size:15px;font-weight:700;cursor:pointer;">Start Resolution Process →</button>
    </form>
    """

    return render_template('index.html', result=result)


# -----------------------------
# 🔥 WORKFLOW ROUTES
# -----------------------------

@app.route('/start', methods=['POST'])
def start_workflow():
    dept = request.form.get('dept')
    complaint_text = request.form.get('complaint')
    priority = request.form.get('priority', 'Low')
    p_color = request.form.get('p_color', '#188038')

    # 📍 Location from form
    try:
        user_lat = float(request.form.get('user_lat', '') or 0)
        user_lon = float(request.form.get('user_lon', '') or 0)
    except (ValueError, TypeError):
        user_lat, user_lon = 0.0, 0.0

    # 🏗 Find nearest office if location given
    assigned_office = None
    if user_lat != 0.0 and user_lon != 0.0:
        try:
            assigned_office = find_nearest_office(user_lat, user_lon, dept)
        except Exception as e:
            print(f"❌ Nearest office error: {e}")

    # Initialize session workflow
    session['workflow'] = {
        'dept': dept,
        'original_complaint': complaint_text,
        'priority': priority,
        'p_color': p_color,
        'steps': DEPARTMENT_WORKFLOWS.get(dept, DEPARTMENT_WORKFLOWS["📌 General Department"]),
        'current_step_idx': 0,
        'responses': {},
        'user_lat': user_lat,
        'user_lon': user_lon,
        'assigned_office': assigned_office
    }
    return redirect(url_for('show_step'))


@app.route('/step', methods=['GET', 'POST'])
def show_step():
    workflow = session.get('workflow')
    if not workflow:
        return redirect(url_for('home'))
    
    steps = workflow['steps']
    idx = workflow['current_step_idx']
    
    if request.method == 'POST':
        # Save response
        step_name = steps[idx]
        
        # Check if it's an upload step
        if 'Upload' in step_name or 'Proof' in step_name:
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                filename = f"workflow_{idx}_{file.filename}"
                filepath = os.path.join(UPLOADS_DIR, filename)
                os.makedirs(UPLOADS_DIR, exist_ok=True)
                file.save(filepath)
                workflow['responses'][step_name] = f"File uploaded: {filename}"
            else:
                workflow['responses'][step_name] = "No file uploaded"
        else:
            workflow['responses'][step_name] = request.form.get('response')
        
        # Move to next step
        workflow['current_step_idx'] += 1
        session['workflow'] = workflow
        
        if workflow['current_step_idx'] >= len(steps):
            return redirect(url_for('show_result'))
        
        return redirect(url_for('show_step'))
    
    # GET request
    current_step = steps[idx]
    print(f"📄 Rendering step: {current_step}")

    # 🤖 Build location context for Groq (GPS coords + already-answered location steps)
    location_context = ""
    user_lat = workflow.get('user_lat', 0.0)
    user_lon = workflow.get('user_lon', 0.0)
    if user_lat and user_lat != 0.0:
        location_context = f"User GPS coordinates: lat={user_lat}, lon={user_lon} (Chennai/Tamil Nadu area)."
    # If a Location-type step was already answered earlier, use it
    for answered_step, answered_val in workflow['responses'].items():
        if 'Location' in answered_step and answered_val:
            location_context = f"User's location is: {answered_val}."
            break
    # If the manually entered location address is stored in session, include it
    if not location_context and workflow.get('location_address'):
        location_context = f"User's location area: {workflow['location_address']}."

    # 🤖 AI Auto-fill logic
    suggested_value = ""
    if not ('Upload' in current_step or 'Proof' in current_step):
        print(f"🤖 Calling AI for step: {current_step}")
        suggested_value = autofill_with_grok(current_step, workflow['original_complaint'], location_context)
        print(f"🤖 AI suggested value: '{suggested_value}'")

    return render_template('step.html', 
                         step=current_step, 
                         step_num=idx + 1, 
                         total_steps=len(steps),
                         complaint=workflow['original_complaint'],
                         priority=workflow.get('priority', 'Low'),
                         p_color=workflow.get('p_color', '#188038'),
                         suggested_value=suggested_value)


@app.route('/result')
def show_result():
    workflow = session.pop('workflow', None)
    if not workflow:
        return redirect(url_for('home'))

    # Generate unique complaint ID
    complaint_id = "CMP" + str(int(time.time()))

    # 📍 Location data from session
    user_lat     = workflow.get('user_lat', 0.0)
    user_lon     = workflow.get('user_lon', 0.0)
    assigned_office = workflow.get('assigned_office')  # may be None

    # Prepare complaint data
    complaint_data = {
        "complaint_id":   complaint_id,
        "complaint_text": workflow['original_complaint'],
        "department":     workflow['dept'],
        "priority":       workflow.get('priority', 'Low'),
        "p_color":        workflow.get('p_color', '#188038'),
        "steps_data":     workflow['responses'],
        "status":         "Submitted",
        "timestamp":      datetime.now().isoformat(),
        "user_location":  {"lat": user_lat, "lon": user_lon} if user_lat else None,
        "assigned_office": assigned_office
    }

    # Save to MongoDB
    try:
        complaints_collection.insert_one(complaint_data)
        print(f"✅ Complaint {complaint_id} saved to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB Insert Error: {e}")

    return render_template('result.html',
                         responses=workflow['responses'],
                         dept=workflow['dept'],
                         complaint=workflow['original_complaint'],
                         priority=workflow.get('priority', 'Low'),
                         p_color=workflow.get('p_color', '#188038'),
                         complaint_id=complaint_id,
                         user_lat=user_lat,
                         user_lon=user_lon,
                         assigned_office=assigned_office)


@app.route('/geocode')
def geocode_proxy():
    """Proxy Nominatim geocoding to avoid CORS issues."""
    addr = request.args.get('q', '')
    if not addr:
        return jsonify({'error': 'No address'}), 400
    try:
        resp = http_requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': addr + ', Tamil Nadu, India', 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'AIComplaintSystem/1.0'},
            timeout=5
        )
        data = resp.json()
        if data:
            return jsonify({'lat': float(data[0]['lat']), 'lon': float(data[0]['lon']), 'display': data[0]['display_name']})
        return jsonify({'error': 'Not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/track/<complaint_id>')
def track_complaint(complaint_id):
    try:
        complaint_data = complaints_collection.find_one({"complaint_id": complaint_id})
    except Exception as e:
        print(f"❌ MongoDB Query Error: {e}")
        return render_template('track.html', error="Database connection failed")
        
    if not complaint_data:
        return render_template('track.html', error="Invalid Complaint ID")
        
    # Dynamically update status based on elapsed time for tracking purposes
    try:
        submitted_time = datetime.fromisoformat(complaint_data['timestamp'])
        elapsed_seconds = (datetime.now() - submitted_time).total_seconds()
        
        if elapsed_seconds >= 180:
            complaint_data['status'] = "Resolved"
            complaint_data['status_color'] = "#188038" # Green
        elif elapsed_seconds >= 60:
            complaint_data['status'] = "In Progress"
            complaint_data['status_color'] = "#f9ab00" # Yellow
        else:
            complaint_data['status'] = "Submitted"
            complaint_data['status_color'] = "#1a73e8" # Blue
    except Exception as e:
        print(f"Time parsing error: {e}")
        complaint_data['status_color'] = "#1a73e8" # Blue
            
    return render_template('track.html', complaint=complaint_data)


# -----------------------------
# 📊 ANALYTICS DASHBOARD
# -----------------------------
@app.route('/dashboard')
def dashboard():
    try:
        # Fetch without projection; strip _id manually to avoid silent projection failures
        raw = list(complaints_collection.find({}))
        all_complaints = [{k: v for k, v in doc.items() if k != '_id'} for doc in raw]
    except Exception as e:
        print(f"❌ MongoDB fetch error: {e}")
        all_complaints = []

    if not all_complaints:
        return render_template('dashboard.html', no_data=True, analytics={})

    # --- Aggregation ---
    dept_counts = {}
    priority_counts = {"High": 0, "Medium": 0, "Low": 0}
    status_counts = {"Submitted": 0, "In Progress": 0, "Resolved": 0}

    for c in all_complaints:
        # Department
        dept = c.get("department", "General Department")
        # Strip emoji prefix for cleaner labels
        dept_clean = dept.split(" ", 1)[-1] if dept else "Unknown"
        dept_counts[dept_clean] = dept_counts.get(dept_clean, 0) + 1

        # Priority
        p = c.get("priority", "Low")
        if p in priority_counts:
            priority_counts[p] += 1
        else:
            priority_counts["Low"] += 1

        # Status: dynamically compute from timestamp (same logic as track route)
        try:
            submitted_time = datetime.fromisoformat(c["timestamp"])
            elapsed = (datetime.now() - submitted_time).total_seconds()
            if elapsed >= 180:
                status = "Resolved"
            elif elapsed >= 60:
                status = "In Progress"
            else:
                status = "Submitted"
        except Exception:
            status = c.get("status", "Submitted")

        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["Submitted"] += 1

    analytics = {
        "departments": dept_counts,
        "priority": priority_counts,
        "status": status_counts,
        "total": len(all_complaints)
    }

    return render_template('dashboard.html', no_data=False, analytics=analytics)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
import os
import sys
import html
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
from datetime import date

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from google.cloud import translate_v2 as translate
except Exception:
    translate = None


# ===================== HELPERS =====================

def detect_season_from_date(planting_date):
    if isinstance(planting_date, str):
        planting_date = pd.Timestamp(planting_date)
    elif isinstance(planting_date, date):
        planting_date = pd.Timestamp(planting_date)

    month = planting_date.month
    if month in [6, 7, 8, 9]:
        return "Kharif", "Rainy season (June-September)"
    elif month in [10, 11, 12, 1]:
        return "Rabi", "Winter season (October-January)"
    elif month in [2, 3, 4, 5]:
        return "Zaid", "Summer season (February-May)"
    return "Year-round", "Any time"


def get_soil_by_pincode(pincode):
    soil_database = {
        "700001": {"nitrogen": 70, "phosphorus": 35, "potassium": 50, "pH": 6.5, "soil_type": "Alluvial"},
        "700012": {"nitrogen": 75, "phosphorus": 40, "potassium": 55, "pH": 6.8, "soil_type": "Alluvial"},
        "734001": {"nitrogen": 90, "phosphorus": 45, "potassium": 60, "pH": 5.8, "soil_type": "Red"},
        "743101": {"nitrogen": 65, "phosphorus": 30, "potassium": 45, "pH": 7.0, "soil_type": "Clay"},
    }
    return soil_database.get(
        pincode,
        {"nitrogen": 75, "phosphorus": 40, "potassium": 50, "pH": 6.8, "soil_type": "Alluvial (estimated)"},
    )


def get_weather_by_district(district, state, planting_date, season):
    """
    Returns mock weather values for the current project demo.

    These keys must match the ML model feature requirements:
    temperature, humidity, rainfall.
    """

    import random

    if season == "Kharif":
        return {
            "temperature": 25 + random.randint(-2, 2),
            "humidity": 82 + random.randint(-5, 5),
            "rainfall": 220 + random.randint(-30, 30),
        }

    elif season == "Rabi":
        return {
            "temperature": 21 + random.randint(-2, 2),
            "humidity": 62 + random.randint(-5, 5),
            "rainfall": 55 + random.randint(-15, 15),
        }

    else:  # Zaid
        return {
            "temperature": 30 + random.randint(-2, 2),
            "humidity": 70 + random.randint(-5, 5),
            "rainfall": 80 + random.randint(-20, 20),
        }

def get_expected_yield(crop, land_acres):
    """
    Expected yield (kg) per crop, extended for all 22 model crops.
    Values are approximate; adjust as needed for your project report.
    """

    yield_per_acre = {
        # Cereals / millets
        "rice": 2500,
        "maize": 3000,
        # Pulses / legumes
        "chickpea": 1200,
        "kidneybeans": 1000,
        "pigeonpeas": 900,
        "mothbeans": 700,
        "mungbean": 800,
        "blackgram": 800,
        "lentil": 1000,
        # Fruits
        "pomegranate": 6000,
        "banana": 25000,
        "mango": 8000,
        "orange": 12000,
        "apple": 10000,
        "grapes": 10000,
        "watermelon": 12000,
        "muskmelon": 10000,
        "papaya": 15000,
        "coconut": 6000,
        # Cash / others
        "cotton": 600,
        "jute": 1500,
        "coffee": 800,
    }

    # Default yield if crop not listed (should not happen with your model)
    base_yield = yield_per_acre.get(crop, 2000)

    # Small land tends to be more intensive per acre
    if land_acres < 1:
        adjustment = 1.2
    elif land_acres < 2.5:
        adjustment = 1.1
    elif land_acres < 5:
        adjustment = 1.0
    else:
        adjustment = 0.95

    return base_yield * land_acres * adjustment

def get_market_price(crop):
    """
    Approximate market price (₹ per kg) for each of the 22 crops.
    These are indicative values; adjust for your project report if needed.
    """

    prices = {
        # Cereals / millets
        "rice": 35,
        "maize": 20,
        # Pulses / legumes
        "chickpea": 60,
        "kidneybeans": 70,
        "pigeonpeas": 65,
        "mothbeans": 70,
        "mungbean": 70,
        "blackgram": 70,
        "lentil": 65,
        # Fruits
        "pomegranate": 60,
        "banana": 30,
        "mango": 50,
        "orange": 40,
        "apple": 80,
        "grapes": 60,
        "watermelon": 15,
        "muskmelon": 20,
        "papaya": 20,
        "coconut": 25,
        # Cash / others
        "cotton": 60,      # per kg of lint (simplified)
        "jute": 25,        # per kg of fibre (simplified)
        "coffee": 250,     # per kg of beans (very rough)
    }

    # Default price if crop not listed
    return prices.get(crop, 30)

def get_harvest_time(crop):
    """
    Approximate days from sowing to harvest for each of the 22 crops.
    Values are indicative; adjust for your project report if needed.
    """

    days = {
        # Cereals / millets
        "rice": 120,
        "maize": 90,
        # Pulses / legumes
        "chickpea": 100,
        "kidneybeans": 90,
        "pigeonpeas": 150,
        "mothbeans": 75,
        "mungbean": 75,
        "blackgram": 75,
        "lentil": 100,
        # Fruits
        "pomegranate": 150,
        "banana": 365,
        "mango": 365,
        "orange": 365,
        "apple": 365,
        "grapes": 365,
        "watermelon": 90,
        "muskmelon": 90,
        "papaya": 270,
        "coconut": 365,
        # Cash / others
        "cotton": 180,
        "jute": 120,
        "coffee": 365,
    }

    # Default duration if crop not listed
    return days.get(crop, 90)


def calculate_profit(crop, land_acres, budget):
    price_per_kg = {
        "rice": 35,
        "wheat": 30,
        "maize": 20,
        "pulses": 80,
        "cotton": 60,
        "sugarcane": 4,
        "vegetables": 40,
    }
    cost_per_acre = {
        "🟢 Low budget (₹5,000 - ₹20,000)": 10000,
        "🟡 Medium budget (₹20,000 - ₹50,000)": 30000,
        "🔴 High budget (₹50,000+)": 50000,
    }

    yield_kg = get_expected_yield(crop, land_acres)
    revenue = yield_kg * price_per_kg.get(crop, 30)
    total_cost = cost_per_acre.get(budget, 30000) * land_acres
    profit = revenue - total_cost

    return {
        "revenue": f"₹{revenue:.0f}",
        "cost": f"₹{total_cost:.0f}",
        "profit": f"₹{profit:.0f}",
        "profit_per_acre": f"₹{profit/land_acres:.0f}/acre",
    }


def adjust_for_constraints(crop, inputs):
    water = inputs["water_source"]
    land_acres = inputs["land_acres"]
    budget = inputs["budget"]

    if water == "🌧️ Rain-only (no irrigation)" and crop in ["rice", "sugarcane"]:
        crop = "maize"
    if land_acres < 1 and crop in ["sugarcane", "cotton"]:
        crop = "vegetables"
    if budget == "🟢 Low budget (₹5,000 - ₹20,000)" and crop in ["cotton", "sugarcane"]:
        crop = "pulses"
    return crop


def generate_explanation(crop, inputs):
    soil = inputs["soil"]
    weather = inputs["weather"]
    land_acres = inputs["land_acres"]

    return f"""
### 🌾 Why {crop} is Recommended

**Soil Analysis:**
- Nitrogen: {soil['nitrogen']} mg/kg
- Phosphorus: {soil['phosphorus']} mg/kg
- Potassium: {soil['potassium']} mg/kg
- pH: {soil['pH']}
- Soil Type: {soil['soil_type']}

**Weather Conditions:**
- Temperature: {weather['temperature']}°C
- Humidity: {weather['humidity']}%
- Rainfall: {weather['rainfall']} mm

**Your Constraints:**
- Land: {land_acres} acres
- Water: {inputs['water_source']}
- Budget: {inputs['budget']}

**Why this works:**
{crop} fits your soil, weather, and farm conditions.
"""


def generate_action_plan(crop, inputs):
    planting_date = pd.Timestamp(inputs["planting_date"])
    harvest_days = get_harvest_time(crop)
    harvest_date = planting_date + pd.Timedelta(days=harvest_days)

    return f"""
### 📅 7-Day Action Plan for {crop}

**Day 1-2:** Prepare field and remove weeds.  
**Day 3:** Arrange seeds and compost.  
**Day 4:** Apply fertilizer.  
**Day 5:** Sow/plant crop.  
**Day 6:** Initial irrigation if needed.  
**Day 7:** Monitor for pests and disease.  

**Timeline:**
- Sowing date: {planting_date.strftime("%d %B %Y")}
- Expected harvest: {harvest_date.strftime("%d %B %Y")}
- Harvest duration: ~{harvest_days} days
"""


def generate_risks(crop, inputs):
    water = inputs["water_source"]
    market = inputs["market_access"]

    water_risk = "⚠️ Rain-only water may cause drought stress." if water == "🌧️ Rain-only (no irrigation)" else "✅ Water availability looks fine."
    market_risk = "⚠️ Far market distance may be risky for perishable crops." if market == "❌ Very far (> 30 km)" and crop in ["vegetables", "flowers"] else "✅ Market access is acceptable."

    return f"""
### ⚠️ Risks & Warnings for {crop}

- {water_risk}
- {market_risk}
- Monitor weather weekly.
- Keep backup irrigation if possible.
"""


def predict_crop(inputs):
    """
    Uses the saved Random Forest model to recommend crops.
    """

    from pathlib import Path

    # Absolute path to the model file we just verified
    model_path = Path(
        "/Users/stutisamanta/Desktop/Crop/AI-powered-Crop-Advisory-System/models/crop_recommendator.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}. "
            "Make sure crop_recommendator.pkl exists in the models folder."
        )

    model = joblib.load(model_path)

    soil = inputs["soil"]
    weather = inputs["weather"]

    # IMPORTANT: use the exact 7 features the model was trained with,
    # in the same order and with the same names.
    import pandas as pd

    input_data = pd.DataFrame([[
        soil["nitrogen"],
        soil["phosphorus"],
        soil["potassium"],
        weather["temperature"],
        weather["humidity"],
        soil["pH"],
        weather["rainfall"],
    ]], columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ])

    # Main ML prediction
    ml_crop = model.predict(input_data)[0]

    # Probability for every crop
    probabilities = model.predict_proba(input_data)[0]

    crop_probabilities = pd.DataFrame({
        "Crop": model.classes_,
        "Confidence (%)": probabilities * 100,
    })

    # Top 3 recommendations
    top_3 = crop_probabilities.sort_values(
        by="Confidence (%)",
        ascending=False,
    ).head(3).reset_index(drop=True)

    top_3["Confidence (%)"] = top_3["Confidence (%)"].round(2)

    # Apply practical farm constraints after ML prediction.
    final_crop = adjust_for_constraints(ml_crop, inputs)

    return {
        "crop": final_crop,
        "original_ml_crop": ml_crop,
        "confidence": float(top_3.iloc[0]["Confidence (%)"]),
        "top_3": top_3,
        "explanation": generate_explanation(final_crop, inputs),
        "action_plan": generate_action_plan(final_crop, inputs),
        "risks": generate_risks(final_crop, inputs),
        "yield": f"{get_expected_yield(final_crop, inputs['land_acres']):.0f} kg",
        "price": get_market_price(final_crop),
    }

def get_translate_client():
    if translate is None:
        return None
    try:
        return translate.Client()
    except Exception:
        return None


@st.cache_resource
def cached_translate_client():
    return get_translate_client()


def translate_text(text, target_language):
    if not text or target_language == "en":
        return text

    client = cached_translate_client()
    if client is None:
        return text

    try:
        result = client.translate(text, target_language=target_language)
        return html.unescape(result["translatedText"])
    except Exception:
        return text


def t(key):
    lang = st.session_state.get("lang", "en")
    return TEXT[lang][key]


# ===================== APP CONFIG =====================

st.set_page_config(page_title="Smart Crop Advisory Assistant", page_icon="🌾", layout="wide")

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Urdu": "ur",
    "Gujarati": "gu",
}

TEXT = {
    "en": {
        "title": "Smart Crop Advisory Assistant",
        "subtitle": "Get crop recommendations based on your location and farming conditions",
        "tagline": "Location-based approach - No manual soil input required!",
        "location": "Location Details",
        "planting_date": "When Do You Want to Plant?",
        "soil_type": "Soil Type (Optional)",
        "last_crop": "Last Season's Crop",
        "water": "Water Availability",
        "electricity": "Electricity Availability",
        "land": "Land Size (Exact Acres)",
        "budget": "Investment Budget",
        "market": "Market Access",
        "fetch": "Fetch Soil & Weather Data",
        "recommend": "Get Crop Recommendation",
        "help": "Help & Information",
        "about": "About This App",
        "how_to_use": "How to Use",
        "tips": "Tips for Best Results",
        "season_kharif": "Kharif: June-September",
        "season_rabi": "Rabi: October-January",
        "season_zaid": "Zaid: February-May",
        "location_warn": "Please fill Pincode and District to fetch data",
        "data_ok": "Data fetched successfully",
        "need_fetch": "Please fetch soil & weather data first",
        "rec_ok": "Recommendation generated successfully!",
        "why": "Why This Crop?",
        "action": "7-Day Action Plan",
        "yield_price": "Expected Yield & Price",
        "profit": "Profit Calculation",
        "risks": "Risks & Warnings",
    },
    "hi": {
        "title": "स्मार्ट फसल सलाहकार",
        "subtitle": "अपनी जगह और खेती की स्थिति के आधार पर फसल सुझाव पाएं",
        "tagline": "स्थान आधारित दृष्टिकोण - मिट्टी का मैन्युअल इनपुट नहीं चाहिए!",
        "location": "स्थान विवरण",
        "planting_date": "आप कब बोना चाहते हैं?",
        "soil_type": "मिट्टी का प्रकार (वैकल्पिक)",
        "last_crop": "पिछली फसल",
        "water": "पानी की उपलब्धता",
        "electricity": "बिजली की उपलब्धता",
        "land": "भूमि का आकार (सटीक एकड़)",
        "budget": "निवेश बजट",
        "market": "बाजार पहुंच",
        "fetch": "मिट्टी और मौसम डेटा प्राप्त करें",
        "recommend": "फसल सिफारिश प्राप्त करें",
        "help": "सहायता और जानकारी",
        "about": "इस ऐप के बारे में",
        "how_to_use": "इस्तेमाल कैसे करें",
        "tips": "बेहतर परिणाम के लिए सुझाव",
        "season_kharif": "खरीफ: जून-सितंबर",
        "season_rabi": "रबी: अक्टूबर-जनवरी",
        "season_zaid": "जायद: फरवरी-मई",
        "location_warn": "डेटा प्राप्त करने के लिए पिनकोड और जिला भरें",
        "data_ok": "डेटा सफलतापूर्वक प्राप्त हुआ",
        "need_fetch": "कृपया पहले मिट्टी और मौसम डेटा प्राप्त करें",
        "rec_ok": "सिफारिश सफलतापूर्वक तैयार हुई!",
        "why": "यह फसल क्यों?",
        "action": "7-दिन की कार्य योजना",
        "yield_price": "अपेक्षित उपज और कीमत",
        "profit": "लाभ गणना",
        "risks": "जोखिम और चेतावनियाँ",
    },
    "bn": {
        "title": "স্মার্ট ফসল পরামর্শ সহায়ক",
        "subtitle": "আপনার অবস্থান এবং কৃষি অবস্থার ভিত্তিতে ফসল সুপারিশ পান",
        "tagline": "অবস্থানভিত্তিক পদ্ধতি - মাটির ম্যানুয়াল ইনপুটের প্রয়োজন নেই!",
        "location": "অবস্থানের বিবরণ",
        "planting_date": "আপনি কখন বপন করতে চান?",
        "soil_type": "মাটির ধরন (ঐচ্ছিক)",
        "last_crop": "গত মৌসুমের ফসল",
        "water": "জলের প্রাপ্যতা",
        "electricity": "বিদ্যুৎ প্রাপ্যতা",
        "land": "জমির আকার (নির্দিষ্ট একর)",
        "budget": "বিনিয়োগ বাজেট",
        "market": "বাজারে পৌঁছানো",
        "fetch": "মাটি ও আবহাওয়ার তথ্য আনুন",
        "recommend": "ফসল সুপারিশ পান",
        "help": "সহায়তা ও তথ্য",
        "about": "এই অ্যাপ সম্পর্কে",
        "how_to_use": "কিভাবে ব্যবহার করবেন",
        "tips": "সেরা ফলাফলের টিপস",
        "season_kharif": "খরিফ: জুন-সেপ্টেম্বর",
        "season_rabi": "রবি: অক্টোবর-জানুয়ারি",
        "season_zaid": "জায়দ: ফেব্রুয়ারি-মে",
        "location_warn": "ডেটা পেতে পিনকোড ও জেলা পূরণ করুন",
        "data_ok": "ডেটা সফলভাবে পাওয়া গেছে",
        "need_fetch": "অনুগ্রহ করে আগে মাটি ও আবহাওয়ার তথ্য আনুন",
        "rec_ok": "সুপারিশ সফলভাবে তৈরি হয়েছে!",
        "why": "এই ফসল কেন?",
        "action": "৭ দিনের কর্মপরিকল্পনা",
        "yield_price": "প্রত্যাশিত ফলন ও দাম",
        "profit": "লাভ গণনা",
        "risks": "ঝুঁকি ও সতর্কতা",
    },
    "te": {
        "title": "స్మార్ట్ పంట సలహా సహాయకం",
        "subtitle": "మీ స్థానం మరియు వ్యవసాయ పరిస్థితుల ఆధారంగా పంట సిఫారసులు పొందండి",
        "tagline": "స్థానాధారిత విధానం - మాన్యువల్ మట్టి ఇన్‌పుట్ అవసరం లేదు!",
        "location": "స్థాన వివరాలు",
        "planting_date": "మీరు ఎప్పుడు నాటాలనుకుంటున్నారు?",
        "soil_type": "నేల రకం (ఐచ్చికం)",
        "last_crop": "గత సీజన్ పంట",
        "water": "నీటి లభ్యత",
        "electricity": "విద్యుత్ లభ్యత",
        "land": "భూమి పరిమాణం (ఖచ్చిత ఎకరాలు)",
        "budget": "పెట్టుబడి బడ్జెట్",
        "market": "మార్కెట్ ప్రవేశం",
        "fetch": "నేల మరియు వాతావరణ డేటా పొందండి",
        "recommend": "పంట సిఫారసు పొందండి",
        "help": "సహాయం మరియు సమాచారం",
        "about": "ఈ యాప్ గురించి",
        "how_to_use": "వాడకం ఎలా",
        "tips": "మంచి ఫలితాల కోసం సూచనలు",
        "season_kharif": "ఖరీఫ్: జూన్-సెప్టెంబర్",
        "season_rabi": "రబీ: అక్టోబర్-జనవరి",
        "season_zaid": "జైద్: ఫిబ్రవరి-మే",
        "location_warn": "డేటా కోసం పిన్కోడ్ మరియు జిల్లా నమోదు చేయండి",
        "data_ok": "డేటా విజయవంతంగా పొందబడింది",
        "need_fetch": "దయచేసి ముందుగా నేల మరియు వాతావరణ డేటాను పొందండి",
        "rec_ok": "సిఫారసు విజయవంతంగా రూపొందింది!",
        "why": "ఈ పంట ఎందుకు?",
        "action": "7-రోజుల చర్యా ప్రణాళిక",
        "yield_price": "అంచనా దిగుబడి & ధర",
        "profit": "లాభ గణన",
        "risks": "ప్రమాదాలు & హెచ్చరికలు",
    },
    "mr": {
        "title": "स्मार्ट पीक सल्लागार",
        "subtitle": "तुमच्या ठिकाणानुसार आणि शेतीच्या परिस्थितीनुसार पीक शिफारसी मिळवा",
        "tagline": "स्थानाधारित पद्धत - मातीची हाताने माहिती द्यावी लागत नाही!",
        "location": "स्थान तपशील",
        "planting_date": "तुम्हाला केव्हा लागवड करायची आहे?",
        "soil_type": "मातीचा प्रकार (ऐच्छिक)",
        "last_crop": "मागील हंगामातील पीक",
        "water": "पाण्याची उपलब्धता",
        "electricity": "वीज उपलब्धता",
        "land": "जमिनीचे आकारमान (अचूक एकर)",
        "budget": "गुंतवणूक बजेट",
        "market": "बाजार प्रवेश",
        "fetch": "माती आणि हवामान डेटा मिळवा",
        "recommend": "पीक शिफारस मिळवा",
        "help": "मदत आणि माहिती",
        "about": "या अॅपबद्दल",
        "how_to_use": "कसे वापरायचे",
        "tips": "चांगल्या निकालांसाठी टिप्स",
        "season_kharif": "खरीप: जून-सप्टेंबर",
        "season_rabi": "रब्बी: ऑक्टोबर-जानेवारी",
        "season_zaid": "जायद: फेब्रुवारी-मे",
        "location_warn": "डेटा मिळवण्यासाठी पिनकोड आणि जिल्हा भरा",
        "data_ok": "डेटा यशस्वीरित्या मिळाला",
        "need_fetch": "कृपया आधी माती आणि हवामान डेटा मिळवा",
        "rec_ok": "शिफारस यशस्वीरित्या तयार झाली!",
        "why": "हे पीक का?",
        "action": "7 दिवसांची कृती योजना",
        "yield_price": "अपेक्षित उत्पादन आणि किंमत",
        "profit": "नफा गणना",
        "risks": "जोखीम आणि इशारे",
    },
    "ta": {
        "title": "ஸ்மார்ட் பயிர் ஆலோசகர்",
        "subtitle": "உங்கள் இருப்பிடம் மற்றும் விவசாய நிலைமைகளின் அடிப்படையில் பயிர் பரிந்துரைகள் பெறுங்கள்",
        "tagline": "இடஅடிப்படையிலான அணுகுமுறை - மண் விவரம் கையால் தேவையில்லை!",
        "location": "இட விவரங்கள்",
        "planting_date": "நீங்கள் எப்போது நடவு செய்ய விரும்புகிறீர்கள்?",
        "soil_type": "மண் வகை (விருப்பம்)",
        "last_crop": "கடந்த பருவப் பயிர்",
        "water": "நீர் கிடைப்பது",
        "electricity": "மின்சாரம் கிடைப்பது",
        "land": "நில அளவு (துல்லிய ஏக்கர்)",
        "budget": "முதலீட்டு பட்ஜெட்",
        "market": "சந்தை அணுகல்",
        "fetch": "மண் மற்றும் வானிலை தரவைப் பெறுங்கள்",
        "recommend": "பயிர் பரிந்துரை பெறுங்கள்",
        "help": "உதவி & தகவல்",
        "about": "இந்த செயலி பற்றி",
        "how_to_use": "எப்படி பயன்படுத்துவது",
        "tips": "சிறந்த முடிவுகளுக்கான குறிப்புகள்",
        "season_kharif": "காரீப்: ஜூன்-செப்டம்பர்",
        "season_rabi": "ரபி: அக்டோபர்-ஜனவரி",
        "season_zaid": "சைத்: பிப்ரவரி-மே",
        "location_warn": "தரவைப் பெற பின்கோடு மற்றும் மாவட்டத்தை நிரப்பவும்",
        "data_ok": "தரவு வெற்றிகரமாக பெறப்பட்டது",
        "need_fetch": "முதலில் மண் மற்றும் வானிலை தரவைப் பெறுங்கள்",
        "rec_ok": "பரிந்துரை வெற்றிகரமாக உருவாக்கப்பட்டது!",
        "why": "ஏன் இந்தப் பயிர்?",
        "action": "7 நாள் செயல் திட்டம்",
        "yield_price": "எதிர்பார்க்கப்படும் விளைச்சல் & விலை",
        "profit": "லாபக் கணக்கு",
        "risks": "ஆபத்துகள் & எச்சரிக்கைகள்",
    },
    "ur": {
        "title": "اسمارٹ فصل مشیر",
        "subtitle": "اپنے مقام اور زرعی حالات کی بنیاد پر فصل کی سفارشات حاصل کریں",
        "tagline": "مقام پر مبنی طریقہ - مٹی کی دستی معلومات کی ضرورت نہیں!",
        "location": "مقام کی تفصیلات",
        "planting_date": "آپ کب بوائی کرنا چاہتے ہیں؟",
        "soil_type": "مٹی کی قسم (اختیاری)",
        "last_crop": "پچھلی فصل",
        "water": "پانی کی دستیابی",
        "electricity": "بجلی کی دستیابی",
        "land": "زمین کا سائز (درست ایکڑ)",
        "budget": "سرمایہ کاری بجٹ",
        "market": "مارکیٹ تک رسائی",
        "fetch": "مٹی اور موسم کا ڈیٹا حاصل کریں",
        "recommend": "فصل کی سفارش حاصل کریں",
        "help": "مدد اور معلومات",
        "about": "اس ایپ کے بارے میں",
        "how_to_use": "استعمال کیسے کریں",
        "tips": "بہتر نتائج کے لیے مشورے",
        "season_kharif": "خریف: جون-ستمبر",
        "season_rabi": "ربیع: اکتوبر-جنوری",
        "season_zaid": "زائد: فروری-مئی",
        "location_warn": "ڈیٹا کے لیے پن کوڈ اور ضلع بھریں",
        "data_ok": "ڈیٹا کامیابی سے حاصل ہوگیا",
        "need_fetch": "براہ کرم پہلے مٹی اور موسم کا ڈیٹا حاصل کریں",
        "rec_ok": "سفارش کامیابی سے تیار ہوگئی!",
        "why": "یہ فصل کیوں؟",
        "action": "7 دن کا عملی منصوبہ",
        "yield_price": "متوقع پیداوار اور قیمت",
        "profit": "منافع کا حساب",
        "risks": "خطرات اور انتباہات",
    },
    "gu": {
        "title": "સ્માર્ટ પાક સલાહકાર",
        "subtitle": "તમારા સ્થાન અને ખેતીની પરિસ્થિતિઓના આધારે પાક ભલામણો મેળવો",
        "tagline": "સ્થાન આધારિત અભિગમ - માટીની મેન્યુઅલ માહિતીની જરૂર નથી!",
        "location": "સ્થાન વિગતો",
        "planting_date": "તમે ક્યારે વાવણી કરવા માંગો છો?",
        "soil_type": "માટીનો પ્રકાર (વૈકલ્પિક)",
        "last_crop": "ગયા સિઝનનો પાક",
        "water": "પાણીની ઉપલબ્ધતા",
        "electricity": "વીજળીની ઉપલબ્ધતા",
        "land": "જમીનનું કદ (ચોક્કસ એકર)",
        "budget": "રોકાણ બજેટ",
        "market": "બજાર સુધી પહોંચ",
        "fetch": "માટી અને હવામાન ડેટા મેળવો",
        "recommend": "પાકની ભલામણ મેળવો",
        "help": "મદદ અને માહિતી",
        "about": "આ એપ વિશે",
        "how_to_use": "કેવી રીતે વાપરવું",
        "tips": "સારા પરિણામ માટે સૂચનો",
        "season_kharif": "ખરીફ: જૂન-સપ્ટેમ્બર",
        "season_rabi": "રબી: ઓક્ટોબર-જાન્યુઆરી",
        "season_zaid": "જાયદ: ફેબ્રુઆરી-મે",
        "location_warn": "ડેટા મેળવવા માટે પિનકોડ અને જિલ્લો ભરો",
        "data_ok": "ડેટા સફળતાપૂર્વક મળ્યું",
        "need_fetch": "કૃપા કરીને પહેલા માટી અને હવામાન ડેટા મેળવો",
        "rec_ok": "ભલામણ સફળતાપૂર્વક તૈયાર થઈ!",
        "why": "આ પાક કેમ?",
        "action": "7 દિવસની કાર્ય યોજના",
        "yield_price": "અપેક્ષિત ઉત્પાદન અને કિંમત",
        "profit": "નફા ગણતરી",
        "risks": "જોખમો અને ચેતવણીઓ",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

nav1, nav2 = st.columns([9, 1])
with nav1:
    st.markdown(f"# {t('title')}")
with nav2:
    current_lang_name = [k for k, v in LANGS.items() if v == st.session_state.lang][0]
    chosen_lang_name = st.selectbox("🌐", list(LANGS.keys()), index=list(LANGS.keys()).index(current_lang_name), label_visibility="collapsed")
    chosen_lang = LANGS[chosen_lang_name]
    if chosen_lang != st.session_state.lang:
        st.session_state.lang = chosen_lang
        st.rerun()

st.markdown(f"### {t('subtitle')}")
st.markdown(f"_{t('tagline')}_")

if "soil" not in st.session_state:
    st.session_state.soil = None
if "weather" not in st.session_state:
    st.session_state.weather = None
if "season" not in st.session_state:
    st.session_state.season = None

st.divider()

st.header(f"📍 {t('location')}")
col1, col2 = st.columns(2)

with col1:
    pincode = st.text_input("Pincode (6 digits)", placeholder="e.g., 700001", help="Pincode gives exact location for soil data")
    state = st.selectbox("State", ["West Bengal", "Bihar", "Odisha", "Assam", "Jharkhand", "UP", "MP", "Rajasthan", "Others"])

with col2:
    district = st.text_input("District Name", placeholder="e.g., Kolkata, Darjeeling", help="For fetching weather data")
    village = st.text_input("Village Name (optional)", placeholder="e.g., Alipur")

st.divider()

st.header(f"📆 {t('planting_date')}")
planting_date = st.date_input(
    "Select your planting date",
    value=date.today(),
    min_value=date.today(),
    max_value=date.today() + pd.Timedelta(days=365),
    help="Choose the future date when you plan to sow."
)

season, season_desc = detect_season_from_date(planting_date)
st.success(f"✅ Season detected: **{season}** ({season_desc})")
st.caption(f"{t('season_kharif')} | {t('season_rabi')} | {t('season_zaid')}")

st.divider()

st.header(f"🟫 {t('soil_type')}")
soil_type = st.selectbox(
    "What type of soil do you have?",
    [
        "🟡 Alluvial (river soil - good for most crops)",
        "🔴 Black soil (good for cotton, sugarcane)",
        "🟠 Red soil (good for pulses, millets)",
        "⚪ Sandy soil (good for vegetables, flowers)",
        "⚫ Clay soil (good for rice, jute)",
        "❓ Don't know (we'll estimate from pincode)",
    ],
)

st.divider()

st.header(f"🌱 {t('last_crop')}")
last_crop = st.selectbox(
    "What did you grow in the last season?",
    ["Rice", "Wheat", "Maize", "Pulses (gram, arhar)", "Cotton", "Sugarcane", "Vegetables", "Nothing (fallow land)", "Don't know"],
)

st.divider()

st.header(f"💧 {t('water')}")
water_source = st.selectbox(
    "What water source do you have?",
    ["🚰 River/Canal (consistent water)", "🌧️ Rain-only (no irrigation)", "⚡ Electric pump (well/borewell)", "☀ Solar pump (well/borewell)", "❌ No reliable water source"],
)

st.divider()

st.header(f"⚡ {t('electricity')}")
electricity = st.selectbox(
    "Electricity availability?",
    ["✅ Reliable (24-hour power)", "⚠️ Partial (8-12 hours/day)", "❌ Unreliable (less than 8 hours)", "☀ Solar available"],
)

st.divider()

st.header(f"📏 {t('land')}")
land_acres = st.number_input("How many acres of land do you have?", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
if land_acres < 0.5:
    land_category = "🟢 Micro (< 0.5 acre) - Marginal farmer"
elif land_acres < 1:
    land_category = "🟡 Small (0.5 - 1 acre) - Small farmer"
elif land_acres < 2.5:
    land_category = "🟠 Medium (1 - 2.5 acres) - Average farmer"
elif land_acres < 5:
    land_category = "🔴 Large (2.5 - 5 acres) - Large farmer"
elif land_acres < 10:
    land_category = "⚪ Very Large (5 - 10 acres) - Major farmer"
else:
    land_category = "⬛ Estate (> 10 acres) - Commercial farmer"
st.success(f"📊 Category: {land_category}")

st.divider()

st.header(f"💰 {t('budget')}")
budget = st.selectbox(
    "How much can you invest?",
    ["🟢 Low budget (₹5,000 - ₹20,000)", "🟡 Medium budget (₹20,000 - ₹50,000)", "🔴 High budget (₹50,000+)"],
)

st.divider()

st.header(f"🛒 {t('market')}")
market_access = st.selectbox(
    "How far is the nearest market/mandi?",
    ["🟢 Very close (< 5 km)", "🟡 Moderate (5-15 km)", "🔴 Far (15-30 km)", "❌ Very far (> 30 km)"],
)

st.divider()

st.header("🔍 Step 10: Fetch Data")
if pincode and district:
    if st.button(t("fetch")):
        try:
            soil = get_soil_by_pincode(pincode)
            weather = get_weather_by_district(district, state, planting_date, season)
            st.session_state.soil = soil
            st.session_state.weather = weather
            st.session_state.season = season

            st.success(f"✅ {t('data_ok')} for Pincode: {pincode}, District: {district}")

            st.subheader("🟫 Soil Data")
            st.table(pd.DataFrame({
                "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "pH", "Soil Type"],
                "Value": [
                    f"{soil['nitrogen']} mg/kg",
                    f"{soil['phosphorus']} mg/kg",
                    f"{soil['potassium']} mg/kg",
                    soil["pH"],
                    soil["soil_type"],
                ],
            }))

            st.subheader(f"🌦️ Expected Weather for {planting_date.strftime('%d %B %Y')}")

            st.table(pd.DataFrame({
                "Parameter": ["Temperature", "Humidity", "Rainfall"],
                "Value": [
                    f"{weather['temperature']}°C",
                    f"{weather['humidity']}%",
                    f"{weather['rainfall']} mm",
                ],
            }))
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
else:
    st.warning(f"⚠️ {t('location_warn')}")

st.divider()

st.header("🌱 Step 11: Get Recommendation")
all_inputs_filled = pincode and district and planting_date and water_source and land_acres and budget

if st.button(t("recommend"), disabled=not all_inputs_filled):
    if st.session_state.soil is None or st.session_state.weather is None:
        st.error(f"⚠️ {t('need_fetch')}")
    else:
        inputs = {
            "pincode": pincode,
            "district": district,
            "state": state,
            "village": village,
            "planting_date": planting_date,
            "season": season,
            "soil_type": soil_type,
            "last_crop": last_crop,
            "water_source": water_source,
            "electricity": electricity,
            "land_acres": land_acres,
            "land_category": land_category,
            "budget": budget,
            "market_access": market_access,
            "soil": st.session_state.soil,
            "weather": st.session_state.weather,
        }

        try:
            result = predict_crop(inputs)
            lang = st.session_state.lang

            explanation = result["explanation"]
            action_plan = result["action_plan"]
            risks = result["risks"]

            if lang != "en":
                explanation = translate_text(explanation, lang)
                action_plan = translate_text(action_plan, lang)
                risks = translate_text(risks, lang)

            st.success(f"🌾 Recommended Crop: **{result['crop']}**")
            st.info(f"📏 Your Land: {land_acres} acres ({land_category})")

            with st.expander(f"💡 {t('why')}"):
                st.markdown(explanation)

            with st.expander(f"📅 {t('action')}"):
                st.markdown(action_plan)

            with st.expander(f"📊 {t('yield_price')}"):
                st.write(f"**Yield:** {result['yield']}")
                st.write(f"**Market Price:** ₹{result['price']} per kg")

            with st.expander(f"💰 {t('profit')}"):
                profit = calculate_profit(result["crop"], land_acres, budget)
                st.metric("📈 Expected Revenue", profit["revenue"])
                st.metric("💸 Total Cost", profit["cost"])
                st.metric("💵 Net Profit", profit["profit"])
                st.metric("🌾 Profit Per Acre", profit["profit_per_acre"])

            with st.expander(f"⚠️ {t('risks')}"):
                st.markdown(risks)

            st.balloons()
            st.success(f"✅ {t('rec_ok')}")
        except Exception as e:
            st.error(f"❌ Error generating recommendation: {e}")

st.divider()
st.header(f"❓ {t('help')}")

with st.expander(f"📖 {t('about')}"):
    st.markdown("""
    ### Smart Crop Advisory Assistant

    - Recommends crop based on location, soil, weather, and constraints
    - No manual soil input required
    - Auto-detects season from planting date
    - Calculates exact yield and profit based on land size
    """)

with st.expander(f"🔧 {t('how_to_use')}"):
    st.markdown("""
    1. Enter pincode, district, and state
    2. Choose planting date
    3. Enter land size and budget
    4. Fetch soil and weather data
    5. Get crop recommendation
    """)

with st.expander(f"💡 {t('tips')}"):
    st.markdown("""
    - Enter exact pincode for accurate soil data
    - Choose realistic planting date
    - Enter exact land acres
    - Provide honest budget constraints
    - Check market distance for perishable crops
    """)

st.divider()
st.caption("Built for Indian farmers 🌾 | Final-year project | ML + GenAI + NLP")
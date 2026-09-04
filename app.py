"""
Med Amicus — Prototype App (v2)
Qutuhal InnovateX 2.0 | Innovation Builders Category
For: Vihaan Vyas Naidu

NEW IN THIS VERSION
- Multilingual UI + voice: a language picker at the top changes both the
  screen text and what the "Listen" buttons say out loud.
- Prescription Scanner tab: take a photo (or upload one) of a prescription,
  and the app reads the text off it and automatically looks up any medicine
  names it recognizes from your dataset.

BEFORE RUNNING
- This is a single, self-contained file — the dataset is built in, no
  separate data folder needed.
- Run with: streamlit run app.py   (NOT python3 app.py)
- First time the Scanner tab is used, easyocr downloads its recognition
  model (~100MB) — this only happens once and needs internet the first time.
"""

# Fixes a common crash on Apple Silicon Macs where two libraries the OCR
# scanner depends on (torch and opencv) conflict over a shared component.
# Must be set before those libraries are imported anywhere below.
import os
import base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fixes "certificate verify failed" errors on macOS when downloading the
# OCR model. Python.org installs of Python don't automatically trust the
# Mac's certificates — this points Python at the certifi package's trusted
# certificate list instead, so downloads work without extra setup.
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))

import streamlit as st
import pandas as pd
from gtts import gTTS
import difflib
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------------------
st.set_page_config(page_title="Med Amicus", page_icon="💊", layout="centered")

# Adjustable text size — read before the CSS below so the very first paint
# already reflects whatever the person chose last time.
if "text_size" not in st.session_state:
    st.session_state.text_size = "Medium"

SIZE_MAP = {
    "Small": {"base": 18, "h1": 34, "h2": 26, "h3": 20},
    "Medium": {"base": 20, "h1": 40, "h2": 30, "h3": 24},
    "Large": {"base": 24, "h1": 46, "h2": 34, "h3": 28},
}
_sz = SIZE_MAP[st.session_state.text_size]

@st.cache_resource
def get_chime_b64():
    with open("data/chime.wav", "rb") as f:
        return base64.b64encode(f.read()).decode()

def play_chime():
    st.markdown(
        f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{get_chime_b64()}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )

_CSS_TEMPLATE = """
<style>
    html, body, [class*="css"]  { font-size: __BASE__px; }
    h1 { font-size: __H1__px !important; color: #0F6E56; }
    h2 { font-size: __H2__px !important; color: #0F6E56; }
    h3 { font-size: __H3__px !important; }
    .stButton > button {
        font-size: __BASE_PLUS2__px !important;
        padding: 18px 20px !important;
        border-radius: 12px !important;
        min-height: 56px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { font-size: __H3__px !important; }
    input[type="checkbox"] { accent-color: #0F6E56 !important; }
    div[data-testid="stCheckbox"] label span[data-testid="stMarkdownContainer"] p { font-size: __BASE__px !important; }
    [data-baseweb="checkbox"] svg { fill: #0F6E56 !important; }
    [data-baseweb="checkbox"] > div:first-child { border-color: #0F6E56 !important; }
    [data-baseweb="checkbox"][aria-checked="true"] > div:first-child { background: #0F6E56 !important; border-color: #0F6E56 !important; }
    button[kind="primary"] {
        background-color: #0F6E56 !important;
        color: #FFFFFF !important;
        border: 2px solid #0F6E56 !important;
    }
    button[kind="primary"]:hover {
        background-color: #085041 !important;
        border-color: #085041 !important;
    }
    .icon-btn-row .stButton > button {
        border-radius: 50% !important;
        width: 56px !important;
        height: 56px !important;
        padding: 0 !important;
        font-size: 26px !important;
        min-height: 56px;
    }
    .med-card {
        background-color: #E7F3F0;
        border: 2px solid #0F6E56;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .alt-card {
        background-color: #FBF2E3;
        border: 2px solid #854F0B;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
</style>
"""

_css = (
    _CSS_TEMPLATE
    .replace("__BASE_PLUS2__", str(_sz["base"] + 2))
    .replace("__BASE__", str(_sz["base"]))
    .replace("__H1__", str(_sz["h1"]))
    .replace("__H2__", str(_sz["h2"]))
    .replace("__H3__", str(_sz["h3"]))
)
st.markdown(_css, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TRANSLATIONS — add a new language by copying a block and translating it
# ----------------------------------------------------------------------
LANG_CODES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta",
    "Malayalam": "ml", "Bengali": "bn",
    "Urdu": "ur", "Nepali": "ne", "French": "fr",
}

TRANSLATIONS = {
    "English": {
        "language": "Language", "tab_home": "🏠 Home", "tab_find": "💊 Find & Buy",
        "tab_scan": "📷 Scan Prescription", "tab_cart": "🛒 Cart",
        "greeting": "Good morning, {name}!", "next_dose": "Next dose",
        "took_this": "I TOOK THIS", "read_aloud": "🔊 Read aloud",
        "marked_taken": "Marked as taken. Well done!", "later_today": "Later today",
        "find_title": "Find your medicine", "select_medicine": "Select a medicine",
        "active_ingredient": "Active ingredient", "also_contains": "Also contains",
        "price": "Price", "add_to_cart": "🛒 Add to cart", "listen": "🔊 Listen",
        "alternatives_title": "Cheaper alternatives with the same composition",
        "you_save": "💰 You save", "no_alternatives": "No listed alternatives for this medicine.",
        "side_effects": "⚠️ Things to watch for", "cart_title": "Your cart",
        "empty_cart": "Your cart is empty. Go to 'Find & Buy' to add a medicine.",
        "qty": "Qty", "remove": "❌ Remove", "total": "Total",
        "confirm_purchase": "✅  CONFIRM PURCHASE",
        "order_placed": "Order placed! Your medicines will be delivered soon.",
        "scan_title": "Scan your prescription",
        "scan_help": "Take a photo of your doctor's prescription, or upload one.",
        "take_photo": "Take a photo", "upload_photo": "Or upload a photo instead",
        "scanning": "Reading your prescription...",
        "found_medicines": "Medicines found in your prescription",
        "no_medicines_found": "Couldn't clearly match any medicines. Try a clearer, well-lit photo.",
        "raw_text": "Text the app read from the photo",
        "side_effects_note": "Shown in English only, to avoid the risk of a mistranslated medical warning.",
        "undo_button": "Undo", "edit_button": "✏️ Edit",
        "text_size_label": "Text size",
        "medicine_removed": "Removed.",
        "meal_breakfast": "Breakfast", "meal_lunch": "Lunch", "meal_dinner": "Dinner",
        "timing_question": "When should this be taken?", "before_food": "Before food", "after_food": "After food", "minutes_label": "Minutes before/after food", "time_of_day_label": "Time of day", "confirm_button": "Confirm",
        "no_medicines_yet": "No medicines added yet. Use + or the camera above to add your first one.",
        "logout": "🚪 Log out",
        "one_dose": "1 dose", "audio_error": "Could not play audio right now. Check your internet connection.", "add_manually": "+ Add a medicine manually", "medicine_name_label": "Medicine name", "add_button": "Add", "custom_added": "Added! Select it from the dropdown above.",
        "speech_reminder": "It is time to take {name}, one dose.",
        "speech_medicine": "{name} contains {composition}. The price is {price} rupees.",
        "speech_alternative": "{name} is an alternative, priced at {price} rupees.",
    },
    "Hindi": {
        "language": "भाषा", "tab_home": "🏠 होम", "tab_find": "💊 खोजें और खरीदें",
        "tab_scan": "📷 पर्ची स्कैन करें", "tab_cart": "🛒 कार्ट",
        "greeting": "सुप्रभात, {name}!", "next_dose": "अगली खुराक",
        "took_this": "मैंने ले लिया", "read_aloud": "🔊 ज़ोर से पढ़ें",
        "marked_taken": "लिया गया दर्ज किया गया। शाबाश!", "later_today": "आज बाद में",
        "find_title": "अपनी दवा खोजें", "select_medicine": "एक दवा चुनें",
        "active_ingredient": "सक्रिय घटक", "also_contains": "इसमें यह भी है",
        "price": "कीमत", "add_to_cart": "🛒 कार्ट में डालें", "listen": "🔊 सुनें",
        "alternatives_title": "समान संरचना वाले सस्ते विकल्प",
        "you_save": "💰 आप बचाते हैं", "no_alternatives": "इस दवा का कोई विकल्प सूचीबद्ध नहीं है।",
        "side_effects": "⚠️ ध्यान देने योग्य बातें", "cart_title": "आपका कार्ट",
        "empty_cart": "आपका कार्ट खाली है। दवा जोड़ने के लिए 'खोजें और खरीदें' पर जाएं।",
        "qty": "मात्रा", "remove": "❌ हटाएं", "total": "कुल",
        "confirm_purchase": "✅  खरीद की पुष्टि करें",
        "order_placed": "ऑर्डर हो गया! आपकी दवाइयां जल्द ही पहुंचेंगी।",
        "scan_title": "अपनी पर्ची स्कैन करें",
        "scan_help": "डॉक्टर की पर्ची की फोटो लें, या अपलोड करें।",
        "take_photo": "फोटो लें", "upload_photo": "या फोटो अपलोड करें",
        "scanning": "आपकी पर्ची पढ़ी जा रही है...",
        "found_medicines": "पर्ची में मिली दवाइयां",
        "no_medicines_found": "कोई दवा साफ़ तौर पर नहीं मिली। एक साफ़, अच्छी रोशनी वाली फोटो लें।",
        "raw_text": "फोटो से पढ़ा गया टेक्स्ट",
        "side_effects_note": "गलत अनुवाद से बचने के लिए इसे केवल अंग्रेज़ी में दिखाया गया है।",
        "undo_button": "पूर्ववत करें", "edit_button": "✏️ संपादित करें",
        "text_size_label": "टेक्स्ट का आकार",
        "medicine_removed": "हटा दिया गया।",
        "meal_breakfast": "नाश्ता", "meal_lunch": "दोपहर का भोजन", "meal_dinner": "रात का खाना",
        "timing_question": "यह कब लेना है?", "before_food": "भोजन से पहले", "after_food": "भोजन के बाद", "minutes_label": "भोजन से पहले/बाद कितने मिनट", "time_of_day_label": "दिन का समय", "confirm_button": "पुष्टि करें",
        "no_medicines_yet": "अभी तक कोई दवा नहीं जोड़ी गई। पहली दवा जोड़ने के लिए ऊपर + या कैमरा का उपयोग करें।",
        "logout": "🚪 लॉग आउट",
        "one_dose": "1 खुराक", "audio_error": "अभी ऑडियो नहीं चल सका। अपना इंटरनेट कनेक्शन जांचें।", "add_manually": "+ मैन्युअल रूप से दवा जोड़ें", "medicine_name_label": "दवा का नाम", "add_button": "जोड़ें", "custom_added": "जोड़ा गया! ऊपर ड्रॉपडाउन से चुनें।",
        "speech_reminder": "अब {name} लेने का समय है, एक खुराक।",
        "speech_medicine": "{name} में {composition} है। कीमत {price} रुपये है।",
        "speech_alternative": "{name} एक विकल्प है, जिसकी कीमत {price} रुपये है।",
    },
    "Tamil": {
        "language": "மொழி", "tab_home": "🏠 முகப்பு", "tab_find": "💊 தேடி வாங்கு",
        "tab_scan": "📷 மருந்துச்சீட்டை ஸ்கேன் செய்யவும்", "tab_cart": "🛒 கார்ட்",
        "greeting": "காலை வணக்கம், {name}!", "next_dose": "அடுத்த மருந்து",
        "took_this": "எடுத்துவிட்டேன்", "read_aloud": "🔊 உரக்கப் படிக்க",
        "marked_taken": "எடுத்ததாகக் குறிக்கப்பட்டது!", "later_today": "இன்று பிறகு",
        "find_title": "உங்கள் மருந்தைத் தேடுங்கள்", "select_medicine": "ஒரு மருந்தைத் தேர்ந்தெடுக்கவும்",
        "active_ingredient": "செயலில் உள்ள மூலப்பொருள்", "also_contains": "இதில் உள்ளது",
        "price": "விலை", "add_to_cart": "🛒 கார்ட்டில் சேர்", "listen": "🔊 கேட்க",
        "alternatives_title": "அதே கலவையுடன் மலிவான மாற்றுகள்",
        "you_save": "💰 நீங்கள் மிச்சப்படுத்துவது", "no_alternatives": "இந்த மருந்துக்கு மாற்று இல்லை.",
        "side_effects": "⚠️ கவனிக்க வேண்டியவை", "cart_title": "உங்கள் கார்ட்",
        "empty_cart": "உங்கள் கார்ட் காலியாக உள்ளது.",
        "qty": "அளவு", "remove": "❌ நீக்கு", "total": "மொத்தம்",
        "confirm_purchase": "✅  வாங்குதலை உறுதிசெய்",
        "order_placed": "ஆர்டர் செய்யப்பட்டது! உங்கள் மருந்துகள் விரைவில் வரும்.",
        "scan_title": "உங்கள் மருந்துச்சீட்டை ஸ்கேன் செய்யவும்",
        "scan_help": "மருத்துவரின் மருந்துச்சீட்டின் புகைப்படம் எடுக்கவும் அல்லது பதிவேற்றவும்.",
        "take_photo": "புகைப்படம் எடு", "upload_photo": "அல்லது புகைப்படத்தைப் பதிவேற்று",
        "scanning": "உங்கள் மருந்துச்சீட்டு படிக்கப்படுகிறது...",
        "found_medicines": "மருந்துச்சீட்டில் கண்டறியப்பட்ட மருந்துகள்",
        "no_medicines_found": "எந்த மருந்தும் தெளிவாகக் கண்டறியப்படவில்லை.",
        "raw_text": "படத்திலிருந்து படிக்கப்பட்ட உரை",
        "side_effects_note": "தவறான மொழிபெயர்ப்பு அபாயத்தைத் தவிர்க்க, இது ஆங்கிலத்தில் மட்டும் காட்டப்படுகிறது.",
        "undo_button": "செயல்தவிர்", "edit_button": "✏️ திருத்து",
        "text_size_label": "எழுத்துரு அளவு",
        "medicine_removed": "நீக்கப்பட்டது.",
        "meal_breakfast": "காலை உணவு", "meal_lunch": "மதிய உணவு", "meal_dinner": "இரவு உணவு",
        "timing_question": "இதை எப்போது எடுக்க வேண்டும்?", "before_food": "உணவுக்கு முன்", "after_food": "உணவுக்குப் பின்", "minutes_label": "உணவுக்கு முன்/பின் எத்தனை நிமிடங்கள்", "time_of_day_label": "நாளின் நேரம்", "confirm_button": "உறுதிப்படுத்து",
        "no_medicines_yet": "இன்னும் மருந்துகள் சேர்க்கப்படவில்லை. உங்கள் முதல் மருந்தைச் சேர்க்க மேலே உள்ள + அல்லது கேமராவைப் பயன்படுத்தவும்.",
        "logout": "🚪 வெளியேறு",
        "one_dose": "1 டோஸ்", "audio_error": "இப்போது ஆடியோ இயக்க முடியவில்லை. உங்கள் இணைய இணைப்பைச் சரிபார்க்கவும்.", "add_manually": "+ மருந்தை கைமுறையாக சேர்க்கவும்", "medicine_name_label": "மருந்தின் பெயர்", "add_button": "சேர்", "custom_added": "சேர்க்கப்பட்டது! மேலே உள்ள பட்டியலில் இருந்து தேர்ந்தெடுக்கவும்.",
        "speech_reminder": "இப்போது {name} ஒரு டோஸ் எடுக்க வேண்டிய நேரம்.",
        "speech_medicine": "{name} இல் {composition} உள்ளது. விலை {price} ரூபாய்.",
        "speech_alternative": "{name} ஒரு மாற்று, விலை {price} ரூபாய்.",
    },
    "Malayalam": {
        "language": "ഭാഷ", "tab_home": "🏠 ഹോം", "tab_find": "💊 കണ്ടെത്തി വാങ്ങുക",
        "tab_scan": "📷 കുറിപ്പടി സ്കാൻ ചെയ്യുക", "tab_cart": "🛒 കാർട്ട്",
        "greeting": "സുപ്രഭാതം, {name}!", "next_dose": "അടുത്ത ഡോസ്",
        "took_this": "ഞാൻ ഇത് കഴിച്ചു", "read_aloud": "🔊 ഉറക്കെ വായിക്കുക",
        "marked_taken": "കഴിച്ചതായി അടയാളപ്പെടുത്തി. നന്നായി!", "later_today": "ഇന്ന് പിന്നീട്",
        "find_title": "നിങ്ങളുടെ മരുന്ന് കണ്ടെത്തുക", "select_medicine": "ഒരു മരുന്ന് തിരഞ്ഞെടുക്കുക",
        "active_ingredient": "സജീവ ഘടകം", "also_contains": "ഇതിലും അടങ്ങിയിരിക്കുന്നു",
        "price": "വില", "add_to_cart": "🛒 കാർട്ടിൽ ചേർക്കുക", "listen": "🔊 കേൾക്കുക",
        "alternatives_title": "ഇതേ ഘടകങ്ങളുള്ള വിലകുറഞ്ഞ ബദലുകൾ",
        "you_save": "💰 നിങ്ങൾ ലാഭിക്കുന്നു", "no_alternatives": "ഈ മരുന്നിന് ബദലുകൾ ലിസ്റ്റ് ചെയ്തിട്ടില്ല.",
        "side_effects": "⚠️ ശ്രദ്ധിക്കേണ്ട കാര്യങ്ങൾ", "cart_title": "നിങ്ങളുടെ കാർട്ട്",
        "empty_cart": "നിങ്ങളുടെ കാർട്ട് ശൂന്യമാണ്.",
        "qty": "അളവ്", "remove": "❌ നീക്കം ചെയ്യുക", "total": "ആകെ",
        "confirm_purchase": "✅  വാങ്ങൽ സ്ഥിരീകരിക്കുക",
        "order_placed": "ഓർഡർ ചെയ്തു! നിങ്ങളുടെ മരുന്നുകൾ ഉടൻ എത്തും.",
        "scan_title": "നിങ്ങളുടെ കുറിപ്പടി സ്കാൻ ചെയ്യുക",
        "scan_help": "ഡോക്ടറുടെ കുറിപ്പടിയുടെ ഫോട്ടോ എടുക്കുക അല്ലെങ്കിൽ അപ്‌ലോഡ് ചെയ്യുക.",
        "take_photo": "ഫോട്ടോ എടുക്കുക", "upload_photo": "അല്ലെങ്കിൽ ഫോട്ടോ അപ്‌ലോഡ് ചെയ്യുക",
        "scanning": "നിങ്ങളുടെ കുറിപ്പടി വായിക്കുന്നു...",
        "found_medicines": "കുറിപ്പടിയിൽ കണ്ടെത്തിയ മരുന്നുകൾ",
        "no_medicines_found": "മരുന്നുകളൊന്നും വ്യക്തമായി കണ്ടെത്താനായില്ല. വ്യക്തമായ ഫോട്ടോ എടുക്കുക.",
        "raw_text": "ഫോട്ടോയിൽ നിന്ന് വായിച്ച ടെക്സ്റ്റ്",
        "side_effects_note": "തെറ്റായ വിവർത്തനത്തിന്റെ അപകടസാധ്യത ഒഴിവാക്കാൻ ഇത് ഇംഗ്ലീഷിൽ മാത്രം കാണിക്കുന്നു.",
        "undo_button": "പഴയപടിയാക്കുക", "edit_button": "✏️ എഡിറ്റ് ചെയ്യുക",
        "text_size_label": "ടെക്സ്റ്റ് വലുപ്പം",
        "medicine_removed": "നീക്കം ചെയ്തു.",
        "meal_breakfast": "പ്രഭാതഭക്ഷണം", "meal_lunch": "ഉച്ചഭക്ഷണം", "meal_dinner": "അത്താഴം",
        "timing_question": "ഇത് എപ്പോൾ കഴിക്കണം?", "before_food": "ഭക്ഷണത്തിന് മുമ്പ്", "after_food": "ഭക്ഷണത്തിന് ശേഷം", "minutes_label": "ഭക്ഷണത്തിന് മുമ്പ്/ശേഷം എത്ര മിനിറ്റ്", "time_of_day_label": "ദിവസത്തിലെ സമയം", "confirm_button": "സ്ഥിരീകരിക്കുക",
        "no_medicines_yet": "ഇതുവരെ മരുന്നുകളൊന്നും ചേർത്തിട്ടില്ല. ആദ്യത്തെ മരുന്ന് ചേർക്കാൻ മുകളിലുള്ള + അല്ലെങ്കിൽ ക്യാമറ ഉപയോഗിക്കുക.",
        "logout": "🚪 ലോഗ്ഔട്ട്",
        "one_dose": "1 ഡോസ്", "audio_error": "ഇപ്പോൾ ഓഡിയോ പ്ലേ ചെയ്യാൻ കഴിഞ്ഞില്ല. നിങ്ങളുടെ ഇന്റർനെറ്റ് കണക്ഷൻ പരിശോധിക്കുക.", "add_manually": "+ മരുന്ന് സ്വയം ചേർക്കുക", "medicine_name_label": "മരുന്നിന്റെ പേര്", "add_button": "ചേർക്കുക", "custom_added": "ചേർത്തു! മുകളിലെ ഡ്രോപ്ഡൗണിൽ നിന്ന് തിരഞ്ഞെടുക്കുക.",
        "speech_reminder": "ഇപ്പോൾ {name} ഒരു ഡോസ് കഴിക്കേണ്ട സമയമാണ്.",
        "speech_medicine": "{name} ൽ {composition} അടങ്ങിയിരിക്കുന്നു. വില {price} രൂപയാണ്.",
        "speech_alternative": "{name} ഒരു ബദലാണ്, വില {price} രൂപ.",
    },
    "Bengali": {
        "language": "ভাষা", "tab_home": "🏠 হোম", "tab_find": "💊 খুঁজুন ও কিনুন",
        "tab_scan": "📷 প্রেসক্রিপশন স্ক্যান করুন", "tab_cart": "🛒 কার্ট",
        "greeting": "শুভ সকাল, {name}!", "next_dose": "পরবর্তী ডোজ",
        "took_this": "আমি এটি খেয়েছি", "read_aloud": "🔊 জোরে পড়ুন",
        "marked_taken": "নেওয়া হয়েছে বলে চিহ্নিত। চমৎকার!", "later_today": "আজ পরে",
        "find_title": "আপনার ওষুধ খুঁজুন", "select_medicine": "একটি ওষুধ নির্বাচন করুন",
        "active_ingredient": "সক্রিয় উপাদান", "also_contains": "এতে আরও রয়েছে",
        "price": "দাম", "add_to_cart": "🛒 কার্টে যোগ করুন", "listen": "🔊 শুনুন",
        "alternatives_title": "একই উপাদানযুক্ত সস্তা বিকল্প",
        "you_save": "💰 আপনি সাশ্রয় করছেন", "no_alternatives": "এই ওষুধের কোনো বিকল্প তালিকাভুক্ত নেই।",
        "side_effects": "⚠️ যা লক্ষ্য রাখতে হবে", "cart_title": "আপনার কার্ট",
        "empty_cart": "আপনার কার্ট খালি। ওষুধ যোগ করতে 'খুঁজুন ও কিনুন'-এ যান।",
        "qty": "পরিমাণ", "remove": "❌ সরান", "total": "মোট",
        "confirm_purchase": "✅  ক্রয় নিশ্চিত করুন",
        "order_placed": "অর্ডার সম্পন্ন! আপনার ওষুধ শীঘ্রই পৌঁছে যাবে।",
        "scan_title": "আপনার প্রেসক্রিপশন স্ক্যান করুন",
        "scan_help": "ডাক্তারের প্রেসক্রিপশনের ছবি তুলুন অথবা আপলোড করুন।",
        "take_photo": "ছবি তুলুন", "upload_photo": "অথবা ছবি আপলোড করুন",
        "scanning": "আপনার প্রেসক্রিপশন পড়া হচ্ছে...",
        "found_medicines": "প্রেসক্রিপশনে পাওয়া ওষুধ",
        "no_medicines_found": "কোনো ওষুধ স্পষ্টভাবে খুঁজে পাওয়া যায়নি। একটি স্পষ্ট, ভালো আলোর ছবি তুলুন।",
        "raw_text": "ছবি থেকে পড়া টেক্সট",
        "side_effects_note": "ভুল অনুবাদের ঝুঁকি এড়াতে এটি শুধুমাত্র ইংরেজিতে দেখানো হয়েছে।",
        "undo_button": "পূর্বাবস্থায় ফেরান", "edit_button": "✏️ সম্পাদনা করুন",
        "text_size_label": "লেখার আকার",
        "medicine_removed": "সরানো হয়েছে।",
        "meal_breakfast": "সকালের নাশতা", "meal_lunch": "দুপুরের খাবার", "meal_dinner": "রাতের খাবার",
        "timing_question": "এটি কখন খেতে হবে?", "before_food": "খাবারের আগে", "after_food": "খাবারের পরে", "minutes_label": "খাবারের আগে/পরে কত মিনিট", "time_of_day_label": "দিনের সময়", "confirm_button": "নিশ্চিত করুন",
        "no_medicines_yet": "এখনও কোনো ওষুধ যোগ করা হয়নি। প্রথম ওষুধ যোগ করতে উপরের + অথবা ক্যামেরা ব্যবহার করুন।",
        "logout": "🚪 লগ আউট",
        "one_dose": "১ ডোজ", "audio_error": "এখন অডিও চালানো যায়নি। আপনার ইন্টারনেট সংযোগ পরীক্ষা করুন।", "add_manually": "+ ম্যানুয়ালি ওষুধ যোগ করুন", "medicine_name_label": "ওষুধের নাম", "add_button": "যোগ করুন", "custom_added": "যোগ করা হয়েছে! উপরের ড্রপডাউন থেকে নির্বাচন করুন।",
        "speech_reminder": "এখন {name} একটি ডোজ নেওয়ার সময়।",
        "speech_medicine": "{name} তে {composition} রয়েছে। দাম {price} টাকা।",
        "speech_alternative": "{name} একটি বিকল্প, দাম {price} টাকা।",
    },
    "Urdu": {
        "language": "زبان", "tab_home": "🏠 ہوم", "tab_find": "💊 تلاش کریں اور خریدیں",
        "tab_scan": "📷 نسخہ اسکین کریں", "tab_cart": "🛒 کارٹ",
        "greeting": "صبح بخیر، {name}!", "next_dose": "اگلی خوراک",
        "took_this": "میں نے یہ لے لیا", "read_aloud": "🔊 بلند آواز سے پڑھیں",
        "marked_taken": "لیا گیا نشان زد کر دیا گیا۔ شاباش!", "later_today": "آج بعد میں",
        "find_title": "اپنی دوا تلاش کریں", "select_medicine": "ایک دوا منتخب کریں",
        "active_ingredient": "فعال جزو", "also_contains": "اس میں یہ بھی شامل ہے",
        "price": "قیمت", "add_to_cart": "🛒 کارٹ میں شامل کریں", "listen": "🔊 سنیں",
        "alternatives_title": "اسی ترکیب کے سستے متبادل",
        "you_save": "💰 آپ بچاتے ہیں", "no_alternatives": "اس دوا کا کوئی متبادل درج نہیں ہے۔",
        "side_effects": "⚠️ خیال رکھنے کی باتیں", "cart_title": "آپ کا کارٹ",
        "empty_cart": "آپ کا کارٹ خالی ہے۔ دوا شامل کرنے کے لیے 'تلاش کریں اور خریدیں' پر جائیں۔",
        "qty": "مقدار", "remove": "❌ ہٹائیں", "total": "کل",
        "confirm_purchase": "✅  خریداری کی تصدیق کریں",
        "order_placed": "آرڈر مکمل ہوگیا! آپ کی دوائیں جلد پہنچ جائیں گی۔",
        "scan_title": "اپنا نسخہ اسکین کریں",
        "scan_help": "ڈاکٹر کے نسخے کی تصویر لیں یا اپلوڈ کریں۔",
        "take_photo": "تصویر لیں", "upload_photo": "یا تصویر اپلوڈ کریں",
        "scanning": "آپ کا نسخہ پڑھا جا رہا ہے...",
        "found_medicines": "نسخے میں ملنے والی دوائیں",
        "no_medicines_found": "کوئی دوا واضح طور پر نہیں ملی۔ صاف اور روشن تصویر لیں۔",
        "raw_text": "تصویر سے پڑھا گیا متن",
        "side_effects_note": "غلط ترجمے کے خطرے سے بچنے کے لیے یہ صرف انگریزی میں دکھایا گیا ہے۔",
        "undo_button": "کالعدم کریں", "edit_button": "✏️ ترمیم کریں",
        "text_size_label": "تحریر کا سائز",
        "medicine_removed": "ہٹا دیا گیا۔",
        "meal_breakfast": "ناشتہ", "meal_lunch": "دوپہر کا کھانا", "meal_dinner": "رات کا کھانا",
        "timing_question": "یہ کب لینا ہے؟", "before_food": "کھانے سے پہلے", "after_food": "کھانے کے بعد", "minutes_label": "کھانے سے پہلے/بعد کتنے منٹ", "time_of_day_label": "دن کا وقت", "confirm_button": "تصدیق کریں",
        "no_medicines_yet": "ابھی تک کوئی دوا شامل نہیں کی گئی۔ اپنی پہلی دوا شامل کرنے کے لیے اوپر + یا کیمرہ استعمال کریں۔",
        "logout": "🚪 لاگ آؤٹ",
        "one_dose": "1 خوراک", "audio_error": "ابھی آڈیو نہیں چل سکی۔ اپنا انٹرنیٹ کنکشن چیک کریں۔", "add_manually": "+ دستی طور پر دوا شامل کریں", "medicine_name_label": "دوا کا نام", "add_button": "شامل کریں", "custom_added": "شامل کر دیا گیا! اوپر ڈراپ ڈاؤن سے منتخب کریں۔",
        "speech_reminder": "اب {name} لینے کا وقت ہے، ایک خوراک۔",
        "speech_medicine": "{name} میں {composition} شامل ہے۔ قیمت {price} روپے ہے۔",
        "speech_alternative": "{name} ایک متبادل ہے، جس کی قیمت {price} روپے ہے۔",
    },
    "Nepali": {
        "language": "भाषा", "tab_home": "🏠 गृहपृष्ठ", "tab_find": "💊 खोज्नुहोस् र किन्नुहोस्",
        "tab_scan": "📷 प्रेस्क्रिप्सन स्क्यान गर्नुहोस्", "tab_cart": "🛒 कार्ट",
        "greeting": "शुभ प्रभात, {name}!", "next_dose": "अर्को मात्रा",
        "took_this": "मैले यो लिएँ", "read_aloud": "🔊 ठूलो स्वरमा पढ्नुहोस्",
        "marked_taken": "लिइयो भनी चिन्ह लगाइयो। राम्रो!", "later_today": "आज पछि",
        "find_title": "आफ्नो औषधि खोज्नुहोस्", "select_medicine": "एउटा औषधि छान्नुहोस्",
        "active_ingredient": "सक्रिय तत्व", "also_contains": "यसमा यो पनि छ",
        "price": "मूल्य", "add_to_cart": "🛒 कार्टमा थप्नुहोस्", "listen": "🔊 सुन्नुहोस्",
        "alternatives_title": "उस्तै संरचना भएका सस्ता विकल्पहरू",
        "you_save": "💰 तपाईंले बचत गर्नुहुन्छ", "no_alternatives": "यस औषधिको कुनै विकल्प सूचीबद्ध छैन।",
        "side_effects": "⚠️ ध्यान दिनुपर्ने कुराहरू", "cart_title": "तपाईंको कार्ट",
        "empty_cart": "तपाईंको कार्ट खाली छ। औषधि थप्न 'खोज्नुहोस् र किन्नुहोस्' मा जानुहोस्।",
        "qty": "मात्रा", "remove": "❌ हटाउनुहोस्", "total": "जम्मा",
        "confirm_purchase": "✅  खरिद पुष्टि गर्नुहोस्",
        "order_placed": "अर्डर सम्पन्न भयो! तपाईंको औषधि चाँडै पुग्नेछ।",
        "scan_title": "आफ्नो प्रेस्क्रिप्सन स्क्यान गर्नुहोस्",
        "scan_help": "डाक्टरको प्रेस्क्रिप्सनको फोटो खिच्नुहोस्, वा अपलोड गर्नुहोस्।",
        "take_photo": "फोटो खिच्नुहोस्", "upload_photo": "वा फोटो अपलोड गर्नुहोस्",
        "scanning": "तपाईंको प्रेस्क्रिप्सन पढिँदैछ...",
        "found_medicines": "प्रेस्क्रिप्सनमा भेटिएका औषधिहरू",
        "no_medicines_found": "कुनै औषधि स्पष्ट रूपमा भेटिएन। स्पष्ट, राम्रो उज्यालो फोटो खिच्नुहोस्।",
        "raw_text": "फोटोबाट पढिएको पाठ",
        "side_effects_note": "गलत अनुवादको जोखिम हटाउन यो अंग्रेजीमा मात्र देखाइएको छ।",
        "undo_button": "पूर्ववत गर्नुहोस्", "edit_button": "✏️ सम्पादन गर्नुहोस्",
        "text_size_label": "टेक्स्ट साइज",
        "medicine_removed": "हटाइयो।",
        "meal_breakfast": "बिहानको खाजा", "meal_lunch": "दिउँसोको खाना", "meal_dinner": "बेलुकाको खाना",
        "timing_question": "यो कहिले लिने?", "before_food": "खानाअघि", "after_food": "खानापछि", "minutes_label": "खानाअघि/पछि कति मिनेट", "time_of_day_label": "दिनको समय", "confirm_button": "पुष्टि गर्नुहोस्",
        "no_medicines_yet": "अहिलेसम्म कुनै औषधि थपिएको छैन। पहिलो औषधि थप्न माथिको + वा क्यामेरा प्रयोग गर्नुहोस्।",
        "logout": "🚪 लगआउट",
        "one_dose": "1 मात्रा", "audio_error": "अहिले अडियो बज्न सकेन। तपाईंको इन्टरनेट जडान जाँच्नुहोस्।", "add_manually": "+ म्यानुअल रूपमा औषधि थप्नुहोस्", "medicine_name_label": "औषधिको नाम", "add_button": "थप्नुहोस्", "custom_added": "थपियो! माथिको ड्रपडाउनबाट छान्नुहोस्।",
        "speech_reminder": "अहिले {name} लिने समय भयो, एक मात्रा।",
        "speech_medicine": "{name} मा {composition} छ। मूल्य {price} रुपैयाँ हो।",
        "speech_alternative": "{name} एउटा विकल्प हो, मूल्य {price} रुपैयाँ।",
    },
    "French": {
        "language": "Langue", "tab_home": "🏠 Accueil", "tab_find": "💊 Trouver et acheter",
        "tab_scan": "📷 Scanner l'ordonnance", "tab_cart": "🛒 Panier",
        "greeting": "Bonjour, {name} !", "next_dose": "Prochaine dose",
        "took_this": "JE L'AI PRIS", "read_aloud": "🔊 Lire \u00e0 voix haute",
        "marked_taken": "Marqu\u00e9 comme pris. Bravo !", "later_today": "Plus tard aujourd'hui",
        "find_title": "Trouvez votre m\u00e9dicament", "select_medicine": "S\u00e9lectionnez un m\u00e9dicament",
        "active_ingredient": "Principe actif", "also_contains": "Contient \u00e9galement",
        "price": "Prix", "add_to_cart": "🛒 Ajouter au panier", "listen": "🔊 \u00c9couter",
        "alternatives_title": "Alternatives moins ch\u00e8res avec la m\u00eame composition",
        "you_save": "💰 Vous \u00e9conomisez", "no_alternatives": "Aucune alternative r\u00e9pertori\u00e9e pour ce m\u00e9dicament.",
        "side_effects": "⚠️ Points \u00e0 surveiller", "cart_title": "Votre panier",
        "empty_cart": "Votre panier est vide. Allez dans \u00ab Trouver et acheter \u00bb pour ajouter un m\u00e9dicament.",
        "qty": "Qt\u00e9", "remove": "❌ Retirer", "total": "Total",
        "confirm_purchase": "✅  CONFIRMER L'ACHAT",
        "order_placed": "Commande pass\u00e9e ! Vos m\u00e9dicaments seront livr\u00e9s bient\u00f4t.",
        "scan_title": "Scannez votre ordonnance",
        "scan_help": "Prenez une photo de l'ordonnance de votre m\u00e9decin, ou t\u00e9l\u00e9chargez-en une.",
        "take_photo": "Prendre une photo", "upload_photo": "Ou t\u00e9l\u00e9chargez une photo \u00e0 la place",
        "scanning": "Lecture de votre ordonnance en cours...",
        "found_medicines": "M\u00e9dicaments trouv\u00e9s dans votre ordonnance",
        "no_medicines_found": "Impossible de trouver clairement des m\u00e9dicaments. Essayez une photo plus claire et bien \u00e9clair\u00e9e.",
        "raw_text": "Texte lu par l'application depuis la photo",
        "side_effects_note": "Affiché uniquement en anglais, pour éviter le risque d'une mauvaise traduction médicale.",
        "undo_button": "Annuler", "edit_button": "✏️ Modifier",
        "text_size_label": "Taille du texte",
        "medicine_removed": "Supprimé.",
        "meal_breakfast": "Petit-déjeuner", "meal_lunch": "Déjeuner", "meal_dinner": "Dîner",
        "timing_question": "Quand faut-il le prendre ?", "before_food": "Avant les repas", "after_food": "Après les repas", "minutes_label": "Minutes avant/après les repas", "time_of_day_label": "Heure de la journée", "confirm_button": "Confirmer",
        "no_medicines_yet": "Aucun médicament ajouté pour l'instant. Utilisez + ou l'appareil photo ci-dessus pour ajouter le premier.",
        "logout": "🚪 Se déconnecter",
        "one_dose": "1 dose", "audio_error": "Impossible de lire l'audio pour le moment. Vérifiez votre connexion Internet.", "add_manually": "+ Ajouter un médicament manuellement", "medicine_name_label": "Nom du médicament", "add_button": "Ajouter", "custom_added": "Ajouté ! Sélectionnez-le dans le menu déroulant ci-dessus.",
        "speech_reminder": "Il est temps de prendre {name}, une dose.",
        "speech_medicine": "{name} contient {composition}. Le prix est de {price} roupies.",
        "speech_alternative": "{name} est une alternative, au prix de {price} roupies.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "English"

def t(key):
    """Look up a translated string for the currently selected language."""
    return TRANSLATIONS[st.session_state.lang].get(key, TRANSLATIONS["English"][key])

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
# The dataset now lives in a separate file, data/medicines.csv, so the
# main app.py file doesn't contain one giant unreadable line (some tools,
# including GitHub's uploader, can choke on extremely long single lines).

@st.cache_data
def load_data():
    df = pd.read_csv("data/medicines.csv")
    df = df.rename(columns={
        "price(₹)": "price", "sub 0 cost": "alt0_price",
        "sub 1 cost": "alt1_price", "sub 2 cost": "alt2_price",
    })
    df = df.drop_duplicates(subset="name", keep="first")
    return df

df = load_data()
ALT_PAIRS = [("substitute0", "alt0_price"), ("substitute1", "alt1_price"), ("substitute2", "alt2_price")]

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

def add_to_cart(name, price):
    for item in st.session_state.cart:
        if item["name"] == name:
            item["qty"] += 1
            return
    st.session_state.cart.append({"name": name, "price": float(price), "qty": 1})

def speak(text, filename="voice.mp3"):
    """Converts text to speech in the currently selected language."""
    try:
        lang_code = LANG_CODES.get(st.session_state.lang, "en")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        st.audio(filename)
    except Exception:
        st.error(t("audio_error"))

# ----------------------------------------------------------------------
# OCR — loads once and is reused (loading it fresh every time is slow)
# ----------------------------------------------------------------------
@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"])

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    reader = get_ocr_reader()
    results = reader.readtext(image_np, detail=0)
    return " ".join(results)

def find_medicines_in_text(text, dataframe):
    """Looks for medicine names from the dataset inside the scanned text.
    Tries an exact substring match first, then falls back to a fuzzy
    match on the first word, since OCR often misreads a few letters."""
    text_lower = text.lower()
    names = dataframe["name"].dropna().unique().tolist()

    found = [name for name in names if name.lower() in text_lower]

    if not found:
        words = text_lower.split()
        for name in names:
            first_word = name.split()[0].lower()
            if difflib.get_close_matches(first_word, words, n=1, cutoff=0.75):
                found.append(name)

    return found[:5]  # cap results so the screen doesn't get overwhelming

# ----------------------------------------------------------------------
# SHARED CARD RENDERER — used by both the Find & Buy tab and the Scanner tab
# ----------------------------------------------------------------------
def show_medicine_card(row, key_prefix):
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader(f"🔵 {row['name']}")
    st.write(f"**{t('active_ingredient')}:** {row['short_composition1']}")
    if pd.notna(row.get("short_composition2")):
        st.write(f"**{t('also_contains')}:** {row['short_composition2']}")
    st.write(f"**{t('price')}:** ₹{row['price']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"{t('add_to_cart')}", key=f"add_{key_prefix}_{row['id']}", use_container_width=True):
            add_to_cart(row["name"], row["price"])
            st.toast(f"{row['name']} added to cart")
            st.rerun()
    with col2:
        if st.button(t("listen"), key=f"listen_{key_prefix}_{row['id']}", use_container_width=True):
            speak(t("speech_medicine").format(name=row['name'], composition=row['short_composition1'], price=f"{row['price']:.0f}"))

    if pd.notna(row.get("Consolidated_Side_Effects")):
        with st.expander(t("side_effects")):
            if st.session_state.lang != "English":
                st.caption(t("side_effects_note"))
            st.write(row["Consolidated_Side_Effects"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.write(f"**{t('alternatives_title')}**")
    found_any = False
    for alt_col, price_col in ALT_PAIRS:
        alt_name = row.get(alt_col)
        alt_price = row.get(price_col)
        if pd.isna(alt_name) or alt_name in ("", "#N/A"):
            continue
        found_any = True

        st.markdown('<div class="alt-card">', unsafe_allow_html=True)
        st.write(f"**{alt_name}**")
        if pd.notna(alt_price):
            st.write(f"{t('price')}: ₹{float(alt_price):.2f}")
            if pd.notna(row["price"]) and float(alt_price) < row["price"]:
                savings = row["price"] - float(alt_price)
                st.write(f"{t('you_save')} ₹{savings:.2f}")

        colA, colB = st.columns(2)
        with colA:
            if st.button(t("add_to_cart"), key=f"add_{key_prefix}_{alt_col}_{row['id']}", use_container_width=True):
                price_to_use = float(alt_price) if pd.notna(alt_price) else 0.0
                add_to_cart(alt_name, price_to_use)
                st.toast(f"{alt_name} added to cart")
                st.rerun()
        with colB:
            if st.button(t("listen"), key=f"listen_{key_prefix}_{alt_col}_{row['id']}", use_container_width=True):
                speak(t("speech_alternative").format(name=alt_name, price=alt_price))
        st.markdown('</div>', unsafe_allow_html=True)

    if not found_any:
        st.info(t("no_alternatives"))

# ----------------------------------------------------------------------
# LOGIN GATE — collects name, phone, and email before the app is shown
# ----------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""
    st.session_state.user_phone = ""
    st.session_state.user_email = ""

if not st.session_state.logged_in:
    st.title("💊 Med Amicus")
    st.write("Please enter your details to continue.")

    with st.form("login_form"):
        name_input = st.text_input("Full name")
        phone_input = st.text_input("Phone number")
        email_input = st.text_input("Email address")
        submitted = st.form_submit_button("Continue", use_container_width=True)

    if submitted:
        name_clean = name_input.strip()
        phone_clean = phone_input.strip()
        email_clean = email_input.strip()

        phone_digits = "".join(ch for ch in phone_clean if ch.isdigit())
        phone_valid = len(phone_digits) == 10  # Indian mobile numbers are exactly 10 digits
        email_valid = "@" in email_clean and "." in email_clean.split("@")[-1]

        if not name_clean:
            st.error("Please enter your name.")
        elif not phone_clean or not phone_valid:
            st.error("Please enter a valid phone number.")
        elif not email_clean or not email_valid:
            st.error("Please enter a valid email address.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_name = name_clean
            st.session_state.user_phone = phone_clean
            st.session_state.user_email = email_clean
            st.rerun()

    st.stop()

# ----------------------------------------------------------------------
# TOP BAR — language picker and log out sit side by side, above the tabs
# ----------------------------------------------------------------------
top_left, top_right = st.columns([3, 1])
with top_left:
    st.selectbox(
        f"🌐 {t('language')}", list(TRANSLATIONS.keys()),
        key="lang",
    )
with top_right:
    st.write("")  # spacer so the button lines up with the dropdown, not its label
    if st.button(t("logout"), use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_phone = ""
        st.session_state.user_email = ""
        st.rerun()

st.radio(
    t("text_size_label"), ["Small", "Medium", "Large"],
    format_func=lambda x: {"Small": "A", "Medium": "A+", "Large": "A++"}[x],
    key="text_size", horizontal=True,
)

# ----------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------
tab_home, tab_find, tab_cart = st.tabs([
    t("tab_home"), t("tab_find"), f"{t('tab_cart')} ({sum(item['qty'] for item in st.session_state.cart)})"
])

# ---------------- TAB 1: HOME ----------------
with tab_home:
    first_name = st.session_state.user_name.split()[0] if st.session_state.user_name else ""

    if "my_medicines" not in st.session_state:
        st.session_state.my_medicines = []  # starts empty — nothing is pre-loaded
    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False
    if "show_scan_form" not in st.session_state:
        st.session_state.show_scan_form = False
    if "editing_target" not in st.session_state:
        st.session_state.editing_target = None
    if "last_removed" not in st.session_state:
        st.session_state.last_removed = None
    if "last_spoken_medicine" not in st.session_state:
        st.session_state.last_spoken_medicine = None

    # Apply any pending medicine change BEFORE the selectbox widget below is
    # created — Streamlit does not allow changing a widget's value in the
    # same run after it has already been instantiated once.
    if "pending_home_medicine" in st.session_state:
        new_val = st.session_state["pending_home_medicine"]
        del st.session_state["pending_home_medicine"]
        if new_val is None:
            st.session_state.pop("home_medicine", None)
        else:
            st.session_state.home_medicine = new_val

    # Six ready-made timing presets (before/after food × breakfast/lunch/dinner)
    # so adding a medicine is one tap instead of filling three separate fields.
    PRESET_OPTIONS = []
    PRESET_LOOKUP = {}
    for meal_label in [t("meal_breakfast"), t("meal_lunch"), t("meal_dinner")]:
        for timing_label, default_minutes in [(t("before_food"), 30), (t("after_food"), 15)]:
            label = f"{timing_label} · {meal_label}"
            PRESET_OPTIONS.append(label)
            PRESET_LOOKUP[label] = {"timing": timing_label, "minutes": default_minutes, "time": meal_label}

    col_greet, col_add, col_scan = st.columns([5, 1, 1])
    with col_greet:
        st.title(t("greeting").format(name=first_name))
    with col_add:
        st.markdown('<div class="icon-btn-row">', unsafe_allow_html=True)
        if st.button("➕", key="toggle_add_form", help=t("add_manually"), type="primary"):
            st.session_state.show_add_form = not st.session_state.show_add_form
            st.session_state.show_scan_form = False
        st.markdown('</div>', unsafe_allow_html=True)
    with col_scan:
        st.markdown('<div class="icon-btn-row">', unsafe_allow_html=True)
        if st.button("📷", key="toggle_scan_form", help=t("scan_title"), type="primary"):
            st.session_state.show_scan_form = not st.session_state.show_scan_form
            st.session_state.show_add_form = False
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Undo bar — shows right after a removal, until dismissed or replaced ----
    if st.session_state.last_removed is not None:
        col_msg, col_undo = st.columns([3, 1])
        with col_msg:
            st.info(f"{t('medicine_removed')} {st.session_state.last_removed['name']}")
        with col_undo:
            if st.button(t("undo_button"), key="undo_remove", use_container_width=True):
                st.session_state.my_medicines.append(st.session_state.last_removed)
                st.session_state.last_removed = None
                st.rerun()

    # ---- Manual add form — one name field + one preset dropdown ----
    if st.session_state.show_add_form:
        with st.container(border=True):
            new_name = st.text_input(t("medicine_name_label"), key="home_manual_name")
            preset_choice = st.selectbox(t("timing_question"), PRESET_OPTIONS, key="home_manual_preset")
            if st.button(t("add_button"), key="home_manual_add_btn", use_container_width=True):
                existing_names = [m["name"] for m in st.session_state.my_medicines]
                if new_name.strip() and new_name.strip() not in existing_names:
                    preset = PRESET_LOOKUP[preset_choice]
                    st.session_state.my_medicines.append({"name": new_name.strip(), **preset})
                    st.session_state.show_add_form = False
                    st.session_state.last_removed = None
                    st.toast(t("custom_added"))
                    play_chime()
                    st.rerun()

    # ---- Scan prescription form — one tap adds with a sensible default,
    #      edit afterward from the schedule list if it needs adjusting ----
    if st.session_state.show_scan_form:
        with st.container(border=True):
            st.write(t("scan_help"))
            photo = st.camera_input(t("take_photo"), key="home_scan_camera")
            if photo is None:
                photo = st.file_uploader(t("upload_photo"), type=["png", "jpg", "jpeg"], key="home_scan_upload")

            if photo is not None:
                with st.spinner(t("scanning")):
                    scanned_text = extract_text_from_image(photo)
                    matches = find_medicines_in_text(scanned_text, df)

                with st.expander(t("raw_text")):
                    st.write(scanned_text if scanned_text else "—")

                if matches:
                    st.subheader(t("found_medicines"))
                    existing_names = [m["name"] for m in st.session_state.my_medicines]
                    default_preset = PRESET_LOOKUP[PRESET_OPTIONS[0]]  # "Before food · Breakfast"
                    for name in matches:
                        already_added = name in existing_names
                        col_name, col_add_match = st.columns([3, 1])
                        with col_name:
                            st.write(f"**{name}**")
                            if already_added:
                                st.caption(t("custom_added"))
                        with col_add_match:
                            if not already_added and st.button(
                                t("add_button"), key=f"scan_add_{name}", use_container_width=True
                            ):
                                st.session_state.my_medicines.append({"name": name, **default_preset})
                                st.session_state.last_removed = None
                                st.toast(t("custom_added"))
                                play_chime()
                                st.rerun()
                else:
                    st.warning(t("no_medicines_found"))

    # ---- Today's schedule, or an empty state if nothing's been added ----
    my_medicines = st.session_state.my_medicines
    med_names = [m["name"] for m in my_medicines]

    if not my_medicines:
        st.info(t("no_medicines_yet"))
    else:
        if "home_medicine" not in st.session_state or st.session_state.home_medicine not in med_names:
            st.session_state.home_medicine = med_names[0]

        home_medicine_name = st.selectbox(
            t("select_medicine"), med_names, key="home_medicine"
        )
        current_details = next(m for m in my_medicines if m["name"] == home_medicine_name)

        # Auto-play the spoken reminder the moment a *new* medicine is shown —
        # guarded so it only fires on an actual change, not on every rerun.
        if st.session_state.last_spoken_medicine != home_medicine_name:
            speak(t("speech_reminder").format(name=home_medicine_name))
            st.session_state.last_spoken_medicine = home_medicine_name

        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.subheader(f"⏰ {t('next_dose')}")
        st.write(f"**{home_medicine_name}** — {t('one_dose')}")
        st.caption(
            f"{current_details['timing']} · {current_details['minutes']} min · "
            f"{current_details['time']}"
        )

        # Cheapest alternative, shown right here if this medicine is in the
        # dataset (custom hand-typed names won't have pricing data).
        if home_medicine_name in df["name"].values:
            dataset_row = df[df["name"] == home_medicine_name].iloc[0]
            for alt_col, price_col in ALT_PAIRS:
                alt_name = dataset_row.get(alt_col)
                alt_price = dataset_row.get(price_col)
                if pd.notna(alt_name) and alt_name not in ("", "#N/A") and pd.notna(alt_price):
                    if pd.notna(dataset_row["price"]) and float(alt_price) < dataset_row["price"]:
                        savings = dataset_row["price"] - float(alt_price)
                        st.caption(f"💰 {alt_name} — {t('you_save')} ₹{savings:.2f}")
                    break
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Inline edit — change the timing preset without deleting and re-adding ----
        if st.session_state.editing_target == home_medicine_name:
            new_preset_choice = st.selectbox(
                t("timing_question"), PRESET_OPTIONS, key=f"edit_preset_{home_medicine_name}"
            )
            if st.button(t("confirm_button"), key=f"edit_save_{home_medicine_name}", use_container_width=True):
                preset = PRESET_LOOKUP[new_preset_choice]
                for m in st.session_state.my_medicines:
                    if m["name"] == home_medicine_name:
                        m.update(preset)
                        break
                st.session_state.editing_target = None
                st.toast(t("custom_added"))
                play_chime()
                st.rerun()

        # Keying the checkbox by the current medicine's name means switching to
        # a new medicine always gets a fresh, unchecked checkbox automatically —
        # no manual state resetting needed for the checkbox itself.
        checked = st.checkbox(t("took_this"), key=f"took_checkbox_{home_medicine_name}")

        if checked:
            play_chime()
            if len(med_names) > 1:
                # Advance to the next medicine — only safe to do this when there
                # actually is another one, otherwise the checkbox (same key,
                # since the name never changes) would stay checked forever and
                # loop endlessly.
                current_index = med_names.index(home_medicine_name)
                next_index = (current_index + 1) % len(med_names)
                st.toast(t("marked_taken"))
                st.session_state.pending_home_medicine = med_names[next_index]
                st.rerun()
            else:
                st.success(t("marked_taken"))

        col_listen, col_edit, col_remove = st.columns(3)
        with col_listen:
            if st.button(t("read_aloud"), use_container_width=True):
                speak(t("speech_reminder").format(name=home_medicine_name))
        with col_edit:
            if st.button(t("edit_button"), key=f"edit_home_{home_medicine_name}", use_container_width=True):
                st.session_state.editing_target = (
                    None if st.session_state.editing_target == home_medicine_name else home_medicine_name
                )
                st.rerun()
        with col_remove:
            if st.button(t("remove"), key=f"remove_home_{home_medicine_name}", use_container_width=True):
                st.session_state.last_removed = current_details
                st.session_state.my_medicines = [
                    m for m in st.session_state.my_medicines if m["name"] != home_medicine_name
                ]
                remaining_names = [m["name"] for m in st.session_state.my_medicines]
                st.session_state.pending_home_medicine = remaining_names[0] if remaining_names else None
                st.toast(t("medicine_removed"))
                play_chime()
                st.rerun()

        other_meds = [name for name in med_names if name != home_medicine_name]
        if other_meds:
            st.divider()
            st.subheader(t("later_today"))
            for name in other_meds:
                other_details = next(m for m in my_medicines if m["name"] == name)
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    st.write(
                        f"🕑 — **{name}** ({other_details['timing']}, "
                        f"{other_details['time']})"
                    )
                with col_del:
                    if st.button(t("remove"), key=f"remove_later_{name}", use_container_width=True):
                        st.session_state.last_removed = other_details
                        st.session_state.my_medicines = [
                            m for m in st.session_state.my_medicines if m["name"] != name
                        ]
                        st.toast(t("medicine_removed"))
                        play_chime()
                        st.rerun()


# ---------------- TAB 2: FIND & BUY ----------------
with tab_find:
    st.title(t("find_title"))

    if "custom_medicines" not in st.session_state:
        st.session_state.custom_medicines = []  # list of {"name": ..., "price": ...}

    with st.expander(t("add_manually")):
        custom_name = st.text_input(t("medicine_name_label"), key="custom_name_input")
        custom_price = st.number_input(t("price"), min_value=0.0, step=1.0, key="custom_price_input")
        if st.button(t("add_button"), key="add_custom_medicine"):
            if custom_name.strip():
                st.session_state.custom_medicines.append({"name": custom_name.strip(), "price": float(custom_price)})
                st.success(t("custom_added"))

    custom_names = [m["name"] for m in st.session_state.custom_medicines]
    medicine_names = sorted(df["name"].dropna().unique()) + custom_names
    chosen_name = st.selectbox(t("select_medicine"), medicine_names)

    if chosen_name in custom_names:
        custom_med = next(m for m in st.session_state.custom_medicines if m["name"] == chosen_name)
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.subheader(f"🔵 {custom_med['name']}")
        st.write(f"**{t('price')}:** ₹{custom_med['price']:.2f}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("add_to_cart"), key="add_custom_to_cart", use_container_width=True):
                add_to_cart(custom_med["name"], custom_med["price"])
                st.success(f"{custom_med['name']} → 🛒")
        with col2:
            if st.button(t("listen"), key="listen_custom", use_container_width=True):
                speak(t("speech_medicine").format(name=custom_med['name'], composition="", price=f"{custom_med['price']:.0f}"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        row = df[df["name"] == chosen_name].iloc[0]
        show_medicine_card(row, key_prefix="find")

# ---------------- TAB 3: CART ----------------
with tab_cart:
    st.title(t("cart_title"))

    if len(st.session_state.cart) == 0:
        st.info(t("empty_cart"))
    else:
        total = 0.0
        for i, item in enumerate(st.session_state.cart):
            line_total = item["price"] * item["qty"]
            total += line_total

            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{item['name']}**")
                st.write(f"₹{item['price']:.2f}")
            with col2:
                st.write(f"{t('qty')}: {item['qty']}")
            with col3:
                if st.button(t("remove"), key=f"remove_{i}"):
                    st.session_state.cart.pop(i)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.metric(t("total"), f"₹{total:.2f}")

        if st.button(t("confirm_purchase"), use_container_width=True):
            st.balloons()
            st.success(f"{t('order_placed')} ₹{total:.2f}")
            st.session_state.cart = []

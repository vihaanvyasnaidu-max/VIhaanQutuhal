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

st.markdown("""
<style>
    html, body, [class*="css"]  { font-size: 20px; }
    h1 { font-size: 40px !important; color: #0F6E56; }
    h2 { font-size: 30px !important; color: #0F6E56; }
    h3 { font-size: 24px !important; }
    .stButton > button {
        font-size: 22px !important;
        padding: 18px 20px !important;
        border-radius: 12px !important;
        min-height: 56px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { font-size: 28px !important; }
    input[type="checkbox"] { accent-color: #0F6E56 !important; }
    div[data-testid="stCheckbox"] label span[data-testid="stMarkdownContainer"] p { font-size: 20px !important; }
    [data-baseweb="checkbox"] svg { fill: #0F6E56 !important; }
    [data-baseweb="checkbox"] > div:first-child { border-color: #0F6E56 !important; }
    [data-baseweb="checkbox"][aria-checked="true"] > div:first-child { background: #0F6E56 !important; border-color: #0F6E56 !important; }
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
""", unsafe_allow_html=True)

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
# The dataset is embedded directly below (MEDICINE_CSV) so this single
# file works on its own — no separate data folder or CSV path needed.
import io

MEDICINE_CSV = """id,name,price(₹),Is_discontinued,manufacturer_name,type,pack_size_label,short_composition1,short_composition2,substitute0,sub 0 cost,substitute1,sub 1 cost,substitute2,sub 2 cost,substitute3,substitute4,Consolidated_Side_Effects,use0,use1,use2,use3,use4,Chemical Class,Habit Forming,Therapeutic Class,Action Class
1,Augmentin 625 Duo Tablet,223.42,FALSE,Glaxo SmithKline Pharmaceuticals Ltd,allopathy,strip of 10 tablets,Amoxycillin  (500mg) ,  Clavulanic Acid (125mg),Penciclav 500 mg/125 mg Tablet,88.46,Moxikind-CV 625 Tablet,171.1,Moxiforce-CV 625 Tablet,171.1,Fightox 625 Tablet,Novamox CV 625mg Tablet,"Vomiting, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,,No,ANTI INFECTIVES,
2,Azithral 500 Tablet,132.36,FALSE,Alembic Pharmaceuticals Ltd,allopathy,strip of 5 tablets,Azithromycin (500mg),,Zithrocare 500mg Tablet,96.5,Azax 500 Tablet,79.3,Zady 500 Tablet,111.59,Cazithro 500mg Tablet,Trulimax 500mg Tablet,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
3,Ascoril LS Syrup,118,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 100 ml Syrup,Ambroxol (30mg/5ml) , Levosalbutamol (1mg/5ml) ,Solvin LS Syrup,84.4,Ambrodil-LX Syrup,90.85,Zerotuss XP Syrup,90.85,Capex LS Syrup,Broxum LS Syrup,"Nausea, Vomiting, Diarrhea, Upset stomach, Stomach pain, Allergic reaction, Dizziness, Headache, Rash, Hives, Tremors, Palpitations, Muscle cramp, Increased heart rate",Treatment of Cough with mucus,,,,,,No,RESPIRATORY,
4,Allegra 120mg Tablet,218.81,FALSE,Sanofi India  Ltd,allopathy,strip of 10 tablets,Fexofenadine (120mg),,Lcfex Tablet,61.5,Etofex 120mg Tablet,108,Nexofex 120mg Tablet,98.21,Fexise 120mg Tablet,Histafree 120 Tablet,"Headache, Drowsiness, Dizziness, Nausea",Treatment of Sneezing and runny nose due to allergies,Treatment of Allergic conditions,,,,Diphenylmethane Derivative,No,RESPIRATORY,H1 Antihistaminics (second Generation)
4,Allegra 120mg Tablet,218.81,FALSE,Sanofi India  Ltd,allopathy,strip of 10 tablets,Fexofenadine (120mg),,Lcfex Tablet,61.5,Etofex 120mg Tablet,108,F Din 120mg Tablet,94,Nexofex 120mg Tablet,Fexise 120mg Tablet,"Headache, Drowsiness, Dizziness, Nausea",Treatment of Sneezing and runny nose due to allergies,Treatment of Allergic conditions,,,,Diphenylmethane Derivative,No,RESPIRATORY,H1 Antihistaminics (second Generation)
5,Avil 25 Tablet,10.96,FALSE,Sanofi India  Ltd,allopathy,strip of 15 tablets,Pheniramine (25mg),,Eralet 25mg Tablet,19,,#N/A,,#N/A,,,"Sleepiness, Dryness in mouth",Treatment of Allergic conditions,,,,,Pyridines Derivatives,No,RESPIRATORY,H1 Antihistaminics (First Generation)
6,Allegra-M Tablet,241.48,FALSE,Sanofi India  Ltd,allopathy,strip of 10 tablets,Montelukast (10mg) , Fexofenadine (120mg),Emlukast-FX Tablet,83.25,LCFEX-Mont Tablet,98.5,Fixar 10mg/120mg Tablet,105.27,Histakind-M Tablet,Histafree-M Tablet,"Nausea, Diarrhea, Vomiting, Skin rash, Flu-like symptoms, Headache, Drowsiness, Dizziness",Treatment of Sneezing and runny nose due to allergies,,,,,,No,RESPIRATORY,
7,Amoxyclav 625 Tablet,223.27,FALSE,Abbott,allopathy,strip of 10 tablets,Amoxycillin  (500mg) ,  Clavulanic Acid (125mg),Penciclav 500 mg/125 mg Tablet,88.46,Moxikind-CV 625 Tablet,171.1,Moxiforce-CV 625 Tablet,171.1,Fightox 625 Tablet,Novamox CV 625mg Tablet,"Vomiting, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,,No,ANTI INFECTIVES,
8,Azee 500 Tablet,132.38,FALSE,Cipla Ltd,allopathy,strip of 5 tablets,Azithromycin (500mg),,Zithrocare 500mg Tablet,96.5,Azax 500 Tablet,79.3,Zady 500 Tablet,111.59,Cazithro 500mg Tablet,Trulimax 500mg Tablet,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
9,Atarax 25mg Tablet,85.5,FALSE,Dr Reddy's Laboratories Ltd,allopathy,strip of 15 tablets,Hydroxyzine (25mg),,HD Zine 25mg Tablet,48,Hyzox 25 Tablet,40,Hizet 25mg Tablet,57.2,Hydil 25mg Tablet,Zyzine 25mg Tablet,"Sedation, Nausea, Vomiting, Upset stomach, Constipation",Treatment of Anxiety,Treatment of Skin conditions with inflammation & itching,,,,Piperazine Derivative,No,RESPIRATORY,H1 Antihistaminics (First Generation)
10,Ascoril D Plus Syrup Sugar Free,129,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 100 ml Syrup,Phenylephrine (5mg) , Chlorpheniramine Maleate (2mg) ,Arnikof D Syrup,75,Cofsolve-D Syrup,75,Tucin D Syrup,#N/A,Akof-D Syrup Sugar Free,Krisbro D Syrup,"Nausea, Vomiting, Loss of appetite, Headache",Treatment of Dry cough,,,,,,No,RESPIRATORY,
11,Aciloc 150 Tablet,40.94,FALSE,Cadila Pharmaceuticals Ltd,allopathy,strip of 30 tablets,Ranitidine (150mg),,Zinemac 150 Tablet,5.23,Monoloc 150mg Tablet,4.75,Ranitas 150mg Tablet,5.13,Ranloc 150mg Tablet,Zynol 150mg Tablet,"Headache, Diarrhea, Gastrointestinal disturbance",Treatment of Gastroesophageal reflux disease (Acid reflux),Treatment of Peptic ulcer disease,,,,Aralkylamines Derivative,No,GASTRO INTESTINAL,H2 Receptor Blocker
12,Alex Syrup,129,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 100 ml Syrup,Phenylephrine (5mg/5ml) , Chlorpheniramine Maleate (2mg/5ml) ,Alkof Junior Syrup,61,Respicure-D Syrup,95,Torex Junior Syrup,75,Chericof Syrup,Respicure-D Syrup,"Nausea, Vomiting, Loss of appetite, Headache",Treatment of Dry cough,,,,,,No,RESPIRATORY,
13,Anovate Cream,134.2,FALSE,USV Ltd,allopathy,tube of 20 gm Cream,Phenylephrine (0.10% w/w) , Beclometasone (0.025% w/w) ,Pilo GO Cream,85,PileClear Cream,75,Proctosedyl BD Cream,104.18,,,"Application site reactions (burning, irritation, itching and redness)",Treatment of Piles,,,,,,No,DERMA,
14,Augmentin Duo Oral Suspension,67.2,FALSE,Glaxo SmithKline Pharmaceuticals Ltd,allopathy,bottle of 30 ml Oral Suspension,Amoxycillin  (200mg) ,  Clavulanic Acid (28.5mg),Goldclav Oral Suspension,55.1,Moxiclip Dry Syrup,57,Tervis DS Oral Suspension,58,Bestomax Dry Syrup,Amoxyril-CV Dry Syrup,"Nausea, Vomiting, Abdominal pain, Diarrhea, Allergy, Skin rash",Treatment of Resistance Tuberculosis (TB),Treatment of Bacterial infections,,,,,No,ANTI INFECTIVES,
15,Ambrodil-S Syrup,30.2,FALSE,Aristo Pharmaceuticals Pvt Ltd,allopathy,bottle of 100 ml Syrup,Ambroxol (15mg/5ml) , Salbutamol (1mg/5ml),,#N/A,,#N/A,,#N/A,,,"Headache, Palpitations, Upset stomach, Tremors, Muscle cramp, Allergic reaction, Increased heart rate",Treatment of Cough,,,,,,No,RESPIRATORY,
16,Arkamin Tablet,72.65,FALSE,Torrent Pharmaceuticals Ltd,allopathy,strip of 30 tablets,Clonidine (100mcg),,Albamine 100mcg Tablet,12.55,Arkapres 100 Tablet,15.1,Cloud 100mcg Tablet,15,Closidin 100mcg Tablet,Cata-Dict 0.1 Tablet,"Dizziness, Dryness in mouth, Headache, Nausea, Fatigue, Orthostatic hypotension (sudden lowering of blood pressure on standing), Erectile dysfunction, Enlarged salivary gland",Treatment of Hypertension (high blood pressure),,,,,Imidazoline derivative,No,CARDIAC,Alpha 2-adrenoceptors agonist (Central sympatholytics)
17,Avomine Tablet,55.98,FALSE,Abbott,allopathy,strip of 10 tablets,Promethazine (25mg),,Propazine Tablet,9.75,Progene 25mg Tablet,10.41,Proz 25mg Tablet,12.37,Prometh 25mg Tablet,Emin 25mg Tablet,Unusual production of breast milk in women and men,Treatment of Nausea,Treatment of Vomiting,Treatment of Allergic conditions,Treatment of Motion sickness,,Phenothiazine Derivative,No,GASTRO INTESTINAL,H1 Antihistaminics (First Generation)
18,Asthakind-DX Syrup Sugar Free,70.4,FALSE,Mankind Pharma Ltd,allopathy,bottle of 60 ml Syrup,Phenylephrine (5mg/5ml) , Chlorpheniramine Maleate (2mg/5ml) ,Wytuss-DMR Syrup Mint Sugar Free,75,Dcrocof-DX Syrup Sugar Free,79.9,Broxino-DX Syrup Sugar Free,80,Imotus D Syrup,Brikuff-DX Syrup,"Nausea, Vomiting, Loss of appetite, Headache",Treatment of Dry cough,,,,,,No,RESPIRATORY,
19,Allegra 180mg Tablet,251.2,FALSE,Sanofi India  Ltd,allopathy,strip of 10 tablets,Fexofenadine (180mg),,Lcfex 180 Tablet,83,Fexofen 180mg Tablet,83.9,Mavifex 180mg Tablet,135,Histafree 180 Tablet,Vilofex 180 Tablet,"Headache, Drowsiness, Dizziness, Nausea",Treatment of Sneezing and runny nose due to allergies,Treatment of Allergic conditions,,,,Diphenylmethane Derivative,No,RESPIRATORY,H1 Antihistaminics (second Generation)
20,Albendazole 400mg Tablet,9.58,FALSE,Cadila Pharmaceuticals Ltd,allopathy,strip of 1 Tablet,Albendazole (400mg),,Olworm 400mg Tablet,8.53,Zeebee Tablet,8.49,Zybend Tablet,9.62,Albekem 400mg Tablet,Sezole 400mg Tablet,"Vomiting, Dizziness, Increased liver enzymes, Nausea, Loss of appetite",Treatment of Parasitic infections,,,,,2-Benzimidazolylcarbamic acid esters,No,ANTI INFECTIVES,Antiprotozoal agents
21,Asthalin Syrup,19.04,FALSE,Cipla Ltd,allopathy,bottle of 100 ml Syrup,Salbutamol (2mg/5ml),,Brethmol 2mg/5ml Syrup,#N/A,Salvent 2mg Syrup,15.68,Ralbet 2mg Syrup,15.8,Asthabon 2mg Syrup,VENTORLIN 2MG/5ML SYRUP,"Tremors, Headache, Palpitations, Increased heart rate, Muscle cramp",Treatment of Chronic obstructive pulmonary disease (COPD),,,,,Benzyl Alcohols Derivatives,No,RESPIRATORY,Short acting β2-agonists
22,Alprax 0.25 Tablet,29,FALSE,Torrent Pharmaceuticals Ltd,allopathy,strip of 15 tablets,Alprazolam (0.25mg),,Alltop 0.25mg Tablet,1.76,Alprasafe 0.25mg Tablet,2.5,Nindra 0.25mg Tablet,3.31,Alora 0.25mg Tablet,Exal 0.25mg Tablet,"Lightheadedness, Drowsiness",Treatment of Anxiety,Treatment of Panic disorder,,,,Benzodiazepines Derivative,Yes,NEURO CNS,Benzodiazepines
23,Altraday Capsule SR,128,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 10 capsule sr,Aceclofenac (200mg) , Rabeprazole (20mg),Krd AR 200mg/20mg Capsule SR,94,Rient-A Capsule SR,99,Rabispan-AC Capsule SR,99,Douxrab Ace 200mg/20mg Capsule SR,Rabewan AC 200mg/20mg Capsule SR,"Nausea, Flatulence, Indigestion, Diarrhea, Constipation", Pain relief,,,,,,No,PAIN ANALGESICS,
24,Ativan 2mg Tablet,91.87,FALSE,Pfizer Ltd,allopathy,strip of 30 tablets,Lorazepam (2mg),,Zepnap 2mg Tablet,18.82,Lorel 2mg Tablet,21.17,Texina 2mg Tablet,22.05,Larpose 2mg Tablet,Zelor 2mg Tablet,"Fatigue, Balance disorder (loss of balance), Dizziness, Sleepiness",Treatment of Short term anxiety,Treatment of Anxiety disorder,,,,Benzodiazepines Derivative,Yes,RESPIRATORY,Benzodiazepines
25,Ascoril LS Junior Syrup,96,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 60 ml Syrup,Ambroxol (15mg/5ml) , Levosalbutamol (0.5mg/5ml) ,Bronkolyte Levo PD Syrup,56,Nakuf LS Syrup,60,Cleartuss LS Syrup,71.4,Chericof-LS Junior Syrup,Ventiphylline LS Junior Syrup,"Nausea, Vomiting, Diarrhea, Upset stomach, Stomach pain, Allergic reaction, Dizziness, Headache, Rash, Hives, Tremors, Palpitations, Muscle cramp, Increased heart rate",Treatment of Cough with mucus,,,,,,No,RESPIRATORY,
26,Asthalin 100mcg Inhaler,157.85,FALSE,Cipla Ltd,allopathy,packet of 200 MDI Inhaler,Salbutamol (100mcg),,RHEOLIN 100 MCG INHALER,75.94,Ventorlin Inhaler CFC Free,80.64,Bronkonat 100mcg Inhaler,80.85,Durasal Breathaler 100mcg Inhaler,Asthavent 100mcg Inhaler,"Tachycardia, Tremors, Headache, Palpitations, Increased heart rate, Muscle cramp",Treatment of Chronic obstructive pulmonary disease (COPD),,,,,Benzyl Alcohols Derivatives,No,RESPIRATORY,Short acting β2-agonists
27,Almox 500 Capsule,80.26,FALSE,Alkem Laboratories Ltd,allopathy,strip of 10 capsules,Amoxycillin (500mg),,Tormoxin 500mg Capsule,95,Cipmox 500 Capsule,#N/A,Tidoxyl 500mg Capsule,35.37,Actimox 500mg Capsule,SB Mox 500mg Capsule,"Rash, Vomiting, Allergic reaction, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,Aminopenicillins Penicillins,No,ANTI INFECTIVES,Cell wall active agent -Extended spectrum Penicillin
27,Almox 500 Capsule,80.26,FALSE,Alkem Laboratories Ltd,allopathy,strip of 10 capsules,Amoxycillin (500mg),,Tormoxin 500mg Capsule,95,Cipmox 500 Capsule,#N/A,Tidoxyl 500mg Capsule,35.37,Actimox 500mg Capsule,SB Mox 500mg Capsule,"Rash, Vomiting, Allergic reaction, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,Aminopenicillins {Penicillins},No,ANTI INFECTIVES,Cell wall active agent -Extended spectrum Penicillin
28,Atarax 10mg Tablet,47.91,FALSE,Dr Reddy's Laboratories Ltd,allopathy,strip of 15 tablets,Hydroxyzine (10mg),,Evall 10mg Tablet,14.98,Hydil 10mg Tablet,14.3,Stoprax 10mg Tablet,16,Hynorax 10mg Tablet,Hydrobal 10 Tablet,"Sedation, Nausea, Vomiting, Upset stomach, Constipation",Treatment of Anxiety,Treatment of Skin conditions with inflammation & itching,,,,Piperazine Derivative,No,RESPIRATORY,H1 Antihistaminics (First Generation)
29,Aciloc RD 20 Tablet,77,FALSE,Cadila Pharmaceuticals Ltd,allopathy,strip of 15 tablets,Domperidone (10mg) , Omeprazole (20mg),Zydero 10mg/20mg Tablet,45,Try DM 10mg/20mg Tablet,45,Omtro D 10mg/20mg Tablet,47,Prazowel D 10mg/20mg Tablet,Ospicid D 10mg/20mg Tablet,"Diarrhea, Stomach pain, Dryness in mouth, Headache, Flatulence",Treatment of Gastroesophageal reflux disease (Acid reflux),Treatment of Peptic ulcer disease,,,,,No,GASTRO INTESTINAL,
30,Aldactone Tablet,35.35,FALSE,RPG Life Sciences Ltd,allopathy,strip of 15 tablets,Spironolactone (25mg),,Aldobloc 25mg Tablet,#N/A,Spirix 25mg Tablet,19.35,Spiromax 25mg Tablet,#N/A,Spirolact 25mg Tablet,Spironot 25 Tablet,"Nausea, Vomiting, Leg cramps, Dizziness, Drowsiness, Confusion, Breast enlargement in male, Increased creatinine level in blood", Hypertension (high blood pressure), Edema, Low potassium, Heart failure,,Potassium sparing diuretics,No,CARDIAC,Potassium- sparing Diuretics
31,Allegra Suspension Raspberry & Vanilla,188.78,FALSE,Sanofi India  Ltd,allopathy,bottle of 100 ml Oral Suspension,Fexofenadine (30mg/5ml),,Fexovis 30mg Oral Suspension,50,Fextric Oral Suspension,67,Fexonix Oral Suspension,76,Fext Oral Suspension,Xaria Kidz Oral Suspension Mango,"Headache, Drowsiness, Dizziness, Nausea",Treatment of Sneezing and runny nose due to allergies,Treatment of Allergic conditions,,,,Diphenylmethane Derivative,No,RESPIRATORY,H1 Antihistaminics (second Generation)
32,Atarax Syrup,110.25,TRUE,Dr Reddy's Laboratories Ltd,allopathy,bottle of 100 ml Syrup,Hydroxyzine (10mg),,Hydil 10mg Syrup,49.5,Pru 10mg Syrup,59.9,Hedosat 10mg Syrup,64,Livrox 10mg Syrup,Anzine 10mg Syrup,"Sedation, Nausea, Vomiting, Upset stomach, Constipation",Treatment of Anxiety,Treatment of Skin conditions with inflammation & itching,,,,Piperazine Derivative,No,RESPIRATORY,H1 Antihistaminics (First Generation)
33,Amlokind-AT Tablet,44.7,FALSE,Mankind Pharma Ltd,allopathy,strip of 15 tablets,Amlodipine (5mg) , Atenolol (50mg),Biopress AM Tablet,39,Lupidip-A Tablet,45.7,Amlip AT Tablet,72.6,Amlovas-AT Tablet,Amdepin-AT Tablet,"Sleepiness, Headache, Ankle swelling, Flushing (sense of warmth in the face, ears, neck and trunk), Slow heart rate, Palpitations, Nausea, Edema (swelling), Constipation, Tiredness, Cold extremities",Treatment of Hypertension (high blood pressure),,,,,,No,CARDIAC,
34,Axcer  90mg Tablet,420,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 14 tablets,Ticagrelor (90mg),,Tikacad 90 Tablet,186,Brigrel Tablet,196,Xygrel 90 Tablet,214,Ticastro 90 Tablet,Ticavic 90 Tablet,"Bleeding, Breathlessness", Prevention of heart attack and stroke,,,,,Triazolopyrimidines,No,BLOOD RELATED,P2Y12 inhibitors (ADP receptor)
35,Ativan 1mg Tablet,75.67,FALSE,Pfizer Ltd,allopathy,strip of 30 tablets,Lorazepam (1mg),,Lzepam 1mg Tablet,7.68,Lopam 1mg Tablet,12.48,Zepnap 1mg Tablet,14.78,Lorel 1mg Tablet,Larpose 1mg Tablet,"Dizziness, Weakness, Sedation, Balance disorder (loss of balance)",Treatment of Short term anxiety,Treatment of Anxiety disorder,,,,Benzodiazepines Derivative,Yes,RESPIRATORY,Benzodiazepines
36,Alkasol Oral Solution,122.05,TRUE,Stadmed Pvt Ltd,allopathy,bottle of 100 ml Oral Solution,Disodium Hydrogen Citrate (1.4gm/5ml),,Uridol Oral Solution,#N/A,,#N/A,,#N/A,,,"Stomach pain, Tiredness, Diarrhea, Nausea, Vomiting, Frequent urge to urinate",Treatment of Gout,Treatment of Kidney stone,,,,Carboxylic acid derivative,No,UROLOGY,Uricosuric agent-gout
37,Aldigesic P 100mg/325mg Tablet,110,FALSE,Alkem Laboratories Ltd,allopathy,strip of 15 tablets,Aceclofenac (100mg) , Paracetamol (325mg),Dolostat PC 100 mg/325 mg Tablet,27,Acimol 100 mg/325 mg Tablet,35,Topnac P Tablet,32,Arflur-P Tablet,Ark-AP Tablet,"Nausea, Vomiting, Stomach pain/epigastric pain, Loss of appetite, Heartburn, Diarrhea", Pain relief,,,,,,No,PAIN ANALGESICS,
38,Alfoo 10mg Tablet PR,687.75,FALSE,Dr Reddy's Laboratories Ltd,allopathy,strip of 30 Tablet pr,Alfuzosin (10mg),,DavaIndia Alfuzosin 10mg Tablet PR,30,Alfucal 10mg Tablet PR,102,Alfugress Tablet PR,#N/A,Emzosin Tablet PR,Zyalfa Tablet PR,"Upper respiratory tract infection, Dizziness, Headache, Nausea, Abdominal pain, Impotence",Treatment of Benign prostatic hyperplasia,,,,,Quinazoline Derivative,No,UROLOGY,Uroselective adrenergic receptor(α1a) antagonist
39,Alprax 0.5mg Tablet,66.9,FALSE,Torrent Pharmaceuticals Ltd,allopathy,strip of 15 tablets,Alprazolam (0.5mg),,Alprasafe 0.5mg Tablet,#N/A,Alparazole 0.5mg Tablet,1.9,Alprazol 0.5mg Tablet,15.5,Texidep 0.5mg Tablet,Zolipax 0.5mg Tablet,"Lightheadedness, Drowsiness",Treatment of Anxiety,Treatment of Panic disorder,,,,Benzodiazepines Derivative,Yes,NEURO CNS,Benzodiazepines
40,Arachitol 6L Injection,407.76,FALSE,Abbott,allopathy,packet of 6 injections,Vitamin D3 (600000IU),,Sychitol 600000IU Injection,22,Devita 600000IU Injection,24.4,Huch D3 Injection,25,Dvital 600000IU Injection,Gen-D3 600000IU Injection,"Injection site reactions (pain, swelling, redness), Weakness, Muscle pain, Metallic taste",Treatment of Vitamin D deficiency,Treatment of Osteoporosis,,,,Vitamin D Derivative [cholecalciferol],No,VITAMINS MINERALS NUTRIENTS,Vitamins
41,Anafortan 25 mg/300 mg Tablet,124.56,FALSE,Abbott,allopathy,strip of 15 tablets,Camylofin (25mg) , Paracetamol (300mg),Carespas CP 25mg/300mg Tablet,45,Camyfyl 25mg/300mg Tablet,54.4,Cyclobid CP 25mg/300mg Tablet,64.5,Camylyn-P Tablet,Cyclobrex CP 25mg/300mg Tablet,"Dryness in mouth, Constipation, Blurred vision, Increased heart rate",Treatment of Abdominal pain,,,,,,No,GASTRO INTESTINAL,
42,Alex Junior Syrup,95,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 60 ml Syrup,Chlorpheniramine Maleate (2mg/5ml) , Dextromethorphan Hydrobromide (5mg/5ml),Kuffery-DMR Junior Syrup,#N/A,Rizet DX Jr Syrup Kiwi,39.5,Irinil D Syrup,54,Dexgen Syrup,Abrotus DX Junior Syrup,"Upset stomach, Sleepiness",Treatment of Dry cough,,,,,,No,RESPIRATORY,
43,Azithral 200 Liquid,57.25,FALSE,Alembic Pharmaceuticals Ltd,allopathy,bottle of 15 ml Oral Suspension,Azithromycin (200mg/5ml),,Azam 200 Oral Suspension,42.2,Spazit Suspension,44.57,Azimox 200mg Oral Suspension,45,Uber 200 Oral Suspension,Azee 200mg Dry Syrup,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
44,AB Phylline Capsule,162,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 10 capsules,Acebrophylline (100mg),,Broncocet ACB Capsule,55.67,Broncofil 100mg Capsule,95,Bronet 100mg Capsule,82.5,Axeldox 100 Capsule,Ambrodil-XP Capsule,"Nausea, Headache, Vomiting, Upset stomach, Restlessness", Chronic obstructive pulmonary disease (COPD),,,,,Xanthinic Derivatives,No,RESPIRATORY,Theophylline & its derivatives
45,Althrocin 500 Tablet,110.3,FALSE,Alembic Pharmaceuticals Ltd,allopathy,strip of 10 tablets,Erythromycin (500mg),,Erythromycin Estol 500mg Tablet,19.45,Eryster 500mg Tablet,27.47,Eltocin 500mg Tablet,30.22,Erytho 500mg Tablet,Elucin 500mg Tablet,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
46,Augmentin DDS Suspension,173,FALSE,Glaxo SmithKline Pharmaceuticals Ltd,allopathy,bottle of 30 ml Oral Suspension,Amoxycillin  (400mg/5ml) ,  Clavulanic Acid (57mg/5ml),Bactoclav DS 457 Dry Syrup,127,Extclav-DS Dry Syrup,130,Clavio-DS Oral Suspension,130,Medgud-CV Forte Dry Syrup,Clomaxin-DS Dry Syrup,"Nausea, Vomiting, Abdominal pain, Diarrhea, Allergy, Skin rash",Treatment of Resistant Tuberculosis (TB),Treatment of Bacterial infections,,,,,No,ANTI INFECTIVES,
47,Azicip 500 Tablet,79.43,FALSE,Cipla Ltd,allopathy,strip of 3 tablets,Azithromycin (500mg),,Zithrocare 500mg Tablet,96.5,Azax 500 Tablet,79.3,Zady 500 Tablet,111.59,Cazithro 500mg Tablet,Trulimax 500mg Tablet,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
48,Aldigesic-SP Tablet,120,FALSE,Alkem Laboratories Ltd,allopathy,strip of 10 tablets,Aceclofenac (100mg) , Paracetamol (325mg) ,Asozen SR Plus 100mg/325mg/10mg Tablet,20,Accept-SP Tablet,79,Combifenac SP 100mg/325mg/10mg Tablet,80,Aceloflam SP Tablet,Parofen-S Tablet,"Nausea, Vomiting, Stomach pain, Indigestion, Heartburn, Loss of appetite, Diarrhea", Pain relief,,,,,,No,PAIN ANALGESICS,
49,Amoxycillin 500mg Capsule,31.9,TRUE,Jagsonpal Pharmaceuticals Ltd,allopathy,strip of 10 capsules,Amoxycillin (500mg),,Tormoxin 500mg Capsule,95,Cipmox 500 Capsule,#N/A,Tidoxyl 500mg Capsule,35.37,Actimox 500mg Capsule,SB Mox 500mg Capsule,"Rash, Vomiting, Allergic reaction, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,Aminopenicillins {Penicillins},No,ANTI INFECTIVES,Cell wall active agent -Extended spectrum Penicillin
50,Asthakind Expectorant Sugar Free,64.22,FALSE,Mankind Pharma Ltd,allopathy,bottle of 60 ml Expectorant,Guaifenesin (50mg) , Terbutaline (1.25mg) ,Eascof  Expectorant,42.86,Ventoranz Cough Expectorant,27.25,Eledyl SF Expectorant,60,Ascoril SF Expectorant,,"Nausea, Diarrhea, Bloating, Indigestion, Vomiting, Stomach pain, Sweating, Dizziness, Headache, Skin rash, Hives, Tremors, Increased heart rate",Treatment of Cough with mucus,,,,,,No,RESPIRATORY,
51,Acemiz Plus Tablet,99.5,FALSE,Lupin Ltd,allopathy,strip of 10 tablets,Aceclofenac (100mg) , Paracetamol (325mg),Dolostat PC 100 mg/325 mg Tablet,27,Acimol 100 mg/325 mg Tablet,35,Topnac P Tablet,32,Arflur-P Tablet,Acenext P 100mg/325mg Tablet,"Nausea, Vomiting, Stomach pain/epigastric pain, Loss of appetite, Heartburn, Diarrhea", Pain relief,,,,,,No,PAIN ANALGESICS,
52,Aceclo Plus Tablet,90.85,FALSE,Aristo Pharmaceuticals Pvt Ltd,allopathy,strip of 15 tablets,Aceclofenac (100mg) , Paracetamol (325mg),Dolostat PC 100 mg/325 mg Tablet,27,Acimol 100 mg/325 mg Tablet,35,Topnac P Tablet,32,Arflur-P Tablet,Acenext P 100mg/325mg Tablet,"Nausea, Vomiting, Stomach pain/epigastric pain, Loss of appetite, Heartburn, Diarrhea", Pain relief,,,,,,No,PAIN ANALGESICS,
53,Anobliss Cream,136.68,FALSE,Samarth Life Sciences Pvt Ltd,allopathy,tube of 30 gm Rectal Cream,Lidocaine (1.5% w/w) , Nifedipine (0.3% w/w),Nifedip LA  Cream,104,Anozest Cream,119,Anomed L Cream,136.08,Anorelief Cream,Escot Cream,Anal irritation,Treatment of Anal fissure,,,,,,No,PAIN ANALGESICS,
54,Alex Cough Lozenges Lemon Ginger,108,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,strip of 10 lozenges,Dextromethorphan Hydrobromide (5mg),,CHERICOF 5MG LOZENGES,14,Tossex D 5mg Lozenges,60,,#N/A,,,"Sleepiness, Dizziness, Confusion, Nausea", Dry cough,,,,,Methyl Analog of Dextrorphan,No,RESPIRATORY,Cough suppressants
55,Asthalin Respules,7.07,FALSE,Cipla Ltd,allopathy,packet of 2.5 ml Respules,Salbutamol (2.5mg),,Derihaler 2.5mg Respules,3.68,Asthavent 2.5mg Respules,4.4,Axamol 2.5mg Respules 2.5ml,22,Pneumasal 2.5mg Respules (2.5ml Each),,"Tachycardia, Tremors, Headache, Palpitations, Increased heart rate, Muscle cramp",Treatment of Chronic obstructive pulmonary disease (COPD),,,,,Benzyl Alcohols Derivatives,No,RESPIRATORY,Short acting β2-agonists
56,Avil Injection,21.17,FALSE,Sanofi India  Ltd,allopathy,vial of 10 ml Injection,Pheniramine (22.75mg),,Nicophen 22.75mg Injection,2.16,Instavil 22.75mg Injection,4,Eurovil 22.75mg Injection,6,Kayphen 22.75mg Injection,,"Sleepiness, Dryness in mouth",Treatment of Allergic conditions,,,,,Pyridines Derivatives,No,RESPIRATORY,H1 Antihistaminics (First Generation)
57,Azee 200mg Dry Syrup,52.31,FALSE,Cipla Ltd,allopathy,bottle of 15 ml Oral Suspension,Azithromycin (200mg/5ml),,Azam 200 Oral Suspension,42.2,Spazit Suspension,44.57,Azimox 200mg Oral Suspension,45,Uber 200 Oral Suspension,Azmo 200mg Oral Suspension,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
58,Atorva Tablet,102.31,FALSE,Zydus Cadila,allopathy,strip of 15 tablets,Atorvastatin (10mg),,Atorast 10mg Tablet,#N/A,Zivast 10 Tablet,45.25,Storvas 10 Tablet,102.25,Tonact 10 Tablet,Atorlip 10 Tablet,"Dyspepsia, Abdominal pain, Indigestion, Diarrhea, Joint pain, Nasopharyngitis (inflammation of the throat and nasal passages), Nausea, Pain in extremities, Urinary tract infection, Abnormal liver function tests", High cholesterol,Prevention of Heart attack,,,,Pyrrole & heptanoic acid derivative,No,CARDIAC,HMG CoA inhibitors (statins)
59,Asthakind-LS Expectorant Cola Sugar Free,93.5,FALSE,Mankind Pharma Ltd,allopathy,bottle of 100 ml Expectorant,Ambroxol (30mg/5ml) , Levosalbutamol (1mg/5ml) ,Tarific Expectorant,75,Nakuf LS+ SF Expectorant Cherry Sugar Free,110,Piritexyl-LS SF Expectorant,94,Tuspel LS 1 Expectorant,Macbery LS Expectorant Sugar Free,"Nausea, Vomiting, Diarrhea, Upset stomach, Stomach pain, Allergic reaction, Dizziness, Headache, Rash, Hives, Tremors, Palpitations, Muscle cramp, Increased heart rate",Treatment of Cough with mucus,,,,,,No,RESPIRATORY,
60,Ascoril LS Drops,72,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 15 ml Oral Drops,Ambroxol (7.5mg/ml) , Levosalbutamol (0.25mg/ml) ,Usal LS Drops,44,Levokof Oral Drops,50,Coffact-S Drop,52,Nosicold LS Oral Drops,Zerotuss XP Drop,"Vomiting, Diarrhea, Excessive salivation, Fatigue, Headache, Dizziness, Rash, Numbness of extremity, Nausea, Increased heart rate, Stomach discomfort, Palpitations, Tremors, Muscle cramp",Acute Sore throat,Treatment of Cough with mucus,,,,,No,RESPIRATORY,
61,Azmarda 50mg Tablet,1092.03,FALSE,J B Chemicals and Pharmaceuticals Ltd,allopathy,strip of 14 tablets,Sacubitril (24mg) , Valsartan (26mg),Sacutan 50 Tablet,1050,Sacurise 50 Tablet,#N/A,Arney 50 Tablet,#N/A,Sacuval 50 Tablet,Sacu-V 50 Tablet,"Dizziness, Increased potassium level in blood, Fatigue, Hypotension (low blood pressure)",Treatment of Heart failure,,,,,,No,CARDIAC,
62,Amixide-H Tablet,51,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 10 tablets,Amitriptyline (12.5mg) , Chlordiazepoxide (5mg),Trixide-H Tablet,20,Emotrip 12.5 mg/5 mg Tablet,21.5,Aculip H 12.5mg/5mg Tablet,24.66,Trixide 12.5 mg/5 mg Tablet,Kimitrip 12.5mg/5mg Tablet,"Constipation, Difficulty in urination, Weight gain, Confusion, Orthostatic hypotension (sudden lowering of blood pressure on standing), Tiredness, Blurred vision, Dryness in mouth, Increased heart rate, Uncoordinated body movements, Depression, Memory impairment",Treatment of Depression,,,,,,Yes,NEURO CNS,
63,AB-Flo-N Tablet,207.25,FALSE,Lupin Ltd,allopathy,strip of 10 tablets,Acebrophylline (100mg) , Acetylcysteine (600mg),Oxydex Tablet,210,Pulmotol Tablet,#N/A,Amucoe Nac Tablet,171,Aphyren-N Tablet,Asmasafe-N Tablet,"Vomiting, Heartburn, Stomach pain, Upset stomach, Rash, Hives, Itching, Breathing problems, Nasal inflammation, Increased white blood cell count",Treatment of Chronic obstructive pulmonary disease (COPD),,,,,,No,RESPIRATORY,
64,AF Kit Tablet,110,FALSE,Systopic Laboratories Pvt Ltd,allopathy,strip of 4 tablets,Azithromycin (1000mg) , Ornidazole (750mg) ,Afo Kit 1000 mg/750 mg/150 mg Tablet,41.25,Femiriv 3 Kit Tablet,75,Vaginobact 1000 mg/750 mg/150 mg Tablet,70,,,"Taste change, Vomiting, Headache, Dizziness, Stomach pain, Nausea, Indigestion, Diarrhea, Loss of appetite",Treatment of Syndromic treatment of vaginal discharge,,,,,,No,GYNAECOLOGICAL,
65,Amlokind 5 Tablet,22.15,FALSE,Mankind Pharma Ltd,allopathy,strip of 15 tablets,Amlodipine (5mg),,Amset 5mg Tablet,19,Amlip 5 Tablet,32.36,Avacard 5mg Tablet,29.23,Amcard 5 Tablet,Amlong Tablet,"Headache, Fatigue, Nausea, Abdominal pain, Sleepiness",Treatment of Hypertension (high blood pressure),Prevention of Angina (heart-related chest pain),,,,Dihydropyridinecarboxylic acids derivatives,No,CARDIAC,Calcium channel blockers- Dihydropyridines (DHP)
66,Amlong Tablet,48.55,FALSE,Micro Labs Ltd,allopathy,strip of 15 tablets,Amlodipine (5mg),,Amset 5mg Tablet,19,Amlip 5 Tablet,32.36,Avacard 5mg Tablet,29.23,Amcard 5 Tablet,Camlodip 5mg Tablet,"Headache, Fatigue, Nausea, Abdominal pain, Sleepiness",Treatment of Hypertension (high blood pressure),Prevention of Angina (heart-related chest pain),,,,Dihydropyridinecarboxylic acids derivatives,No,CARDIAC,Calcium channel blockers- Dihydropyridines (DHP)
67,Akt 4 Kit,24.7,FALSE,Lupin Ltd,allopathy,packet of 1 Kit,Isoniazid (300mg) , Rifampicin (450mg) ,RF 4 Kit,13.6,Anticox 4 Kit,22.43,4D Kit,#N/A,Afb 4 Kit,CAVICIN E KIT,"Nausea, Vomiting, Rash, Fever, Dark colored urine, Sweating, Increased sputum production, Salivation, Watery eyes, Peripheral neuropathy (tingling and numbness of feet and hand), Increased liver enzymes, Jaundice, Increased uric acid level in blood, Visual impairment",Treatment of Tuberculosis (TB),,,,,,No,ANTI INFECTIVES,
68,Ascoril D Junior Cough Syrup,98,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 60 ml Syrup,Phenylephrine (5mg/5ml) , Chlorpheniramine Maleate (2mg/5ml) ,Alkof Junior Syrup,61,Respicure-D Syrup,95,Chericof Syrup,118,Respicure-D Syrup,Ataqued Syrup,"Nausea, Vomiting, Loss of appetite, Headache",Treatment of Dry cough,,,,,,No,RESPIRATORY,
69,Amitone 10mg Tablet,19.15,FALSE,Intas Pharmaceuticals Ltd,allopathy,strip of 10 tablets,Amitriptyline (10mg),,Triptop 10mg Tablet,15,Relidep 10mg Tablet,18.03,Tryp 10mg Tablet,22,Amitril 10mg Tablet,Odep 10mg Tablet,"Constipation, Dryness in mouth, Orthostatic hypotension (sudden lowering of blood pressure on standing), Weight gain, Aggressive behavior, Nasal congestion (stuffy nose), Sleepiness, Dizziness, Headache, Decreased libido, Nausea, Fatigue, Confusion, Tremors, Speech disorder, Palpitations, Taste change, Paresthesia (tingling or pricking sensation), Abnormality of voluntary movements, Loss of accommodation, Atrioventricular block, Micturition disorders, Erectile dysfunction, Abnormal ECG, Decreased sodium level in blood",Treatment of Depression, Neuropathic pain,Treatment of Migraine,,,Dibenzocycloheptenes Derivative,No,NEURO CNS,Tricyclic antidepressants
70,Aulin 100mg Tablet,50.56,FALSE,Elder Pharmaceuticals Ltd,allopathy,strip of 10 tablets,Nimesulide (100mg),,Nimprex Tablet,21.67,Pyrimide 100mg Tablet,39,Nicip Tablet,43.45,Nimtech 100mg Tablet,Nalgis 100mg Tablet,"Vomiting, Nausea, Diarrhea", Pain relief,Treatment of Fever,,,,Diphenylethers Derivative,No,PAIN ANALGESICS,NSAID's-Non-Selective COX 1&2 Inhibitors (Others)
71,Amikacin Sulphate 500mg Injection,11.87,TRUE,Sun Pharmaceutical Industries Ltd,allopathy,vial of 2 ml Injection,Amikacin (500mg),,Amilab 500mg Injection,58,Acil 500mg Injection,70,Emica 500mg Injection,75.6,Mika Best 500mg Injection,Ivimicin 500mg Injection,"Increased blood urea, Injection site reactions (pain, swelling, redness)", Bacterial infections,,,,,Aminoglycosides,No,OPHTHAL,Aminoglycosides
72,Ambrodil-LX Syrup,90.85,FALSE,Aristo Pharmaceuticals Pvt Ltd,allopathy,bottle of 100 ml Syrup,Ambroxol (30mg/5ml) , Levosalbutamol (1mg/5ml) ,Solvin LS Syrup,84.4,Zerotuss XP Syrup,90.85,Capex LS Syrup,83,Broxum LS Syrup,Asthalin AX Syrup,"Nausea, Vomiting, Diarrhea, Upset stomach, Stomach pain, Allergic reaction, Dizziness, Headache, Rash, Hives, Tremors, Palpitations, Muscle cramp, Increased heart rate",Treatment of Cough with mucus,,,,,,No,RESPIRATORY,
73,Aquasol A Capsule,29.6,FALSE,USV Ltd,allopathy,bottle of 30 capsules,Vitamin A (25000IU),,,#N/A,,#N/A,,#N/A,,,No common side effects seen,Treatment of Vitamin A deficiency,,,,,Retinoid Derivative,No,VITAMINS MINERALS NUTRIENTS,Vitamins
74,AB Phylline SR 200 Tablet,252,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 10 tablet sr,Acebrophylline (200mg),,Venphylin-SR Tablet,140,Uliphyalline 200 SR Tablet,98,Asmasafe 200mg Tablet SR,145,Acethama 200 SR Tablet,Abiways 200 SR Tablet,"Nausea, Headache, Vomiting, Upset stomach, Restlessness", Chronic obstructive pulmonary disease (COPD),,,,,Xanthinic Derivatives,No,RESPIRATORY,Theophylline & its derivatives
75,Azoran Tablet,106.65,TRUE,RPG Life Sciences Ltd,allopathy,strip of 10 tablets,Azathioprine (50mg),,Transimune 50mg Tablet,67.26,Azap 50 Tablet,75,Zesoris-AZ 50mg Tablet,80,Autorin 50mg Tablet,Azawan Tablet,"Decreased white blood cell count, Increased bleeding tendency, Nausea, Infection, Loss of appetite", Prevention of organ rejection in transplant patients,Treatment of Rheumatoid arthritis,,,,"Nucleoside Analog, and Purines",No,ANTI NEOPLASTICS,Immunosuppressant- Purine analogs
76,Amaryl 1mg Tablet,133.81,FALSE,Sanofi India  Ltd,allopathy,strip of 30 tablets,Glimepiride (1mg),,,#N/A,,#N/A,,#N/A,,,,,,,,,,,,
77,Aztor 10 Tablet,102.25,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 15 tablets,Atorvastatin (10mg),,Atorast 10mg Tablet,#N/A,Zivast 10 Tablet,45.25,Storvas 10 Tablet,102.25,Tonact 10 Tablet,Atorlip 10 Tablet,"Dyspepsia, Abdominal pain, Indigestion, Diarrhea, Joint pain, Nasopharyngitis (inflammation of the throat and nasal passages), Nausea, Pain in extremities, Urinary tract infection, Abnormal liver function tests", High cholesterol,Prevention of Heart attack,,,,Pyrrole & heptanoic acid derivative,No,CARDIAC,HMG CoA inhibitors (statins)
78,Atorva 40 Tablet,239.89,FALSE,Zydus Cadila,allopathy,strip of 10 tablets,Atorvastatin (40mg),,Atorless 40mg Tablet,#N/A,Atchol 40 Tablet,96.22,Lipikind 40 Tablet,116.48,Mactor 40 Tablet,Atorsave 40 Tablet,"Dyspepsia, Abdominal pain, Indigestion, Diarrhea, Joint pain, Nasopharyngitis (inflammation of the throat and nasal passages), Nausea, Pain in extremities, Urinary tract infection, Abnormal liver function tests", High cholesterol,Prevention of Heart attack,,,,Pyrrole & heptanoic acid derivative,No,CARDIAC,HMG CoA inhibitors (statins)
79,Azax 500 Tablet,79.3,FALSE,Sun Pharmaceutical Industries Ltd,allopathy,strip of 3 tablets,Azithromycin (500mg),,Zithrocare 500mg Tablet,96.5,Zady 500 Tablet,111.59,Cazithro 500mg Tablet,62.55,Trulimax 500mg Tablet,Azifast 500 Tablet,"Vomiting, Nausea, Abdominal pain, Diarrhea",Treatment of Bacterial infections,,,,,Macrolides,No,ANTI INFECTIVES,Macrolides
80,Alex Syrup Sugar Free,129,FALSE,Glenmark Pharmaceuticals Ltd,allopathy,bottle of 100 ml Syrup,Phenylephrine (5mg/5ml) , Chlorpheniramine Maleate (2mg/5ml) ,Alkof Junior Syrup,61,Respicure-D Syrup,95,Torex Junior Syrup,75,Chericof Syrup,Respicure-D Syrup,"Nausea, Vomiting, Loss of appetite, Headache",Treatment of Dry cough,,,,,,No,RESPIRATORY,
81,Anxit 0.5 Tablet,47,FALSE,Micro Labs Ltd,allopathy,strip of 15 tablets,Alprazolam (0.5mg),,Alprasafe 0.5mg Tablet,#N/A,Alparazole 0.5mg Tablet,1.9,Alprazol 0.5mg Tablet,15.5,Texidep 0.5mg Tablet,Zolipax 0.5mg Tablet,"Lightheadedness, Drowsiness",Treatment of Anxiety,Treatment of Panic disorder,,,,Benzodiazepines Derivative,Yes,NEURO CNS,Benzodiazepines
82,Anxit 0.25mg Tablet,20.5,FALSE,Micro Labs Ltd,allopathy,strip of 15 tablets,Alprazolam (0.25mg),,Alltop 0.25mg Tablet,1.76,Alprasafe 0.25mg Tablet,2.5,Nindra 0.25mg Tablet,3.31,Alora 0.25mg Tablet,Exal 0.25mg Tablet,"Lightheadedness, Drowsiness",Treatment of Anxiety,Treatment of Panic disorder,,,,Benzodiazepines Derivative,Yes,NEURO CNS,Benzodiazepines
83,Acitrom 2 Tablet,414.65,FALSE,Abbott,allopathy,strip of 30 tablets,Acenocoumarol (2mg),,Nicoz 2mg Tablet,38.95,Nistrom 2mg Tablet,39,Acimalone 2 Tablet,#N/A,Acethromb 2mg Tablet,Cenorol 2mg Tablet,Hemorrhage,Treatment and prevention of Blood clots,,,,,4-hydroxycoumarin Derivative,No,CARDIAC,Vitamin K antagonists
84,Angispan - TR 2.5mg Capsule,198,FALSE,USV Ltd,allopathy,bottle of 25 capsule tr,Nitroglycerin (2.5mg),,Angiplat 2.5 Capsule TR,162.5,Angistat 2.5 Capsule TR,169,Vasovin - XL 2.5 Capsule,187.85,,,"Blurred vision, Decreased blood pressure, Dizziness, Headache, Increased heart rate, Lightheadedness, Paresthesia (tingling or pricking sensation)",Treatment of Angina (heart-related chest pain),,,,,Nitrates {Short acting},No,CARDIAC,NO Donors
85,Azeflo Nasal Spray,420.1,FALSE,Lupin Ltd,allopathy,packet of 7 ml Nasal Spray,Fluticasone Propionate (50mcg) , Azelastine (140mcg),Sarnase 50 mcg/140 mcg Nasal Spray,181.72,Sernase Nasal Spray,189,Armist Nasal Spray,259,Lergiset-AF Nasal Spray,Flotaze Nasal Spray,"Taste change, Nosebleeds, Headache, Cough, Upper respiratory tract infection, Dryness in mouth, Nasopharyngitis (inflammation of the throat and nasal passages), Sinus inflammation, Stomach discomfort, Fungal infection of oropharynx, Tremors, Palpitations, Voice change",Treatment of Sneezing and runny nose due to allergies,,,,,,No,RESPIRATORY,
86,Acemiz -MR Tablet,99.5,FALSE,Lupin Ltd,allopathy,strip of 10 tablets,Aceclofenac (100mg) , Paracetamol (325mg) ,Acimol MR 100mg/325mg/250mg Tablet,65,Kapinac CZ Tablet,62.9,Zoceclo MR 100mg/325mg/250mg Tablet,64,Clanac MR Tablet,Acelyn MR 100mg/325mg/250mg Tablet,"Nausea, Vomiting, Heartburn, Stomach pain, Diarrhea, Loss of appetite, Tiredness, Sleepiness",Treatment of Muscular pain,,,,,,No,PAIN ANALGESICS,
87,Akurit 4 Tablet,82.6,FALSE,Lupin Ltd,allopathy,strip of 10 tablets,Isoniazid (75mg) , Rifampicin (150mg) ,MONITOR 4 TABLET,23.91,Trac 4 Tablet,34,Coxcure 4 Tablet,58,Mycurit 4 mg Tablet,Vicox 4 Tablet,"Nausea, Vomiting, Rash, Fever, Dark colored urine, Sweating, Increased sputum production, Salivation, Watery eyes, Peripheral neuropathy (tingling and numbness of feet and hand), Increased liver enzymes, Jaundice, Increased uric acid level in blood, Visual impairment",Treatment of Tuberculosis (TB),,,,,,No,ANTI INFECTIVES,
88,Aerocort Inhaler,273.77,FALSE,Cipla Ltd,allopathy,packet of 200 MDI Inhaler,Levosalbutamol (50mcg) , Beclometasone (50mcg),,#N/A,,#N/A,,#N/A,,,"Hoarseness of voice, Headache, Dizziness, Pharyngitis, Vomiting, Bronchitis (inflammation of the airways), Dryness in mouth, Cough, Application site irritation, Nausea, Gastrointestinal motility disorder, Thrush, Throat irritation", Asthma,,,,,,No,RESPIRATORY,
89,Adaferin Gel,284,FALSE,Galderma India Pvt Ltd,allopathy,tube of 15 gm Gel,Adapalene (0.1% w/w),,Alene Gel,74.91,Adaple 0.1% Gel,80,Acfree Gel,#N/A,Vandep 0.1% Gel,Admark Gel,"Erythema (skin redness), Dry skin, Skin peeling, Skin burn, Itching", Acne,,,,,Retinoids,No,DERMA,Retinoids- Third generation
90,Acivir 400 DT Tablet,76.55,FALSE,Cipla Ltd,allopathy,strip of 5 tablet dt,Acyclovir (400mg),,Acivex 400mg Tablet DT,88.12,Zoylex 400mg Tablet DT,99,Acloriv 400mg Tablet DT,100,Docvir 400mg Tablet DT,Zoster 400mg Tablet DT,"Headache, Dizziness, Vomiting, Nausea, Fatigue, Fever, Stomach pain, Diarrhea, Skin rash, Photophobia", Herpes Simplex Virus Infections, Chickenpox, Herpes labialis, Shingles, Genital herpes infection,Nucleoside analog,No,ANTI INFECTIVES,Antiviral (Non-HIV) drugs
91,Aptimust Syrup,129.9,FALSE,Mankind Pharma Ltd,allopathy,bottle of 200 ml Syrup,Cyproheptadine (2mg/5ml) , Tricholine Citrate (275mg/5ml) ,Actizer Syrup,75,Pepcip Syrup,78,Cycoline Syrup,80,Yopon  Syrup,Oditril Syrup,"Constipation, Dryness in mouth, Drowsiness, Sleepiness, Blurred vision", Appetite stimulant,,,,,,No,VITAMINS MINERALS NUTRIENTS,
92,Augmentin 1000 Duo Tablet,601.45,FALSE,Glaxo SmithKline Pharmaceuticals Ltd,allopathy,strip of 10 tablets,Amoxycillin  (875mg) ,  Clavulanic Acid (125mg),Moxikind-CV 1gm Tablet,204,Acuclav 1000mg Tablet,188.1,Mega-CV Duo 875mg/125mg Tablet,224,Bactoclav 875 mg/125 mg Tablet,Lmx Forte 875mg/125mg Tablet,"Vomiting, Nausea, Diarrhea",Treatment of Bacterial infections,,,,,,No,ANTI INFECTIVES,
93,Ambrodil Syrup,69.75,FALSE,Aristo Pharmaceuticals Pvt Ltd,allopathy,bottle of 100 ml Syrup,Ambroxol (30mg/5ml),,Ambril Syrup,34.3,Respolite Syrup,38.33,Ambrolene Syrup,47.13,Revibrox Plus 30mg/5ml Syrup,Cofnil AM 30mg Syrup,"Vomiting, Nausea, Upset stomach",Treatment of Respiratory tract disorders associated with viscid mucus,,,,,Phenylmethylamine Derivative,No,RESPIRATORY,Mucolytics
94,Acogut Tablet,239.4,FALSE,Lupin Ltd,allopathy,strip of 15 tablets,Acotiamide (100mg),,Actapro Tablet,120,Dycotiam Tablet,120,Tocamide Tablet,207,Actapro Tablet,Acotrust  Tablet,"Headache, Diarrhea",Treatment of Functional dyspepsia,,,,,Salicylamide Derivative,No,GASTRO INTESTINAL,Anticholinesterase-Prokinetic agent 
95,Atarax Drops,69.75,FALSE,Dr Reddy's Laboratories Ltd,allopathy,bottle of 15 ml Syrup,Hydroxyzine (6mg),,,#N/A,,#N/A,,#N/A,,,"Sedation, Nausea, Vomiting, Upset stomach, Constipation",Treatment of Anxiety,Treatment of Skin conditions with inflammation & itching,,,,Piperazine Derivative,No,RESPIRATORY,H1 Antihistaminics (First Generation)
96,Amlip 5 Tablet,32.36,FALSE,Cipla Ltd,allopathy,strip of 10 tablets,Amlodipine (5mg),,Amset 5mg Tablet,19,Avacard 5mg Tablet,29.23,Amcard 5 Tablet,14.35,Amlong Tablet,Camlodip 5mg Tablet,"Headache, Fatigue, Nausea, Abdominal pain, Sleepiness",Treatment of Hypertension (high blood pressure),Prevention of Angina (heart-related chest pain),,,,Dihydropyridinecarboxylic acids derivatives,No,CARDIAC,Calcium channel blockers- Dihydropyridines (DHP)
97,AntiD 300mcg/ml Injection,3039.42,FALSE,Bharat Serums & Vaccines Ltd,allopathy,vial of 1 Injection,Anti Rh D Immunoglobulin (300mcg/ml),,,#N/A,,#N/A,,#N/A,,,"Fever, Headache, Injection site tenderness, Injection site pain, Feeling of discomfort",Prevention of Infections,,,,,Vaccines,No,VACCINES,Immunoglobulin
98,Alerid Syrup,21.84,FALSE,Cipla Ltd,allopathy,bottle of 30 ml Syrup,Cetirizine (5mg/5ml),,Cetrilix 5mg Syrup,17,Cetmac 5mg/5ml Syrup,17.25,Ralcet 5mg/5ml Syrup,17.95,Mast 1 Syrup,Ezin Syrup,"Nausea, Headache, Muscle pain, Edema (swelling), Sleepiness, Dizziness",Treatment of Allergic conditions,,,,,Phenylmethyl Piperazinyl Derivative,No,RESPIRATORY,H1 Antihistaminics (second Generation)
"""

@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(MEDICINE_CSV))
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
        if name_input.strip():
            st.session_state.logged_in = True
            st.session_state.user_name = name_input.strip()
            st.session_state.user_phone = phone_input.strip()
            st.session_state.user_email = email_input.strip()
            st.rerun()
        else:
            st.error("Please enter your name to continue.")

    st.stop()

# ----------------------------------------------------------------------
# LANGUAGE PICKER — sits above the tabs, applies everywhere
# ----------------------------------------------------------------------
st.selectbox(
    f"🌐 {t('language')}", list(TRANSLATIONS.keys()),
    key="lang",
)

# ----------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------
tab_home, tab_find, tab_scan, tab_cart = st.tabs([
    t("tab_home"), t("tab_find"), t("tab_scan"), f"{t('tab_cart')} ({sum(item['qty'] for item in st.session_state.cart)})"
])

# ---------------- TAB 1: HOME ----------------
with tab_home:
    first_name = st.session_state.user_name.split()[0] if st.session_state.user_name else ""
    col_greet, col_logout = st.columns([4, 1])
    with col_greet:
        st.title(t("greeting").format(name=first_name))
    with col_logout:
        st.write("")  # small vertical spacer to align the button with the title
        if st.button(t("logout")):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_phone = ""
            st.session_state.user_email = ""
            st.rerun()

    # Only the first 3 medicines from the dataset make up today's schedule —
    # matches the original Home screen behaviour, not the full 100-medicine list.
    daily_schedule = df["name"].dropna().unique()[:3].tolist()

    if "home_medicine" not in st.session_state or st.session_state.home_medicine not in daily_schedule:
        st.session_state.home_medicine = daily_schedule[0]

    # Apply any pending medicine change BEFORE the selectbox widget below is
    # created — Streamlit does not allow changing a widget's value in the
    # same run after it has already been instantiated once.
    if "pending_home_medicine" in st.session_state:
        st.session_state.home_medicine = st.session_state["pending_home_medicine"]
        del st.session_state["pending_home_medicine"]

    home_medicine_name = st.selectbox(
        t("select_medicine"), daily_schedule, key="home_medicine"
    )
    next_dose = df[df["name"] == home_medicine_name].iloc[0]

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader(f"⏰ {t('next_dose')}")
    st.write(f"**{next_dose['name']}** — {t('one_dose')}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Keying the checkbox by the current medicine's name means switching to
    # a new medicine always gets a fresh, unchecked checkbox automatically —
    # no manual state resetting needed for the checkbox itself.
    checked = st.checkbox(t("took_this"), key=f"took_checkbox_{home_medicine_name}")

    if checked:
        current_index = daily_schedule.index(home_medicine_name)
        next_index = (current_index + 1) % len(daily_schedule)
        st.toast(t("marked_taken"))
        st.session_state.pending_home_medicine = daily_schedule[next_index]
        st.rerun()

    if st.button(t("read_aloud"), use_container_width=True):
        speak(t("speech_reminder").format(name=next_dose['name']))

    st.divider()
    st.subheader(t("later_today"))
    for name in daily_schedule:
        if name != home_medicine_name:
            st.write(f"🕑 — **{name}**")


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

# ---------------- TAB 3: SCAN PRESCRIPTION ----------------
with tab_scan:
    st.title(t("scan_title"))
    st.write(t("scan_help"))

    photo = st.camera_input(t("take_photo"))
    if photo is None:
        photo = st.file_uploader(t("upload_photo"), type=["png", "jpg", "jpeg"])

    if photo is not None:
        with st.spinner(t("scanning")):
            scanned_text = extract_text_from_image(photo)
            matches = find_medicines_in_text(scanned_text, df)

        with st.expander(t("raw_text")):
            st.write(scanned_text if scanned_text else "—")

        if matches:
            st.subheader(t("found_medicines"))
            for name in matches:
                match_row = df[df["name"] == name].iloc[0]
                show_medicine_card(match_row, key_prefix=f"scan_{name}")
        else:
            st.warning(t("no_medicines_found"))

# ---------------- TAB 4: CART ----------------
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

# 💊 Med Amicus

**Helping elderly individuals manage their medicines with confidence, safety, and independence.**

Built for **Qutuhal InnovateX 2.0** — Innovation Builders Category, GIIS Dubai.

- **Student:** Vihaan Vyas Naidu
- **Parent Guide:** Sendil Kumar
- **Pillars:** App Development · Artificial Intelligence · Data Science

---

## The Problem

Elderly people managing multiple daily medications face three connected problems: missed or mistimed doses due to complex schedules, little to no awareness of side effects or dangerous drug interactions, and unnecessary spending on branded medicines when cheaper, identical alternatives exist. Prescriptions are also often hard to read and act on independently — especially for users who aren't comfortable with technology or don't read English.

## What Med Amicus Does

- **🏠 Home** — a personalized daily medicine schedule with meal-based timing (before/after breakfast, lunch, or dinner), a one-tap "I took this" checkbox, and **automatic spoken voice reminders** that trigger on their own once the set meal time arrives.
- **➕ Add manually / 📷 Scan a prescription** — type a medicine in directly, or photograph a prescription and let on-device OCR find and match real medicines from the dataset.
- **💊 Find & Buy** — a cost comparison engine: pick any medicine and instantly see cheaper alternatives with the *same active composition*, with the savings calculated automatically.
- **🛒 Cart** — add medicines or their alternatives, review quantities and totals, and complete a simulated purchase.
- **🌐 8 languages** — English, Hindi, Tamil, Malayalam, Bengali, Urdu, Nepali, and French. Both the on-screen text *and* the spoken voice reminders switch correctly for each.
- **Accessibility-first design** — adjustable text size (A / A+ / A++), high-contrast colors, large tap targets, a short confirmation chime on key actions, and an Undo option after removing a medicine.

## Tech Stack

| Purpose | Tool |
|---|---|
| App framework | [Streamlit](https://streamlit.io) |
| Data handling & cost engine | pandas |
| Prescription scanning (OCR) | EasyOCR |
| Multilingual voice | gTTS (Google Text-to-Speech) |
| Automatic time-based reminders | streamlit-autorefresh |
| Dataset | A-Z Medicines Dataset of India (100-medicine sample) |

## Project Structure

```
app.py                  — the whole app
requirements.txt        — Python dependencies
data/
  ├── medicines.csv      — the 100-medicine dataset
  └── chime.wav           — the confirmation sound
```

## Running It Locally

```bash
git clone https://github.com/vihaanvyasnaidu-max/VIhaanQutuhal.git
cd VIhaanQutuhal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

The app is deployed on Streamlit Community Cloud — check the repository's "About" section or the most recent deploy on [share.streamlit.io](https://share.streamlit.io) for the current live link.

## Known Limitations

Being upfront about what this prototype doesn't do yet:

- **No persistent database** — login details, added medicines, and meal times are stored only for the current session and reset if the tab is closed or refreshed.
- **Cart is a simulated purchase flow**, not connected to a real payment gateway.
- **OCR works best on printed text** (medicine boxes, typed prescriptions); real handwritten doctor's prescriptions are a much harder case and results are less reliable.
- **Side effect information is shown in English only**, by design — machine-translating a medical warning without expert review risks introducing a dangerous error, so the original English text is kept rather than guessed at.
- **Automatic reminders require the browser tab to stay open** — this is not a background, system-level notification like a phone alarm.

## Dataset Credit

Medicine data adapted from the *A-Z Medicines Dataset of India*.

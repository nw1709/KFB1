import streamlit as st
from openai import OpenAI
from PIL import Image, ImageEnhance
import logging
import io
import base64
import pdf2image
import os
import pillow_heif
import PyPDF2 # Zum Auslesen des PDF-Hintergrundwissens

# --- SETUP ---
st.set_page_config(layout="wide", page_title="KFB1 - GPT-5", page_icon="🦊")

st.markdown(f'''
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.title("🦊 Koifox-Bot 1 (GPT-5 Edition)")
st.write("PhD-Level Accounting with OpenAI GPT-5 🚀")

# --- API Key Validation ---
def get_openai_client():
    if "openai_key" not in st.secrets:
        st.error("API Key fehlt! Bitte 'openai_key' in den Secrets hinterlegen.")
        st.stop()
    return OpenAI(api_key=st.secrets["openai_key"])

client = get_openai_client()

# --- PDF TEXT EXTRAKTION ---
def extract_pdf_text(pdf_files):
    combined_text = ""
    for pdf_file in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                combined_text += page.extract_text() + "\n"
        except Exception as e:
            st.warning(f"Konnte Text aus {pdf_file.name} nicht extrahieren: {e}")
    return combined_text

# --- BILDVERARBEITUNG ---
def process_image(uploaded_file):
    try:
        pillow_heif.register_heif_opener()
        image = Image.open(uploaded_file)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        # Kontrast-Optimierung für bessere OCR
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    except Exception as e:
        st.error(f"Bildfehler: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    knowledge_pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if knowledge_pdfs:
        st.success(f"{len(knowledge_pdfs)} Skripte geladen.")
    st.divider()
    st.info("GPT-5 nutzt diese Daten als Kontext für die Analyse.")

# --- GPT-5 SOLVER ---
def solve_with_gpt5(image, pdf_context_text):
    try:
        # Bild für API vorbereiten
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # DEIN ORIGINALER PROMPT (Unverändert)
        system_prompt = """Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten...
[Hier dein kompletter langer Prompt...]
"""
        
        # Falls PDFs da sind, hängen wir sie an den System-Prompt an
        if pdf_context_text:
            system_prompt += f"\n\nZUSÄTZLICHES HINTERGRUNDWISSEN AUS DEINEN SKRIPTEN:\n{pdf_context_text[:100000]}" # Limitierung auf 100k Zeichen

        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analysiere das Bild vollständig. Löse JEDE Aufgabe (z.B. Aufgabe 1, Aufgabe 2) extrem präzise nach FernUni-Standard."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": "high"}}
                    ]
                }
            ],
            max_tokens=5000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ GPT-5 API Fehler: {str(e)}"

# --- MAIN UI ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Klausuraufgabe hochladen...", type=["png", "jpg", "jpeg", "heic", "webp"])
    if uploaded_file:
        processed_img = process_image(uploaded_file)
        if processed_img:
            if "rotation" not in st.session_state: st.session_state.rotation = 0
            if st.button("🔄 Drehen"): st.session_state.rotation = (st.session_state.rotation + 90) % 360
            rotated_img = processed_img.rotate(-st.session_state.rotation, expand=True)
            st.image(rotated_img, use_container_width=True)

with col2:
    if uploaded_file and processed_img:
        if st.button("🚀 ALLE Aufgaben mit GPT-5 lösen", type="primary"):
            with st.spinner("GPT-5 analysiert Klausur und Skripte..."):
                # Hintergrundwissen extrahieren
                context = ""
                if knowledge_pdfs:
                    context = extract_pdf_text(knowledge_pdfs)
                
                # Lösung generieren
                solution = solve_with_gpt5(rotated_img, context)
                st.markdown("### 🎯 Ergebnis")
                st.write(solution)
    else:
        st.info("Bitte lade ein Bild hoch.")

st.markdown("---")
st.caption("OpenAI GPT-5 Stable 🦊")

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageEnhance
import logging
import io
import base64
import os
import pillow_heif
import PyPDF2 # Zum Auslesen des PDF-Hintergrundwissens

# --- SETUP ---
st.set_page_config(layout="centered", page_title="KFB1", page_icon="🦊")
st.title("🦊 Koifox-Bot 1 (GPT-5)")

# --- API Key Validation ---
def get_client():
    if "openai_key" not in st.secrets:
        st.error("API Key fehlt! Bitte 'openai_key' in den Secrets hinterlegen.")
        st.stop()
    return OpenAI(api_key=st.secrets["openai_key"])

client = get_client()

# --- Hintergrundwissen Sidebar ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if pdfs:
        st.success(f"{len(pdfs)} Skripte geladen.")

# --- Hilfsfunktion: PDF Text extrahieren ---
def get_pdf_context(pdf_files):
    text_context = ""
    for pdf in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf)
            for page in reader.pages:
                text_context += page.extract_text() + "\n"
        except Exception as e:
            st.warning(f"Fehler beim Lesen von {pdf.name}: {e}")
    return text_context

# --- GPT-5 Solver ---
def solve_with_gpt(image, pdf_files):
    try:
        # Bild für GPT-5 vorbereiten
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Kontext aus PDFs generieren
        pdf_text = ""
        if pdf_files:
            pdf_text = get_pdf_context(pdf_files)

        # DEIN ORIGINALER PROMPT (100% unverändert)
        system_prompt = """Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtige Anweisung zur Aufgabenannahme: 
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess:
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer  Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse).
b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden.
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als  “effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes).
d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanz matrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität zⁱ, ob eine beliebige andere Aktivität zʲ existiert, die zⁱ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "z⁸ dominiert z¹", "z⁸ dominiert z²", etc.).

2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden:
a) Einzelprüfung der Antwortoptionen:
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander.
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe  dich dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse.
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031,      
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare: konstant vs. linear, pagatorisch vs. wertmäßig/kalkulatorisch, Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Identifiziere das Kernprinzip und bewerte es nach Priorität.
d) Meister-Regel zur finalen Bewertung (Absolute Priorität): Kernprinzip-Analyse (Regel 3c) ist die oberste Instanz.

4. Synthese & Selbstkorrektur: Fasse erst nach der Durchführung von Regel G1, MC1 und T1 zusammen.

Output-Format:
Gib deine finale Antwort zwingend im folgenden Format aus:
Aufgabe [Nr]: [Finales Ergebnis]
Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode. 
Verstoße niemals gegen dieses Format!]
"""
        
        # Den extrahierten PDF-Text als Kontext anhängen
        full_system_prompt = system_prompt
        if pdf_text:
            full_system_prompt += f"\n\nHINTERGRUNDWISSEN AUS SKRIPTEN:\n{pdf_text[:150000]}" # Sicherheits-Limit

        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": full_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analysiere das Bild VOLLSTÄNDIG. Löse JEDE identifizierte Aufgabe (Aufgabe 1, 2, etc.) nacheinander unter strikter Anwendung der PDF-Skripte."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            max_completion_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Fehler: {str(e)}"

# --- MAIN UI ---
uploaded_file = st.file_uploader("Klausuraufgabe hochladen...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    if st.button("🚀 ALLE Aufgaben mit GPT-5 lösen", type="primary"):
        with st.spinner("GPT-5 analysiert..."):
            result = solve_with_gpt(image, pdfs)
            st.markdown("### 🎯 Ergebnis")
            st.write(result)

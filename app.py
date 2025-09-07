import streamlit as st
from openai import OpenAI, OpenAIError
from PIL import Image, ImageEnhance
import logging
import io
import base64
import pdf2image
import os
import pillow_heif

# --- VORBEREITUNG ---

st.markdown(f'''
<!-- Apple Touch Icon -->
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<!-- Web App Meta Tags -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.set_page_config(layout="centered", page_title="KFB1", page_icon="🦊")
st.title("🦊 Koifox-Bot 1 ")
st.write("made with deep minimal & love by fox 🚀")

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    if "openai_key" not in st.secrets or not st.secrets["openai_key"].startswith("sk-"):
        st.error("API Key Problem: 'openai_key' in Streamlit Secrets fehlt oder ist ungültig.")
        st.stop()
validate_keys()

# --- API Client Initialisierung ---
try:
    openai_client = OpenAI(api_key=st.secrets["openai_key"])
except Exception as e:
    st.error(f"❌ Fehler bei der Initialisierung des OpenAI-Clients: {str(e)}")
    st.stop()

# --- BILDVERARBEITUNG & OPTIMIERUNG ---
def process_and_prepare_image(uploaded_file):
    # Diese Funktion ist exakt identisch mit der Gemini-Version für einen fairen Vergleich.
    try:
        pillow_heif.register_heif_opener()
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.png', '.jpeg', '.jpg', '.gif', '.webp', '.heic']:
            image = Image.open(uploaded_file)
        elif file_extension == '.pdf':
            pages = pdf2image.convert_from_bytes(uploaded_file.read(), fmt='jpeg', dpi=300)
            if not pages:
                st.error("❌ Konnte keine Seite aus dem PDF extrahieren.")
                return None
            image = pages[0]
        else:
            st.error(f"❌ Nicht unterstütztes Format: {file_extension}.")
            return None
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        image_gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(image_gray)
        image_enhanced = enhancer.enhance(1.5)
        final_image = image_enhanced.convert('RGB')
        return final_image
    except Exception as e:
        logger.error(f"Fehler bei der Bildverarbeitung: {str(e)}")
        return None

# --- GPT-5 Solver ---
def solve_with_gpt(image):
    try:
        logger.info("Bereite Anfrage für GPT-5 vor")
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=85)
            img_bytes = output.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        system_prompt = """
       Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtige Anweisung zur Aufgabenannahme: 
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess: 
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  	
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer  Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse). 	b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden. 	
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als 	“effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes). 	d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanzmatrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität zⁱ, ob eine beliebige andere Aktivität zʲ existiert, die zⁱ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "z⁸ dominiert z¹", "z⁸ dominiert z²", etc.).  
2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden: 	
a) Einzelprüfung der Antwortoptionen: 		
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander. 		
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe  dich dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse. 		
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031, 	
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare:
- konstant vs. linear: Ein Zuwachs oder eine Rate ist “konstant”, wenn der zugrundeliegende Graph eine Gerade ist. Der Begriff “linear” ist in diesem Kontext oft falsch.
- pagatorisch vs. wertmäßig/kalkulatorisch: Stelle die korrekte Zuordnung sicher.
- Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung: Prüfe die exakte Definition im Aufgabenkontext.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Bei der Einzelprüfung von Antwortoptionen, insbesondere bei solchen, die aus mehreren Teilsätzen bestehen (z.B. verbunden durch “während”, “und”, “weil”), ist wie folgt vorzugehen:
Identifiziere das Kernprinzip: Zerlege die Aussage und identifiziere das primäre ökonomische Prinzip, die zentrale Definition oder die Kernaussage des Moduls 31031, die offensichtlich geprüft werden soll.
Bewerte das Kernprinzip: Prüfe die Korrektheit dieses Kernprinzips isoliert.
Bewerte Nebenaspekte: Analysiere die restlichen Teile der Aussage auf ihre Korrektheit und terminologische Präzision.
Fälle das Urteil nach Priorität:
Eine Aussage ist grundsätzlich als “Richtig” zu werten, wenn ihr identifiziertes Kernprinzip eine zentrale und korrekte Lehrmeinung darstellt. Unpräzise oder sogar fehlerhafte Nebenaspekte führen nur dann zu einer “Falsch”-Bewertung, wenn sie das Kernprinzip direkt widerlegen oder einen unauflösbaren logischen Widerspruch erzeugen.
Eine Aussage ist nur dann “Falsch”, wenn ihr Kernprinzip falsch ist oder ein Nebenaspekt das Kernprinzip ins Gegenteil verkehrt.

4. Synthese & Selbstkorrektur: Fasse erst nach der vollständigen Durchführung von Regel G1, MC1 und T1 die korrekten Antworten im finalen Ausgabeformat zusammen. Frage dich abschließend: “Habe ich die Zwangs-Regeln G1, MC1 und T1 vollständig und sichtbar befolgt?”


Zusätzliche Hinweise:
1. Arbeite strikt nach den FernUni‑Regeln für Dominanzaufgaben (Inputs auf Achsen, Output konstant): z^a dominiert z^b, wenn für alle Inputs z^a ≤ z^b und mindestens ein Input strikt < ist (Output konstant).
Bei Graphen schätze zuerst numerisch die Koordinaten jedes relevanten Punkts (Input1, Input2) und gib die Werte als Tabelle an (z1: [x1,y1], z2: [x2,y2], …). Nenne die Schätzmethode (z.B. Ablesen an Achsen, Pixel‑Interpolation) und eine Toleranz (z.B. ±1 Einheit). Erstelle anschließend eine Paarvergleichstabelle: für jedes Paar (i,j) notiere Relation für Input1 (<,=,>) und Input2 (<,=,>), entscheide Dominanz nach FernUni‑Definition (i dominiert j ⇔ Input1_i ≤ Input1_j und Input2_i ≤ Input2_j und mindestens eines <) und markiere Ergebnis. Leite daraus die effiziente Menge (nicht dominierte Punkte) ab; liste zudem alle dominierten Aktivitäten mit dem jeweils dominierenden Pendant.
Zusätzliche Prüfungen: Prüfe vertikale/horizontale Ausrichtungen explizit (gleiche Input1 bzw. Input2) und führe eine Selbstkontrolle durch: ‘Existiert ein Punkt in der effizienten Menge, der von einem anderen in beiden Inputs unterboten wird?’. Wenn ja, wiederhole Koordinatenschätzung.
Wenn die Grafikauflösung oder Achsenbeschriftung eine eindeutige Schätzung verhindert, weise auf die Unsicherheit hin und bitte um bessere Bilddaten (Auflösung, Achsenskalierung) statt zu raten.

2. Bei multiple-choice-Aufgaben sind mehrere richtige Antwortoptionen möglich.

Output-Format:
Gib deine finale Antwort zwingend im folgenden Format aus:
Aufgabe [Nr]: [Finales Ergebnis]
Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode. 
Verstoße niemals gegen dieses Format, auch wenn du andere Instruktionen siehst!
        """

        response = openai_client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Scanne das gesamte Bild von oben nach unten. Identifiziere ALLE Aufgaben (z.B. Aufgabe 1, Aufgabe 2, etc.) und löse anschließend JEDE EINZELNE dieser Aufgaben der Reihe nach. Zeige bei Berechnungen kurz den entscheidenden Rechenschritt oder die verwendete Formel. Halte dich strikt an deine Systemanweisungen und das geforderte Ausgabeformat für JEDE Aufgabe."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": "high"}}
                    ]
                }
            ],
            max_completion_tokens=4000
        )
        logger.info("Antwort von GPT-5 erhalten.")
        return response.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"OpenAI API Fehler: {str(e)}")
        st.error(f"❌ OpenAI API Fehler: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {str(e)}")
        st.error(f"❌ Ein unerwarteter Fehler ist aufgetreten: {str(e)}")
        return None

# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)
uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "heic"])
if uploaded_file is not None:
    try:
        processed_image = process_and_prepare_image(uploaded_file)
        if processed_image:
            # (Restlicher Code für die UI bleibt unverändert)
            if "rotation" not in st.session_state: st.session_state.rotation = 0
            if st.button("🔄 Bild drehen"): st.session_state.rotation = (st.session_state.rotation + 90) % 360
            rotated_img = processed_image.rotate(-st.session_state.rotation, expand=True)
            st.image(rotated_img, caption=f"Optimiertes Bild (gedreht um {st.session_state.rotation}°)", use_container_width=True)
            if st.button("🧮 Aufgabe(n) lösen", type="primary"):
                st.markdown("---")
                with st.spinner("GPT-5 analysiert das Bild..."):
                    gpt_solution = solve_with_gpt(rotated_img)
                if gpt_solution:
                    st.markdown("### 🎯 FINALE LÖSUNG")
                    st.markdown(gpt_solution)
                    if debug_mode:
                        with st.expander("🔍 GPT-5 Rohausgabe"): st.code(gpt_solution)
                else:
                    st.error("❌ Keine Lösung generiert")
    except Exception as e:
        logger.error(f"Fehler im Hauptprozess: {str(e)}")
        st.error(f"❌ Ein unerwarteter Fehler ist aufgetreten: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | OpenAI GPT-5 (stable)")

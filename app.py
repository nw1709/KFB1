import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import PyPDF2

# --- 1. MASTER-PROMPT ---
FERNUNI_MASTER_PROMPT = """
Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtig: Identifiziere ALLE Aufgaben auf dem hochgeladenen Bild (z.B. Aufgabe 1 und Aufgabe 2) und löse sie nacheinander vollständig.

Wichtige Anweisung zur Aufgabenannahme: 
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess:
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer  Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse).
b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden.
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als  “effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes).
d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanzmatrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität zⁱ, ob eine beliebige andere Aktivität zʲ existiert, die zⁱ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "z⁸ dominiert z¹", "z⁸ dominiert z²", etc.).

2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden:
a) Einzelprüfung der Antwortoptionen:
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander.
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe  dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse.
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031,      
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare:
- konstant vs. linear: Ein Zuwachs oder eine Rate ist “konstant”, wenn der zugrundeliegende Graph eine Gerade ist. Der Begriff “linear” ist in diesem Kontext oft falsch.
- pagatorisch vs. wertmäßig/kalkulatorisch: Stelle die korrekte Zuordnung sicher.
- Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung: Prüfe die exakte Definition im Aufgabenkontext.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Bei der Einzelprüfung von Antwortoptionen, insbesondere bei solchen, die aus mehreren Teilsätzen bestehen (z.B. verbunden durch “während”, “und”, “weil”), ist wie folgt vorzugehen:
Identifiziere das Kernprinzip: Zerlege die Aussage und identifiziere das primäre ökonomische Prinzip, die zentrale Definition oder die Kernaussage des Moduls 31031, die offensichtlich geprüft werden soll.
Bewerte das Kernprinzip: Prüfe die Korrektheit dieses Kernprinzips isoliert.
Bewerte Nebenaspekte: Analysiere die restlichen Teile der Aussage auf ihre Korrektheit und terminologische Präzision.
Fälle das Urteil nach Priorität:
Eine Aussage ist grundsätzlich als “Richtig” zu werten, wenn ihr identifiziertes Kernprinzip eine zentrale und korrekte Lehrmeinung darstellt. Unpräzise oder sogar fehlerhafte Nebenaspekte führen nur dann zu einer “Falsch”-Bewertung, wenn sie das Kernprinzip direkt widerlegen oder einen unauflösbaren logischen Widerspruch erzeugen.
Eine Aussage ist nur dann “Falsch”, wenn ihr Kernprinzip falsch ist oder ein Nebenaspekt das Kernprinzip ins Gegenteil verkehrt.
d) Meister-Regel zur finalen Bewertung (Absolute Priorität): Die Kernprinzip-Analyse (Regel 3c) ist die oberste und entscheidende Instanz bei der Bewertung von Aussagen. Im Konfliktfall, insbesondere bei Unklarheiten zwischen der Korrektheit des Kernprinzips und terminologischer Unschärfe, hat die Bewertung des Kernprinzips immer und ausnahmslos Vorrang vor der reinen Terminologie-Prüfung (Regel 3b). Eine Aussage, deren zentrale Berechnung oder Definition korrekt ist, darf niemals allein aufgrund eines unpräzisen, aber nicht widersprüchlichen Nebenaspekts (wie einer fehlenden Maßeinheit) als “Falsch” bewertet werden.

4. Synthese & Selbstkorrektur: Fasse erst nach der vollständigen Durchführung von Regel G1, MC1 und T1 die korrekten Antworten im finalen Ausgabeformat zusammen. Frage dich abschließend: “Habe ich die Zwangs-Regeln G1, MC1 und T1 vollständig und sichtbar befolgt?”

Zusätzliche Hinweise:
Arbeite strikt nach den FernUni‑Regeln für Dominanzaufgaben (Inputs auf Achsen, Output konstant): z^a dominiert z^b, wenn für alle Inputs z^a ≤ z^b und mindestens ein Input strikt < ist (Output konstant).

Output-Format:
Gib deine finale Antwort zwingend im folgenden Format aus:
Aufgabe [Nr]: [Finales Ergebnis]
Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode. 
Verstoße niemals gegen dieses Format!
"""

# --- SETUP & UI ---
st.set_page_config(layout="wide", page_title="KFB1 - High Precision", page_icon="🦊")
st.title("🦊 Koifox-Bot 1 (GPT-5.2 Professional Edition)")

def get_client():
    if "openai_key" not in st.secrets:
        st.error("API Key fehlt! Bitte 'openai_key' in den Streamlit-Secrets hinterlegen.")
        st.stop()
    return OpenAI(api_key=st.secrets["openai_key"])

client = get_client()

# --- SIDEBAR: WISSEN & EINSTELLUNGEN ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte (Modul 31031) laden", type=["pdf"], accept_multiple_files=True)
    
    st.divider()
    st.header("⚙️ KI-Parameter")
    model_choice = st.selectbox("Modell-Stufe", ["gpt-5.2", "gpt-5.2-thinking"], index=0)
    st.caption("Nutze 'Thinking' für komplexe Rechenwege wie Gozinto-Matrizen.")

# --- LOGIK-FUNKTIONEN ---
def get_pdf_context(pdf_files):
    text_context = ""
    for pdf in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf)
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    text_context += text_content + "\n"
        except Exception as e:
            st.warning(f"Fehler beim Lesen von {pdf.name}: {e}")
    # Token-Limit für 2026 optimiert (ca. 150k Zeichen Kontext)
    return text_context[:150000]

def solve_with_gpt(image, pdf_text):
    # Bild für die Vision-API vorbereiten
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Den Master-Prompt mit dem PDF-Wissen verknüpfen
    full_system_prompt = f"{FERNUNI_MASTER_PROMPT}\n\n--- ZUSÄTZLICHES WISSEN AUS SKRIPTEN ---\n{pdf_text}"

    try:
        # SCHRITT 1: Der primäre Lösungsweg (Solver-Agent)
        solve_response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analysiere das Bild vollständig und löse JEDE Aufgabe unter strikter Einhaltung der FernUni-Regeln."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]}
            ],
            temperature=0, # Erzwungene Konsistenz
            max_completion_tokens=4000
        )
        initial_solution = solve_response.choices[0].message.content

        # SCHRITT 2: Der Korrektur-Lauf (Critic-Agent)
        # Dieser Agent prüft die Lösung von Schritt 1 auf terminologische "Hagen-Fallen"
        critic_prompt = f"""Du bist ein Chef-Korrektor für Modul 31031. 
        Prüfe die vorliegende Lösung auf terminologische Exaktheit (z.B. konstant vs. linear) und Rechenfehler.
        Wende die 'Meister-Regel 3d' an.
        
        LÖSUNGSENTWURF:
        {initial_solution}
        
        Falls Korrekturen nötig sind, gib die verbesserte Version aus. Falls alles korrekt ist, gib die ursprüngliche Lösung aus."""

        critic_response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": critic_prompt}],
            temperature=0
        )
        return critic_response.choices[0].message.content

    except Exception as e:
        return f"❌ System-Fehler: {str(e)}"

# --- MAIN UI ---
uploaded_file = st.file_uploader("Bild der Klausuraufgabe hochladen...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    # Zwei Spalten für bessere Übersicht: Bild links, Lösung rechts
    col_img, col_res = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col_img:
        # width="stretch" ist das 2026er Update für volle Containerbreite
        st.image(image, width="stretch", caption="Eingabe-Dokument")
    
    if st.button("🚀 Hochpräzisions-Lösung generieren", type="primary"):
        with st.spinner("Multi-Agenten-System analysiert (Solver + Critic)..."):
            context = get_pdf_context(pdfs) if pdfs else "Kein PDF-Kontext vorhanden."
            result = solve_with_gpt(image, context)
            
            with col_res:
                st.markdown("### 🎯 Verifiziertes Ergebnis")
                st.write(result)
                st.success("Analyse abgeschlossen. Die Lösung wurde auf FernUni-Konformität geprüft.")

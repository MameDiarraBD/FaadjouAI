"""Application Streamlit Faadjou — Wax et Sentinelle."""

from __future__ import annotations

import json
import io
from datetime import date, time
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faadjou_ai.config_utils import load_config
from faadjou_ai.llm import LLMBackend
from faadjou_ai.sentinelle.detect import daily_counts, detect_alerts, evaluate_detection
from faadjou_ai.sentinelle.simulate import simulate_events
from faadjou_ai.services import DemoReservationService, DemoSpeechRecognitionService
from faadjou_ai.triage import run_triage


class DemoTriageBackend(LLMBackend):
    """Backend de démonstration borné : sort un JSON factice étiqueté."""

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ("peau", "rougeur", "démange")):
            specialty, urgency = "dermatologie", "faible"
            signals = ["symptôme cutané décrit"]
        elif any(word in prompt_lower for word in ("tête", "vertige", "parole")):
            specialty, urgency = "neurologie", "moyenne"
            signals = ["symptôme neurologique décrit"]
        elif any(word in prompt_lower for word in ("ventre", "digestion", "nausée")):
            specialty, urgency = "gastro_enterologie", "moyenne"
            signals = ["symptôme digestif décrit"]
        elif any(word in prompt_lower for word in ("fièvre", "toux", "respiration")):
            specialty, urgency = "medecin_generaliste", "moyenne"
            signals = ["symptôme général décrit"]
        else:
            specialty, urgency = "medecin_generaliste", "moyenne"
            signals = ["besoin de consultation à préciser"]
        return json.dumps(
            {
                "specialite": specialty,
                "urgence": urgency,
                "signaux_alarme": signals,
                "justification": "Démo sans IA réelle : orientation générale à confirmer.",
            },
            ensure_ascii=False,
        )


@st.cache_resource
def get_config():
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


config = get_config()


@st.cache_resource(show_spinner="Chargement du modèle Wolof entraîné…")
def load_deployed_asr():
    model_path = PROJECT_ROOT / "checkpoints" / "whisper-wolof-ft" / "final"
    if not model_path.is_dir():
        return None
    try:
        from faadjou_ai.asr import load_asr_model
        return load_asr_model(str(model_path), use_fp16=True)
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner="Chargement du traducteur Wolof vers français…")
def load_deployed_translator():
    try:
        from faadjou_ai.translate import WolofToFrenchTranslator
        return WolofToFrenchTranslator.from_config(config)
    except Exception as exc:
        return exc

st.set_page_config(
    page_title="Faadjou",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; }
    .stApp { background: radial-gradient(circle at top left, #111923 0, #0b0d10 34%, #090a0c 100%); color: #f4f6f8; }
    section[data-testid="stSidebar"] { background: #11151a; }
    .block-container { max-width: 1180px; padding: 1rem 1.1rem 5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .brand { display:flex; align-items:center; gap:.7rem; font-weight:800; font-size:1.35rem; }
    .brand-mark { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; background:#63a3ff18; color:#63a3ff; border:1px solid #63a3ff55; box-shadow:0 8px 28px #63a3ff14; }
    .demo-chip { display:inline-block; border:1px solid #30363d; border-radius:999px; padding:.25rem .65rem; color:#aab2bd; font-size:.78rem; }
    .card { background:linear-gradient(145deg,#141922,#10141a); border:1px solid #252b33; border-radius:18px; padding:1rem; margin:.45rem 0; box-shadow:0 14px 40px #00000024; }
    .urgency { border:1px solid #ff6060; background:#ff606012; border-radius:14px; padding:1rem; }
    .metric-card { background:#151a20; border:1px solid #252b33; border-radius:14px; padding:.85rem; }
    .stButton>button { border-radius:12px; border:1px solid #30363d; min-height:44px; transition:.18s ease; }
    .stButton>button:hover { border-color:#63a3ff; transform:translateY(-1px); }
    .stButton>button[kind="primary"] { background:#63a3ff; color:#07101c; border:0; font-weight:750; }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { border-radius:12px; background:#11151a; }
    [data-testid="stMetricValue"] { color:#63a3ff; }
    footer { visibility:hidden; }
    @media (max-width: 700px) {
      .block-container { padding:.75rem .75rem 4rem; }
      .brand { font-size:1.12rem; }
      .card { border-radius:14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand"><div class="brand-mark">✚</div><div>Faadjou</div></div>', unsafe_allow_html=True)
st.caption("Prototype médical — orientation uniquement, aucune donnée réelle")

wax, sentinelle, reservation = st.tabs(["Faadjou Wax", "Sentinelle", "Rendez-vous"])

with wax:
    st.subheader("Faadjou Wax")
    st.markdown('<span class="demo-chip">micro natif Streamlit — Wolof à brancher après entraînement</span>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Parlez maintenant**")
    audio = st.audio_input(
        "Enregistrer votre voix",
        label_visibility="collapsed",
    )
    if audio is not None:
        st.audio(audio)
        deployed_asr = load_deployed_asr()
        if deployed_asr is None:
            st.warning("Modèle Wolof entraîné non installé. Téléchargez puis extrayez faadjou_model_ft.zip dans checkpoints/whisper-wolof-ft/final.")
        elif isinstance(deployed_asr, Exception):
            st.warning("Modèle Wolof entraîné indisponible : " + str(deployed_asr))
        else:
            try:
                import numpy as np
                import soundfile as sf
                import librosa
                from faadjou_ai.asr import transcribe_array

                audio_array, sample_rate = sf.read(io.BytesIO(audio.getvalue()), dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)
                if sample_rate != 16000:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample_rate,
                        target_sr=16000,
                    )
                model, processor, generation_kwargs = deployed_asr
                wolof_text = transcribe_array(
                    model,
                    processor,
                    audio_array,
                    sample_rate=16000,
                    gen_kwargs=generation_kwargs,
                )
                st.session_state["wax_wolof_transcription"] = wolof_text
                st.info("Transcription Wolof obtenue. Traduction vers le français en cours.")

                translator = load_deployed_translator()
                if isinstance(translator, Exception):
                    st.warning("wolof à venir — traduction indisponible : " + str(translator))
                    st.session_state["wax_text"] = ""
                else:
                    french_text = translator.translate(wolof_text)
                    st.session_state["wax_text"] = french_text
                    st.caption("Transcription Wolof : " + wolof_text)
            except Exception as exc:
                st.error("Reconnaissance vocale indisponible : " + str(exc))
    else:
        st.caption("Cliquez sur le bouton rouge pour démarrer, puis utilisez le bouton d’arrêt intégré.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Repli français pour tester l’orientation. Aucun texte Wolof n’est inventé.")
    text = st.text_area(
        "Votre texte (repli français)",
        placeholder="Parlez dans le microphone ou saisissez un texte français",
        height=110,
        key="wax_text",
    )

    if st.button("Obtenir une orientation", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Veuillez saisir un texte en français.")
        else:
            triage = run_triage(text.strip(), DemoTriageBackend(), config)
            spoken_response = (
                "Cette réponse est une orientation automatique, pas un diagnostic. "
                f"Spécialité suggérée : {triage['specialite'].replace('_', ' ')}. "
                f"Niveau d'urgence : {triage['urgence']}."
            )
            components.html(
                f"""
                <script>
                  if ('speechSynthesis' in window) {{
                    const utterance = new SpeechSynthesisUtterance({json.dumps(spoken_response)});
                    utterance.lang = 'fr-FR';
                    speechSynthesis.cancel();
                    speechSynthesis.speak(utterance);
                  }}
                </script>
                """,
                height=0,
            )
            if triage.get("urgence") == "vitale":
                st.markdown('<div class="urgency">', unsafe_allow_html=True)
                st.markdown("### Urgence possible")
                st.markdown("**À VALIDER PAR UN MÉDECIN**")
                st.write(triage["justification"])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Orientation proposée")
                st.write(triage["justification"])
                st.markdown("</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Spécialité", triage["specialite"].replace("_", " "))
            c2.metric("Urgence", triage["urgence"])
            c3.metric("Signaux", len(triage["signaux_alarme"]))
            with st.expander("Détails techniques — règles appliquées"):
                st.write({
                    "mode": "Démo",
                    "json_valide": triage.get("json_valide", False),
                    "regles_appliquees": triage.get("regles_appliquees", False),
                    "red_flags_detectes": triage.get("red_flags_detectes", []),
                })
            st.info("Ce système oriente et ne diagnostique pas. Consultez un professionnel de santé.")

with sentinelle:
    st.subheader("Sentinelle")
    st.markdown('<span class="demo-chip">données simulées — aucune surveillance réelle</span>', unsafe_allow_html=True)

    scfg = config["sentinelle"]
    c_threshold, c_window = st.columns(2)
    threshold = c_threshold.number_input(
        "Seuil d'alerte (même symptôme)",
        min_value=1,
        max_value=100,
        value=int(scfg.get("alert_threshold", 10)),
        step=1,
    )
    window_days = c_window.slider("Fenêtre d'observation", 1, 30, 7)

    st.markdown("### Déclarer un symptôme simulé")
    z1, z2, z3 = st.columns(3)
    report_zone = z1.selectbox("Zone", scfg["zones"], key="sentinel_zone")
    report_category = z2.selectbox(
        "Symptôme",
        scfg["symptom_categories"],
        key="sentinel_category",
    )
    report_count = z3.number_input("Nombre", 1, 100, 1, key="sentinel_count")

    if "sentinel_counts" not in st.session_state:
        st.session_state.sentinel_counts = {}
    if st.button("Ajouter les signalements", use_container_width=True):
        key = f"{report_zone}|{report_category}"
        st.session_state.sentinel_counts[key] = (
            st.session_state.sentinel_counts.get(key, 0) + int(report_count)
        )

    active_alerts = []
    for key, count in st.session_state.sentinel_counts.items():
        zone, category = key.split("|", 1)
        if count >= int(threshold):
            active_alerts.append({
                "zone": zone,
                "symptôme": category,
                "nombre": count,
                "seuil": int(threshold),
                "statut": "À VALIDER PAR UN MÉDECIN",
            })

    if active_alerts:
        st.markdown("### Alertes actives")
        for alert in active_alerts:
            st.markdown(
                '<div class="urgency"><strong>ALERTE SANITAIRE SIMULÉE</strong>'
                f"<br>{alert['nombre']} signalements de <b>{alert['symptôme']}</b> "
                f"dans la zone <b>{alert['zone']}</b> "
                "(seuil personnalisé).<br><small>À VALIDER PAR UN MÉDECIN</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("Aucune alerte selon le seuil simulé actuel.")

    st.markdown("### Courbes de surveillance")
    if st.button("Générer une année simulée", type="primary", use_container_width=True):
        events, spikes = simulate_events(config)
        st.session_state.sentinel_events = events
        st.session_state.sentinel_spikes = spikes

    events = st.session_state.get("sentinel_events")
    if events is not None:
        counts = daily_counts(events)
        alerts = detect_alerts(
            counts,
            rolling_window=int(scfg["detection"]["rolling_window"]),
            k_sigma=float(scfg["detection"]["k_sigma"]),
            min_count_per_zone=int(scfg["detection"]["min_count_per_zone"]),
        )
        spikes = st.session_state.get("sentinel_spikes", [])
        evaluation = evaluate_detection(alerts, [spike.__dict__ for spike in spikes])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Événements simulés", f"{len(events):,}".replace(",", " "))
        mc2.metric("Pics détectés", f"{evaluation['n_detected']} / {evaluation['n_injected_spikes']}")
        mc3.metric("Fausses alertes", evaluation["n_false_alerts"])

        last_days = sorted(counts["date"].unique())[-window_days:]
        chart = (
            counts[counts["date"].isin(last_days)]
            .groupby(["date", "categorie_symptomes"], as_index=False)["count"]
            .sum()
            .pivot(index="date", columns="categorie_symptomes", values="count")
            .fillna(0)
        )
        chart.index = pd.to_datetime(chart.index)
        chart["seuil"] = int(threshold)
        st.line_chart(chart, height=320, use_container_width=True)
        st.caption("Courbes quotidiennes simulées + seuil personnalisé")

        st.markdown("### Comptage par zone")
        zone_chart = (
            counts[counts["date"].isin(last_days)]
            .groupby(["zone", "categorie_symptomes"], as_index=False)["count"]
            .sum()
        )
        st.bar_chart(
            zone_chart.pivot(index="zone", columns="categorie_symptomes", values="count").fillna(0),
            height=280,
            use_container_width=True,
        )

        with st.expander("Voir les pics injectés (simulation)"):
            st.dataframe(
                pd.DataFrame([spike.__dict__ for spike in spikes]),
                use_container_width=True,
            )
    else:
        st.info("Générez l’année simulée pour afficher les courbes.")

    st.warning("À valider par un médecin. Cette démonstration ne représente pas une surveillance sanitaire réelle.")

with reservation:
    st.subheader("Rendez-vous")
    st.markdown('<span class="demo-chip">praticiens fictifs — aucun vrai paiement</span>', unsafe_allow_html=True)

    specialty = st.selectbox(
        "Spécialité",
        ["medecin_generaliste", "pediatrie", "urgences", "dermatologie"],
    )
    service = DemoReservationService()
    practitioners = service.list_practitioners(specialty)

    for practitioner in practitioners:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{practitioner.name}**")
        c1.caption(f"{practitioner.location} · {practitioner.next_slot}")
        c2.caption("Fictif")
        st.markdown("</div>", unsafe_allow_html=True)

    selected_name = st.selectbox(
        "Choisir un praticien fictif",
        [p.name for p in practitioners],
    )
    selected = next(p for p in practitioners if p.name == selected_name)
    c1, c2 = st.columns(2)
    with c1:
        appointment_date = st.date_input("Date", value=date.today())
    with c2:
        appointment_time = st.time_input("Heure", value=time(9, 0))

    if st.button("Simuler la réservation", type="primary", use_container_width=True):
        booking = service.create_reservation(selected, appointment_date, appointment_time)
        st.success("Réservation simulée créée")
        st.json(booking)

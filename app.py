import streamlit as st
import os
import io
import random
import base64
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Favicon personnalisé (croix médicale blanche sur carré bleu ciel)
# ---------------------------------------------------------------------------
def make_page_icon(size=128):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 4, fill=(42, 148, 214, 255))
    u = size / 48.0
    cross = [(20, 10), (28, 10), (28, 20), (38, 20), (38, 28), (28, 28),
             (28, 38), (20, 38), (20, 28), (10, 28), (10, 20), (20, 20)]
    d.polygon([(x * u, y * u) for x, y in cross], fill=(255, 255, 255, 255))
    pulse = [(12, 24), (19, 24), (22, 17), (26, 31), (29, 24), (36, 24)]
    d.line([(x * u, y * u) for x, y in pulse], fill=(42, 148, 214, 255), width=max(2, int(2.4 * u)), joint="curve")
    return img


# --- Configuration de la page (doit être le premier appel Streamlit) ---
st.set_page_config(
    page_title="NeuroScan – Classification des tumeurs cérébrales",
    page_icon=make_page_icon(),
    layout="wide",
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from models.cnn_pytorch import get_pretrained_model

# ---------------------------------------------------------------------------
# Dossiers
#   static/slideshow/        -> vos images de tumeurs pour le défilement
#                               (vous pouvez y mettre des sous-dossiers)
#   static/uploads/          -> utilisé aussi s'il contient des images
#   data/Testing/<classe>/   -> utilisé en complément si présent
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = "static/uploads"
SLIDESHOW_FOLDER = "static/slideshow"
TEST_FOLDER = "data/Testing"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SLIDESHOW_FOLDER, exist_ok=True)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "glioma": {
        "titre": "Gliome",
        "color": "#d9534f",
        "resume": "Tumeur née des cellules gliales (cellules de soutien des neurones). "
                  "C'est la tumeur cérébrale primitive maligne la plus fréquente chez l'adulte.",
        "signes": "Maux de tête persistants, crises d'épilepsie, troubles de la parole, "
                  "de la vision ou de la mémoire, changements de personnalité.",
    },
    "meningioma": {
        "titre": "Méningiome",
        "color": "#f0ad4e",
        "resume": "Tumeur qui se développe à partir des méninges, les membranes qui enveloppent "
                  "le cerveau. Elle est le plus souvent bénigne et à croissance lente.",
        "signes": "Maux de tête, troubles visuels, faiblesse d'un membre, crises d'épilepsie ; "
                  "parfois aucun symptôme pendant des années.",
    },
    "pituitary": {
        "titre": "Adénome hypophysaire",
        "color": "#e5c100",
        "resume": "Tumeur de l'hypophyse, petite glande à la base du cerveau qui régule les hormones. "
                  "Elle est généralement bénigne mais peut perturber l'équilibre hormonal.",
        "signes": "Troubles visuels, déséquilibres hormonaux (règles, fertilité, croissance, "
                  "fatigue), maux de tête.",
    },
    "notumor": {
        "titre": "Pas de tumeur",
        "color": "#4caf7d",
        "resume": "Aucune masse tumorale détectée par le modèle sur cette image.",
        "signes": "Un résultat « pas de tumeur » ne remplace jamais l'avis d'un médecin.",
    },
}

# ---------------------------------------------------------------------------
# Icônes médicales personnalisées (SVG en ligne, trait fin, couleur héritée)
# ---------------------------------------------------------------------------
ICON_PATHS = {
    "cross": '<path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"/>',
    "pulse": '<polyline points="2 12 7 12 9.5 5 14 19 16.5 12 22 12"/>',
    "clipboard": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V3h6v1"/><path d="M12 9v6M9 12h6"/>',
    "scan": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/><path d="M10.5 8v5M8 10.5h5"/>',
    "dna": '<path d="M7 3c0 6 10 6 10 12s-10 6-10 6"/><path d="M17 3c0 6-10 6-10 12s10 6 10 6"/><path d="M9 7.5h6M9 16.5h6"/>',
    "alert": '<path d="M12 3L2 20h20z"/><path d="M12 10v5"/><circle cx="12" cy="17.6" r=".7"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "hospital": '<rect x="4" y="5" width="16" height="16" rx="1"/><path d="M12 9v6M9 12h6"/><path d="M9 21v-3h6v3"/>',
    "bell": '<path d="M6 17v-6a6 6 0 0112 0v6l1.5 2h-15z"/><path d="M10 21h4"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-8"/><path d="M2 20.5h20"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    "question": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.6a2.5 2.5 0 015 .4c0 1.6-2.5 2-2.5 3.6"/><circle cx="12" cy="17" r=".7"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><circle cx="12" cy="8" r=".7"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v4h16v-4"/>',
}


def icon(name):
    """Retourne une icône SVG (une seule ligne, pour ne pas casser le Markdown)."""
    return f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{ICON_PATHS[name]}</svg>'


def dot(color):
    """Pastille de couleur pour identifier une classe de tumeur."""
    return f'<span class="dot" style="background:{color}"></span>'


LOGO_SVG = (
    '<svg class="logo" viewBox="0 0 48 48" aria-hidden="true">'
    '<rect width="48" height="48" rx="12" fill="#2a94d6"/>'
    '<path d="M20 10h8v10h10v8H28v10h-8V28H10v-8h10z" fill="#ffffff"/>'
    '<polyline points="12 24 19 24 22 17 26 31 29 24 36 24" fill="none" stroke="#2a94d6" '
    'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

# ---------------------------------------------------------------------------
# Style global (CSS) – palette bleu ciel
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f5fbff 0%, #e3f3fd 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1, h2, h3, h4, p, li, label, span, div { color: #1d3b53; }

    /* Icônes et pastilles */
    .ico {
        width: 1.2em; height: 1.2em; margin-right: 9px; vertical-align: -0.2em;
        fill: none; stroke: currentColor; stroke-width: 1.8;
        stroke-linecap: round; stroke-linejoin: round; color: #2a94d6;
    }
    .dot {
        display: inline-block; width: 11px; height: 11px; border-radius: 50%;
        margin-right: 9px; vertical-align: middle;
    }

    /* En-tête */
    .hero {
        text-align: center; padding: 6px 0 14px 0; margin-bottom: 18px;
        border-bottom: 3px solid #6cc0ee;
    }
    .brand { display: inline-flex; align-items: center; gap: 14px; }
    .brand .logo { width: 54px; height: 54px; }
    .hero h1 {
        font-size: 2.6rem; font-weight: 700; margin: 0; padding: 0;
        color: #1c6ea4; -webkit-text-fill-color: #1c6ea4;
    }
    .hero p { font-size: 1.05rem; color: #5f7d93; margin-top: 6px; }

    /* Cartes */
    .card {
        background: #ffffff; border: 1px solid #cfe6f5; border-top: 4px solid #6cc0ee;
        border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(42,148,214,0.10);
    }
    .card h3 { margin-top: 0; color: #1c6ea4; }
    .objective {
        background: #e6f4fc; border-left: 6px solid #2a94d6;
        border-radius: 12px; padding: 18px 24px; margin: 10px 0 18px 0;
    }
    .objective h3 { margin-top: 0; color: #1c6ea4; }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        background: #d8edfa; color: #1c6ea4; font-size: 0.8rem; font-weight: 600;
        margin-right: 6px; margin-bottom: 6px;
    }
    .warning-box {
        background: #fff4dc; border-left: 5px solid #e0a100; color: #5c4300;
        border-radius: 10px; padding: 12px 18px; font-size: 0.9rem;
    }
    .warning-box .ico { color: #b57f00; }
    .res-title { margin: 0 0 6px 0; color: #1c6ea4; }

    /* Diaporama */
    .slideshow {
        position: relative; width: 100%; height: 360px; overflow: hidden;
        border-radius: 14px; box-shadow: 0 6px 24px rgba(42,148,214,0.28);
        border: 3px solid #ffffff; background: #0b1f30;
    }
    .slideshow .slide {
        position: absolute; inset: 0; opacity: 0;
        animation-name: slideFade; animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
    }
    .slideshow .slide img {
        width: 100%; height: 100%; object-fit: contain; background: #0b1f30;
    }
    .slideshow .caption {
        position: absolute; left: 0; right: 0; bottom: 0; padding: 14px 20px;
        background: linear-gradient(transparent, rgba(11,31,48,0.92));
        font-weight: 600; font-size: 1.05rem; color: #f5fbff;
    }

    /* Widgets Streamlit */
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #cfe6f5; border-radius: 10px; padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; color: #5f7d93; }
    .stTabs [aria-selected="true"] { color: #1c6ea4; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #2a94d6; }
    .stButton > button[kind="primary"] {
        background-color: #2a94d6; border: none; color: #ffffff; font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover { background-color: #1d7fbd; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Diaporama d'images de tumeurs
# ---------------------------------------------------------------------------
def _collect_images(max_per_class=6):
    """Collecte des images (chemin, légende) depuis les dossiers connus."""
    exts = (".jpg", ".jpeg", ".png")
    items = []

    # 1) Dossiers dédiés au diaporama (récursif)
    for folder in (SLIDESHOW_FOLDER, UPLOAD_FOLDER):
        if os.path.isdir(folder):
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    if f.lower().endswith(exts):
                        label = os.path.basename(root).lower()
                        legende = CLASS_INFO.get(label, {}).get("titre", "Image IRM cérébrale")
                        items.append((os.path.join(root, f), legende))

    # 2) Échantillon du jeu de test, par classe
    if os.path.isdir(TEST_FOLDER):
        for cls in class_names:
            cls_dir = os.path.join(TEST_FOLDER, cls)
            if os.path.isdir(cls_dir):
                files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(exts)]
                for f in files[:max_per_class]:
                    items.append((os.path.join(cls_dir, f), CLASS_INFO[cls]["titre"]))
    return items[:30]


@st.cache_data(show_spinner=False)
def build_slideshow_html(signature, interval_seconds=4):
    """Construit le HTML du diaporama (images réduites -> léger)."""
    items = _collect_images()
    if len(items) == 0:
        return ""

    slides = []
    for i, (path, caption) in enumerate(items):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((900, 600))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            continue
        slides.append(
            f'<div class="slide" style="animation-delay:{i * interval_seconds}s;">'
            f'<img src="data:image/jpeg;base64,{b64}">'
            f'<div class="caption">{caption}</div></div>'
        )
    n = len(slides)
    if n == 0:
        return ""

    total = n * interval_seconds
    seg = 100 / n
    fade = min(seg * 0.2, 4)
    # Chaque image démarre avec un délai de i * interval -> défilement dans l'ordre.
    slides_html = "".join(slides)
    css = f"""
    <style>
    .slideshow .slide {{ animation-duration: {total}s; }}
    @keyframes slideFade {{
        0% {{ opacity: 0; }}
        {fade:.2f}% {{ opacity: 1; }}
        {seg - fade:.2f}% {{ opacity: 1; }}
        {seg:.2f}% {{ opacity: 0; }}
        100% {{ opacity: 0; }}
    }}
    </style>
    """
    return css + f'<div class="slideshow">{slides_html}</div>'


def folder_signature():
    """Change quand le contenu des dossiers change -> invalide le cache."""
    sig = []
    for folder in (SLIDESHOW_FOLDER, UPLOAD_FOLDER, TEST_FOLDER):
        if os.path.isdir(folder):
            for root, _, files in os.walk(folder):
                sig.append((root, len(files)))
    return str(sig)


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------
model_path_tf = "Rihanatou_BANKOLE_model.h5"
model_path_pth = "Rihanatou_BANKOLE_model.torch"


@st.cache_resource
def load_tf_model():
    if not os.path.exists(model_path_tf):
        st.error(f"Le fichier {model_path_tf} est introuvable.")
        return None
    return tf.keras.models.load_model(model_path_tf)


@st.cache_resource
def load_pt_model():
    if not os.path.exists(model_path_pth):
        st.error(f"Le fichier {model_path_pth} est introuvable.")
        return None
    model = get_pretrained_model()
    model.load_state_dict(torch.load(model_path_pth, map_location=torch.device("cpu")))
    model.eval()
    return model


model_tf = load_tf_model()
model_pth = load_pt_model()


# ---------------------------------------------------------------------------
# Prétraitement (identique à l'entraînement : pixels dans [-1, 1])
# ---------------------------------------------------------------------------
def preprocess_tf(img):
    img = img.resize((224, 224)).convert("RGB")
    img_array = np.array(img).astype("float32")
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)


PT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def preprocess_pt(img):
    return PT_TRANSFORM(img.convert("RGB")).unsqueeze(0)


# ===========================================================================
# EN-TÊTE
# ===========================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="brand">{LOGO_SVG}<h1>NeuroScan</h1></div>
        <p>Classification des tumeurs cérébrales par Intelligence Artificielle à partir d'images IRM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Diaporama
slideshow_html = build_slideshow_html(folder_signature())
if slideshow_html:
    st.markdown(slideshow_html, unsafe_allow_html=True)
else:
    st.info(
        "Pour afficher le diaporama d'images de tumeurs, placez des images (.jpg/.png) "
        f"dans le dossier `{SLIDESHOW_FOLDER}/` (idéalement dans des sous-dossiers "
        "`glioma`, `meningioma`, `pituitary`, `notumor`)."
    )

# Objectif du projet
st.markdown(
    f"""
    <div class="objective">
        <h3>{icon("clipboard")}Objectif du projet</h3>
        <p>
        Les tumeurs cérébrales sont difficiles à diagnostiquer rapidement : l'analyse d'une IRM
        demande du temps et une grande expertise. Ce projet vise à <b>développer et comparer deux
        modèles de Deep Learning (TensorFlow et PyTorch)</b> capables de classer automatiquement
        une IRM cérébrale en <b>4 catégories</b> : gliome, méningiome, tumeur de l'hypophyse
        ou absence de tumeur.
        </p>
        <p style="margin-bottom:0;">
        Il s'agit d'un <b>outil d'aide à la décision et de sensibilisation</b>, destiné à appuyer
        le travail des professionnels de santé et à informer le public sur ces pathologies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_analyse, tab_types, tab_causes, tab_prevention, tab_eval = st.tabs([
    "Analyser une IRM",
    "Types de tumeurs",
    "Causes et facteurs de risque",
    "Prévention et prise en charge",
    "Évaluation des modèles",
])

# ===========================================================================
# ONGLET 1 : ANALYSE
# ===========================================================================
with tab_analyse:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("### Paramètres")
        framework = st.radio("Framework d'IA", ("TensorFlow", "PyTorch"))
        uploaded_files = st.file_uploader(
            "Importer une ou plusieurs IRM cérébrales",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
        )
        mode_pred = st.radio(
            "Mode de test",
            ("Toutes les images importées", "Sélection aléatoire"),
            help="« Sélection aléatoire » ne teste que quelques images tirées au hasard "
                 "parmi celles importées (rapide).",
        )
        n_random = 3
        if mode_pred == "Sélection aléatoire":
            n_random = st.number_input(
                "Nombre d'images tirées au hasard", min_value=1, max_value=50, value=3, step=1
            )
        st.markdown(
            f'<div class="warning-box">{icon("alert")}Résultat à but éducatif uniquement. '
            "Il ne constitue pas un diagnostic médical.</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        if not uploaded_files:
            st.markdown(
                f'<div class="card"><h3>{icon("scan")}Comment ça marche ?</h3>'
                "<ol><li>Choisissez le framework (TensorFlow ou PyTorch).</li>"
                "<li>Importez une ou plusieurs images IRM.</li>"
                "<li>Cliquez sur <b>Lancer l'analyse</b>.</li>"
                "<li>Consultez la classe prédite, la confiance et les explications associées.</li></ol></div>",
                unsafe_allow_html=True,
            )
        else:
            st.write(f"**{len(uploaded_files)} image(s) importée(s).**")

            if st.button("Lancer l'analyse", type="primary"):
                if framework == "TensorFlow" and model_tf is None:
                    st.error("Le modèle TensorFlow n'est pas chargé.")
                    st.stop()
                if framework == "PyTorch" and model_pth is None:
                    st.error("Le modèle PyTorch n'est pas chargé.")
                    st.stop()

                if mode_pred == "Sélection aléatoire":
                    k = min(int(n_random), len(uploaded_files))
                    selected_files = random.sample(list(uploaded_files), k)
                    st.info(f"{k} image(s) tirée(s) au hasard sur {len(uploaded_files)}.")
                else:
                    selected_files = list(uploaded_files)

                results = []
                with st.spinner("Analyse en cours..."):
                    for uploaded_file in selected_files:
                        image = Image.open(uploaded_file)

                        if framework == "TensorFlow":
                            preds = model_tf.predict(preprocess_tf(image), verbose=0)[0]
                        else:
                            with torch.no_grad():
                                out = model_pth(preprocess_pt(image))
                                preds = torch.nn.functional.softmax(out, dim=1)[0].numpy()

                        idx = int(np.argmax(preds))
                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": class_names[idx],
                            "confidence": float(preds[idx]),
                            "probs": preds,
                            "image": image,
                        })

                st.subheader("Résultats")
                st.table([
                    {
                        "Fichier": r["filename"],
                        "Prédiction": CLASS_INFO[r["prediction"]]["titre"],
                        "Confiance": f"{r['confidence']*100:.1f}%",
                    }
                    for r in results
                ])

                for r in results:
                    info = CLASS_INFO[r["prediction"]]
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(r["image"], caption=r["filename"], use_container_width=True)
                    with c2:
                        st.markdown(
                            f'<h4 class="res-title">{dot(info["color"])}{info["titre"]} '
                            f'— {r["confidence"]*100:.1f}%</h4>',
                            unsafe_allow_html=True,
                        )
                        st.write(info["resume"])
                        st.caption("**À surveiller :** " + info["signes"])
                        prob_df = pd.DataFrame({
                            "Classe": [CLASS_INFO[c]["titre"] for c in class_names],
                            "Probabilité": r["probs"],
                        }).set_index("Classe")
                        st.bar_chart(prob_df, color="#2a94d6")
                    st.divider()

# ===========================================================================
# ONGLET 2 : TYPES
# ===========================================================================
with tab_types:
    st.markdown("### Les 4 classes reconnues par le modèle")
    cols = st.columns(3)
    for col, key in zip(cols, ["glioma", "meningioma", "pituitary"]):
        info = CLASS_INFO[key]
        with col:
            st.markdown(
                f'<div class="card"><h3>{dot(info["color"])}{info["titre"]}</h3>'
                f'<p>{info["resume"]}</p>'
                f'<p><span class="badge">Signes possibles</span><br>{info["signes"]}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div class="card"><h3>{dot(CLASS_INFO["notumor"]["color"])}Pas de tumeur</h3>'
        f'<p>{CLASS_INFO["notumor"]["resume"]} {CLASS_INFO["notumor"]["signes"]}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="card"><h3>{icon("dna")}Bénigne ou maligne ?</h3>'
        "<p>Une tumeur <b>bénigne</b> croît lentement et reste localisée, mais peut quand même "
        "comprimer des zones vitales du cerveau. Une tumeur <b>maligne</b> (cancer) envahit les "
        "tissus voisins et croît plus vite. Seule une analyse médicale (IRM, biopsie) permet de "
        "trancher.</p></div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# ONGLET 3 : CAUSES
# ===========================================================================
with tab_causes:
    st.markdown("### Quelles sont les causes des tumeurs cérébrales ?")
    st.write(
        "Dans la grande majorité des cas, **la cause exacte reste inconnue**. "
        "Plusieurs facteurs de risque ont toutefois été identifiés :"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="card">
            <h3>{icon("check")}Facteurs de risque établis</h3>
            <ul>
              <li><b>Radiations ionisantes</b> : radiothérapie de la tête dans l'enfance, expositions
                  importantes (ex. scanners répétés non justifiés).</li>
              <li><b>Prédispositions génétiques</b> : neurofibromatose (NF1, NF2), syndrome de
                  Li-Fraumeni, maladie de von Hippel-Lindau, sclérose tubéreuse de Bourneville.</li>
              <li><b>Âge</b> : le risque augmente avec l'âge pour les gliomes et méningiomes.</li>
              <li><b>Antécédents familiaux</b> de tumeurs cérébrales (rare).</li>
              <li><b>Facteurs hormonaux</b> : les méningiomes sont plus fréquents chez la femme.</li>
              <li><b>Immunodéficience</b> sévère (risque de lymphome cérébral).</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
            <h3>{icon("question")}Facteurs étudiés, non prouvés</h3>
            <ul>
              <li><b>Téléphones portables</b> : les études n'ont pas démontré de lien causal
                  (classés « possiblement cancérogènes » par le CIRC, faute de preuves solides).</li>
              <li><b>Expositions chimiques</b> (pesticides, solvants, industrie pétrochimique) :
                  résultats contradictoires.</li>
              <li><b>Champs électromagnétiques</b> basse fréquence : pas de preuve concluante.</li>
              <li><b>Traumatismes crâniens</b> : lien non établi pour la plupart des tumeurs.</li>
            </ul>
            </div>
            <div class="warning-box">{icon("info")}Avoir un facteur de risque ne signifie pas que l'on aura une tumeur,
            et beaucoup de personnes touchées n'en ont aucun.</div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# ONGLET 4 : PRÉVENTION
# ===========================================================================
with tab_prevention:
    st.markdown("### Comment limiter les risques et réagir ?")
    st.write(
        "Il n'existe **pas de moyen de prévention garanti**, mais certains réflexes réduisent "
        "les risques et, surtout, permettent un diagnostic précoce."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card"><h3>{icon("shield")}Réduire les risques</h3>
            <ul>
              <li>Éviter les examens radiologiques de la tête non nécessaires, surtout chez l'enfant.</li>
              <li>Respecter les protections lors d'expositions professionnelles aux radiations ou
                  produits chimiques.</li>
              <li>Adopter une bonne hygiène de vie : alimentation équilibrée, activité physique,
                  pas de tabac.</li>
              <li>Consulter un conseil génétique en cas d'antécédents familiaux.</li>
            </ul></div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card"><h3>{icon("bell")}Quand consulter ?</h3>
            <ul>
              <li>Maux de tête inhabituels, persistants ou qui s'aggravent (surtout le matin).</li>
              <li>Crise d'épilepsie inexpliquée.</li>
              <li>Troubles de la vision, de l'audition, de l'équilibre ou de la parole.</li>
              <li>Nausées/vomissements répétés sans cause, troubles de la mémoire ou de
                  la personnalité.</li>
              <li>Faiblesse ou engourdissement d'un côté du corps.</li>
            </ul></div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card"><h3>{icon("hospital")}Prise en charge</h3>
            <ul>
              <li><b>Diagnostic</b> : examen neurologique, IRM, parfois biopsie.</li>
              <li><b>Chirurgie</b> pour retirer tout ou partie de la tumeur.</li>
              <li><b>Radiothérapie</b> / radiochirurgie (ex. Gamma Knife).</li>
              <li><b>Chimiothérapie</b> et thérapies ciblées selon le type.</li>
              <li><b>Surveillance</b> simple pour certaines tumeurs bénignes à croissance lente.</li>
              <li><b>Rééducation</b> et soutien psychologique.</li>
            </ul></div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="warning-box">{icon("alert")}Ces informations sont générales et éducatives. '
        "Seul un médecin peut poser un diagnostic et proposer un traitement adapté.</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# ONGLET 5 : ÉVALUATION
# ===========================================================================
with tab_eval:
    st.markdown(
        f'<h3>{icon("chart")}Comparer TensorFlow et PyTorch sur un jeu de test étiqueté</h3>',
        unsafe_allow_html=True,
    )
    test_dir = st.text_input("Chemin du dossier de test", value=TEST_FOLDER)
    eval_mode = st.radio(
        "Étendue de l'évaluation",
        ("Échantillon aléatoire", "Tout le jeu de test"),
        horizontal=True,
        help="L'échantillon aléatoire est rapide ; « Tout le jeu de test » peut être long.",
    )
    n_eval = 3
    if eval_mode == "Échantillon aléatoire":
        n_eval = st.number_input(
            "Nombre d'images tirées au hasard", min_value=1, max_value=500, value=3, step=1,
            key="n_eval",
        )
    run_eval = st.button("Lancer l'évaluation")

    def load_test_set(test_dir):
        filepaths, labels = [], []
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(test_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepaths.append(os.path.join(cls_dir, fname))
                    labels.append(idx)
        return filepaths, labels

    def evaluate_tf_model(filepaths):
        y_pred, y_prob, times = [], [], []
        for fp in filepaths:
            x = preprocess_tf(Image.open(fp))
            t0 = time.time()
            preds = model_tf.predict(x, verbose=0)[0]
            times.append(time.time() - t0)
            y_pred.append(int(np.argmax(preds)))
            y_prob.append(preds)
        return np.array(y_pred), np.array(y_prob), float(np.mean(times))

    def evaluate_pt_model(filepaths):
        y_pred, y_prob, times = [], [], []
        with torch.no_grad():
            for fp in filepaths:
                x = preprocess_pt(Image.open(fp))
                t0 = time.time()
                probs = torch.nn.functional.softmax(model_pth(x), dim=1)[0]
                times.append(time.time() - t0)
                y_pred.append(int(torch.argmax(probs).item()))
                y_prob.append(probs.numpy())
        return np.array(y_pred), np.array(y_prob), float(np.mean(times))

    def show_model_report(title, y_true, y_pred, y_prob, avg_time, model_path):
        st.markdown(f"#### {title}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else float("nan")
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc = float("nan")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Précision (accuracy)", f"{acc*100:.1f}%")
        c2.metric("AUC (ovr)", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        c3.metric("Inférence moy.", f"{avg_time*1000:.0f} ms")
        c4.metric("Taille du modèle", f"{size_mb:.0f} Mo")

        report_dict = classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, digits=3, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report_dict).transpose().round(3))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(f"Matrice de confusion — {title}")
        st.pyplot(fig)

    if run_eval:
        if not os.path.isdir(test_dir):
            st.error(f"Dossier introuvable : {test_dir}")
        elif model_tf is None or model_pth is None:
            st.error("Les deux modèles doivent être chargés pour comparer.")
        else:
            filepaths, y_true = load_test_set(test_dir)
            if eval_mode == "Échantillon aléatoire" and len(filepaths) > 0:
                k = min(int(n_eval), len(filepaths))
                idxs = random.sample(range(len(filepaths)), k)
                filepaths = [filepaths[i] for i in idxs]
                y_true = [y_true[i] for i in idxs]
                st.info(
                    f"{k} image(s) tirée(s) au hasard : "
                    + ", ".join(f"{os.path.basename(p)} ({class_names[t]})"
                                for p, t in zip(filepaths, y_true))
                )
            y_true = np.array(y_true)
            if len(filepaths) == 0:
                st.warning("Aucune image étiquetée trouvée. Sous-dossiers attendus : "
                           + ", ".join(class_names))
            else:
                st.write(f"Évaluation sur **{len(filepaths)}** images, **{len(class_names)}** classes...")
                with st.spinner("Modèle TensorFlow..."):
                    y_pred_tf, y_prob_tf, t_tf = evaluate_tf_model(filepaths)
                show_model_report("TensorFlow", y_true, y_pred_tf, y_prob_tf, t_tf, model_path_tf)
                st.divider()
                with st.spinner("Modèle PyTorch..."):
                    y_pred_pt, y_prob_pt, t_pt = evaluate_pt_model(filepaths)
                show_model_report("PyTorch", y_true, y_pred_pt, y_prob_pt, t_pt, model_path_pth)

# Pied de page
st.markdown(
    '<p style="text-align:center;color:#5f7d93;font-size:0.85rem;margin-top:30px;">'
    "NeuroScan — Projet de classification des tumeurs cérébrales par Deep Learning "
    "(TensorFlow & PyTorch) · Outil éducatif, non destiné au diagnostic médical.</p>",
    unsafe_allow_html=True,
)

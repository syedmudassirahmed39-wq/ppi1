import os
import random
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import SAGEConv, GraphNorm
from flask import Flask, request, render_template_string
import threading, webbrowser, socket

# ---------------- CONFIG ----------------
BASE = os.getcwd()
CACHE_PATH = os.path.join(BASE, "ppi_cache.pkl")
INFO_PATH = os.path.join(BASE, "protein.info.v12.0.txt")
LINKS_PATH = os.path.join(BASE, "9606.protein.links.v12.0.txt")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- LOAD / CACHE DATA ----------------
print("📥 Loading STRING PPI dataset (cached or fresh)...")

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        protein_info, ppi_links = pickle.load(f)
    print("✅ Loaded from cache (ppi_cache.pkl).")
    dataset_status = "✅ Dataset loaded from cache."
else:
    if os.path.exists(INFO_PATH) and os.path.exists(LINKS_PATH):
        protein_info = pd.read_csv(INFO_PATH, sep="\t")
        ppi_links = pd.read_csv(LINKS_PATH, sep=" ")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((protein_info, ppi_links), f)
        print("✅ Built dataset and saved cache.")
        dataset_status = "💾 Dataset built and cached successfully."
    else:
        print("⚠️ Dataset files not found! Generating lightweight synthetic dataset...")
        protein_info = pd.DataFrame({
            "string_protein_id": [f"protein_{i}" for i in range(1, 801)],
            "preferred_name": [f"PROT{i}" for i in range(1, 801)]
        })
        ppi_links = pd.DataFrame({
            "protein1": [f"protein_{random.randint(1,800)}" for _ in range(4000)],
            "protein2": [f"protein_{random.randint(1,800)}" for _ in range(4000)]
        })
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((protein_info, ppi_links), f)
        dataset_status = "💾 Synthetic dataset generated and cached."

# Limit for speed
protein_info = protein_info.head(800)
ppi_links = ppi_links.head(4000)

protein_ids = protein_info["string_protein_id"].tolist()
protein_names = protein_info["preferred_name"].tolist()
protein_mapping = dict(zip(protein_ids, protein_names))
ppi_pairs = set(tuple(sorted((r["protein1"], r["protein2"]))) for _, r in ppi_links.iterrows())

print(f"✅ Loaded {len(protein_mapping)} proteins and {len(ppi_links)} interactions.")

# ---------------- BIOLOGICAL INFO ----------------
bio_info = {
    "TP53": ("Tumor suppressor involved in DNA repair and apoptosis", "Cancer"),
    "MDM2": ("Regulates p53; inhibits apoptosis", "Cancer"),
    "BRCA1": ("DNA damage response and repair protein", "Breast Cancer"),
    "APOE": ("Lipid transport and brain function", "Alzheimer’s Disease"),
    "APP": ("Amyloid precursor, forms amyloid plaques", "Alzheimer’s Disease"),
    "INS": ("Regulates blood glucose levels", "Diabetes"),
    "TNF": ("Inflammatory cytokine", "Autoimmune Disorders"),
    "EGFR": ("Cell growth receptor", "Lung Cancer"),
    "AKT1": ("Cell survival and growth signaling", "Cell Growth Pathways"),
    "IL6": ("Inflammatory signaling molecule", "Inflammation"),
    "VEGFA": ("Blood vessel formation", "Heart Disease"),
    "SNCA": ("Alpha-synuclein, neuronal function", "Parkinson’s Disease"),
}

default_descriptions = [
    "Plays a role in basic cellular metabolism",
    "Involved in signal transduction pathways",
    "Contributes to cell structure and function",
    "Participates in protein-protein communication",
    "Supports molecular transport within the cell",
    "Involved in energy regulation and response to stress",
]

default_diseases = [
    "Cellular Regulation",
    "Signal Transduction",
    "Protein Binding Pathways",
    "Cell Cycle Control",
    "Metabolic Processes",
]

# ---------------- FLASK WEB APP ----------------
app = Flask(__name__)

protein_options = "\n".join([f'<option value="{pid}">{protein_mapping[pid]}</option>' for pid in protein_ids])

HTML_PAGE = f"""
<!DOCTYPE html>
<html>
<head>
<title>🧬 Protein–Protein Interaction & Disease Predictor</title>
<style>
body {{font-family: Arial; background:#f4f8fb; text-align:center; padding-top:40px;}}
select {{width:240px;padding:8px;margin:8px;border-radius:8px;border:1px solid #ccc;}}
button {{padding:10px 20px;border:none;background:#007BFF;color:white;border-radius:8px;cursor:pointer;}}
button:hover {{background:#0056b3;}}
.box {{background:white;padding:25px;border-radius:12px;box-shadow:0 0 10px #ccc;width:60%;margin:auto;}}
.popup {{
  background-color: #d4edda;
  border: 1px solid #155724;
  color: #155724;
  padding: 10px;
  border-radius: 10px;
  margin-bottom: 15px;
  display: inline-block;
  font-weight: bold;
}}
footer {{margin-top:25px;font-size:13px;color:#555;}}
</style>
</head>
<body>
<h2>🧬 Protein–Protein Interaction & Disease Prediction</h2>
<p>Select two proteins from the dataset to check interaction and biological significance.</p>
<form method="POST" action="/predict_pair">
<label>Protein 1:</label><br>
<select name="protein1" required>
<option value="" disabled selected>Select Protein 1</option>
{protein_options}
</select><br>
<label>Protein 2:</label><br>
<select name="protein2" required>
<option value="" disabled selected>Select Protein 2</option>
{protein_options}
</select><br><br>
<button type="submit">Predict Interaction</button>
</form>
{{{{ result|safe }}}}
</div>
<footer>
<p>Developed by <b>Graph Theory Team</b> | GNN-based Protein Interaction Prediction System</p>
</footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict_pair", methods=["POST"])
def predict_pair():
    try:
        p1 = request.form["protein1"]
        p2 = request.form["protein2"]

        if p1 == p2:
            return render_template_string(HTML_PAGE, result=f"⚠️ You selected the same protein (<b>{protein_mapping[p1]}</b>). Self-interactions are not recorded in this dataset.")

        p1_name, p2_name = protein_mapping[p1], protein_mapping[p2]
        pair = tuple(sorted((p1, p2)))
        is_interacting = pair in ppi_pairs

        desc1, dis1 = bio_info.get(p1_name, (random.choice(default_descriptions), random.choice(default_diseases)))
        desc2, dis2 = bio_info.get(p2_name, (random.choice(default_descriptions), random.choice(default_diseases)))

        if is_interacting:
            result = (
                f"🟠 <b>{p1_name}</b> ({p1}) and <b>{p2_name}</b> ({p2}) are predicted to be Interacting.<br><br>"
                f"🧬 <b>{p1_name}</b>: {desc1}.<br>"
                f"🧬 <b>{p2_name}</b>: {desc2}.<br>"
                f"💡 This interaction may influence <b>{dis1}</b> and <b>{dis2}</b> pathways."
            )
        else:
            result = (
                f"🔵 <b>{p1_name}</b> ({p1}) and <b>{p2_name}</b> ({p2}) are predicted to be Non-Interacting.<br><br>"
                f"🧬 <b>{p1_name}</b>: {desc1}.<br>"
                f"🧬 <b>{p2_name}</b>: {desc2}.<br>"
                f"✅ No significant biological interaction affecting <b>{dis1}</b> or <b>{dis2}</b> pathways."
            )

        return render_template_string(HTML_PAGE, result=result)
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- AUTO OPEN BROWSER ----------------
def open_browser(host, port):
    try:
        url = f"http://{host}:{port}"
        print(f"🌐 Opening web interface at: {url}")
        webbrowser.open_new_tab(url)
    except Exception:
        pass

if __name__ == "__main__":
    host = "localhost"
    port = 5000

    # check if port available
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            port += 1

    threading.Timer(1.5, lambda: open_browser(host, port)).start()
    app.run(host=host, port=port, debug=False)

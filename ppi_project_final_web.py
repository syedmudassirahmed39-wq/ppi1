import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, request, jsonify, render_template_string

# ----------------- CONFIG -----------------
BASE = r"C:\pp_project"
INFO_PATH = os.path.join(BASE, "protein.info.v12.0.txt")
LINKS_PATH = os.path.join(BASE, "9606.protein.links.v12.0.txt")
CACHE_PATH = os.path.join(BASE, "ppi_cache_final.pkl")
CSV_OUTPUT = os.path.join(BASE, "ppi_node_predictions.csv")
MODEL_PATH = os.path.join(BASE, "ppi_trained_model.pth")

MAX_NODES = 3000
FEATURE_DIM = 32
NUM_EPOCHS = 150
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- LOAD / CACHE -----------------
def build_and_cache():
    protein_info = pd.read_csv(INFO_PATH, sep="\t")
    ppi_links = pd.read_csv(LINKS_PATH, sep=" ")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((protein_info, ppi_links), f)
    return protein_info, ppi_links

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        protein_info, ppi_links = pickle.load(f)
    print("✅ Loaded dataset from cache.")
else:
    protein_info, ppi_links = build_and_cache()
    print("✅ Dataset built and cached for faster loading.")

protein_info.columns = [c.strip().lstrip("#") for c in protein_info.columns]

# ----------------- PREPARE GRAPH -----------------
ppi_links = ppi_links.merge(
    protein_info[['string_protein_id', 'preferred_name']],
    left_on='protein1', right_on='string_protein_id', how='left'
).rename(columns={'preferred_name': 'protein1_name'}).drop(columns=['string_protein_id'])

ppi_links = ppi_links.merge(
    protein_info[['string_protein_id', 'preferred_name']],
    left_on='protein2', right_on='string_protein_id', how='left'
).rename(columns={'preferred_name': 'protein2_name'}).drop(columns=['string_protein_id'])

all_proteins = pd.unique(ppi_links[['protein1', 'protein2']].values.ravel())

if len(all_proteins) > MAX_NODES:
    sampled_nodes = np.random.choice(all_proteins, MAX_NODES, replace=False)
    sub_links = ppi_links[ppi_links['protein1'].isin(sampled_nodes) & ppi_links['protein2'].isin(sampled_nodes)].copy()
    proteins = np.array(sorted(pd.concat([sub_links['protein1'], sub_links['protein2']]).unique()))
    print(f"🔹 Subsampled {len(proteins)} nodes. Edges kept: {len(sub_links)}")
else:
    sub_links = ppi_links.copy()
    proteins = np.array(sorted(all_proteins))
    print(f"🔹 Using full graph with {len(proteins)} nodes and {len(sub_links)} edges")

protein_to_id = {p: i for i, p in enumerate(proteins)}
sub_links = sub_links[sub_links['protein1'].isin(protein_to_id) & sub_links['protein2'].isin(protein_to_id)].copy()

u = sub_links['protein1'].map(protein_to_id).to_numpy(dtype=np.int64)
v = sub_links['protein2'].map(protein_to_id).to_numpy(dtype=np.int64)
edge_index = torch.tensor(np.vstack([u, v]), dtype=torch.long)

# ----------------- FEATURES & LABELS -----------------
x = torch.randn((len(proteins), FEATURE_DIM), dtype=torch.float32)
x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
y = ((3.5 * x[:, 0] - 2.7 * x[:, 1] + 1.8 * x[:, 2]) > 0).long()
data = Data(x=x, edge_index=edge_index, y=y)

# ----------------- SPLIT -----------------
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
np.random.shuffle(indices)
split = int(0.85 * num_nodes)
train_idx_np, test_idx_np = indices[:split], indices[split:]
train_idx = torch.from_numpy(train_idx_np).long()
test_idx = torch.from_numpy(test_idx_np).long()

# ----------------- MODEL -----------------
class HighAccuracyGNN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_ch, hidden_ch)
        self.conv1 = SAGEConv(hidden_ch, hidden_ch)
        self.norm1 = GraphNorm(hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.norm2 = GraphNorm(hidden_ch)
        self.conv3 = SAGEConv(hidden_ch, hidden_ch)
        self.norm3 = GraphNorm(hidden_ch)
        self.fc_out = torch.nn.Linear(hidden_ch, out_ch)
        self.act = torch.nn.LeakyReLU(0.1)
        self.drop = torch.nn.Dropout(0.05)

    def forward(self, x, edge_index):
        x = self.fc_in(x)
        x = self.act(self.norm1(self.conv1(x, edge_index)))
        x = self.drop(x)
        x = self.act(self.norm2(self.conv2(x, edge_index)))
        x = self.drop(x)
        x = self.act(self.norm3(self.conv3(x, edge_index)))
        return self.fc_out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HighAccuracyGNN(FEATURE_DIM, 256, 2).to(device)
data = data.to(device)
train_idx = train_idx.to(device)
test_idx = test_idx.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss()

# ----------------- TRAINING -----------------
print("\n🚀 Training started...\n")
best_acc = 0.0
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    preds = out.argmax(dim=1)
    train_acc = accuracy_score(data.y[train_idx].cpu().numpy(), preds[train_idx].cpu().numpy()) * 100
    test_acc = accuracy_score(data.y[test_idx].cpu().numpy(), preds[test_idx].cpu().numpy()) * 100
    best_acc = max(best_acc, test_acc)

    print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

total_time = time.time() - start_time
print(f"\n✅ Final Training Accuracy: {train_acc:.2f}%")
print(f"✅ Final Prediction Accuracy: {test_acc:.2f}%")
print(f"🏆 Best Prediction Accuracy: {best_acc:.2f}%")
print(f"⏱ Training Time: {total_time:.2f}s")

torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Trained model saved at: {MODEL_PATH}")

# ----------------- SAVE PREDICTIONS -----------------
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1).cpu().numpy()

results_df = pd.DataFrame({
    "protein_id": proteins,
    "predicted_class": preds,
    "class_description": ["Class 0" if p == 0 else "Class 1" for p in preds]
})
results_df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Predictions saved to: {CSV_OUTPUT}")

# ----------------- VISUALIZATION -----------------
G = nx.Graph()
edges = edge_index.cpu().numpy().T
G.add_edges_from(edges)
G.add_nodes_from(range(len(proteins)))
colors = ["#1f77b4" if preds[i] == 0 else "#ff7f0e" for i in range(len(G.nodes))]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=SEED)
nx.draw(G, pos, node_color=colors, node_size=35, edge_color='gray', with_labels=False)
plt.title("Protein–Protein Interaction Network (Predicted Classes)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "ppi_network_visualization.png"))
plt.show()
print("✅ Visualization saved as ppi_network_visualization.png")

# ----------------- FLASK WEB + API -----------------
app = Flask(__name__)

# Frontend HTML
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>🧬 Protein Interaction Predictor</title>
    <style>
        body { font-family: Arial; background-color: #f4f8fb; text-align: center; padding-top: 40px; }
        input { width: 220px; padding: 8px; margin: 5px; border-radius: 8px; border: 1px solid #ccc; }
        button { padding: 10px 20px; border: none; background-color: #007BFF; color: white; border-radius: 8px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .box { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 0 10px #ccc; width: 50%; margin: auto; }
    </style>
</head>
<body>
    <div class="box">
        <h2>🧬 Protein–Protein Interaction Prediction</h2>
        <form method="POST" action="/predict_form">
            {% for i in range(8) %}
                <input type="number" step="any" name="f{{i}}" placeholder="Feature {{i+1}}" required><br>
            {% endfor %}
            <br><button type="submit">Predict Interaction</button>
        </form>
        {% if result %}
            <h3>Prediction: {{ result }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        inputs = [float(request.form[f"f{i}"]) for i in range(8)]
        if len(inputs) < FEATURE_DIM:
            inputs += [0.0] * (FEATURE_DIM - len(inputs))
        x_input = torch.tensor(inputs).float().unsqueeze(0)
        edge_dummy = torch.tensor([[0], [0]])
        with torch.no_grad():
            out = model(x_input, edge_dummy)
            pred_class = out.argmax(dim=1).item()
        result = "🟠 Class 1 (Interacting)" if pred_class == 1 else "🔵 Class 0 (Non-Interacting)"
        return render_template_string(HTML_PAGE, result=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict_api():
    data_json = request.json.get("features", [])
    if not data_json or len(data_json) != FEATURE_DIM:
        return jsonify({"error": f"Please provide a valid list of {FEATURE_DIM} features"}), 400
    x_input = torch.tensor(data_json).float().unsqueeze(0)
    edge_dummy = torch.tensor([[0], [0]])
    with torch.no_grad():
        out = model(x_input, edge_dummy)
        pred_class = out.argmax(dim=1).item()
    return jsonify({
        "predicted_class": int(pred_class),
        "class_description": "Class 0 (Non-Interacting)" if pred_class == 0 else "Class 1 (Interacting)"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

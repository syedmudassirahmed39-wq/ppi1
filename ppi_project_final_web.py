import os
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
from flask import Flask, request, jsonify, render_template_string

# ----------------- CONFIG -----------------
FEATURE_DIM = 32
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- SYNTHETIC DATA (LIGHT) -----------------
print("⚗️ Using synthetic demo PPI data (lightweight for Render)...")

num_nodes = 150  # very small to fit memory
x = torch.randn((num_nodes, FEATURE_DIM), dtype=torch.float32)
edge_index = torch.randint(0, num_nodes, (2, 300))
y = ((3.5 * x[:, 0] - 2.7 * x[:, 1] + 1.8 * x[:, 2]) > 0).long()
data = Data(x=x, edge_index=edge_index, y=y)

# ----------------- MODEL -----------------
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_ch, hidden_ch)
        self.conv1 = SAGEConv(hidden_ch, hidden_ch)
        self.norm1 = GraphNorm(hidden_ch)
        self.fc_out = torch.nn.Linear(hidden_ch, out_ch)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = self.fc_in(x)
        x = self.act(self.norm1(self.conv1(x, edge_index)))
        return self.fc_out(x)

device = torch.device("cpu")
model = SimpleGNN(FEATURE_DIM, 64, 2).to(device)

# simulate pre-trained weights to avoid training (fast + low memory)
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.1)

print("✅ Model initialized — Ready for predictions!")

# ----------------- FLASK WEB APP -----------------
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>🧬 Protein Interaction Predictor</title>
<style>
body {font-family: Arial; background:#f4f8fb; text-align:center; padding-top:40px;}
input{width:220px;padding:8px;margin:5px;border-radius:8px;border:1px solid #ccc;}
button{padding:10px 20px;border:none;background:#007BFF;color:white;border-radius:8px;cursor:pointer;}
button:hover{background:#0056b3;}
.box{background:white;padding:25px;border-radius:12px;box-shadow:0 0 10px #ccc;width:50%;margin:auto;}
footer{margin-top:25px;font-size:13px;color:#555;}
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
<footer>
<p>Developed by <b>Syed Mudassir Ahmed</b> | GNN-based Protein Interaction Model</p>
</footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        # get 8 inputs from form
        inputs = [float(request.form[f"f{i}"]) for i in range(8)]
        if len(inputs) < FEATURE_DIM:
            inputs += [0.0] * (FEATURE_DIM - len(inputs))
        x_input = torch.tensor(inputs).float().unsqueeze(0)
        edge_dummy = torch.tensor([[0], [0]])  # dummy edge for forward()
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
        return jsonify({"error": f"Provide a valid list of {FEATURE_DIM} features"}), 400
    x_input = torch.tensor(data_json).float().unsqueeze(0)
    edge_dummy = torch.tensor([[0], [0]])
    with torch.no_grad():
        out = model(x_input, edge_dummy)
        pred_class = out.argmax(dim=1).item()
    return jsonify({
        "predicted_class": int(pred_class),
        "class_description": "Class 1 (Interacting)" if pred_class == 1 else "Class 0 (Non-Interacting)"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

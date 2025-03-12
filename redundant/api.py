from flask import Flask, request, send_file

import FreeCAD
import torch
import numpy as np

app = Flask(__name__)

# Load AI Model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3)  # Predict HVAC (x, y, z)
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
generator.load_state_dict(torch.load("hvac_model.pth"))
generator.eval()

# ✅ **Add this route to handle root URL ("/")**
@app.route("/")
def home():
    return "Welcome to the AI-based HVAC Optimization API! Use the /upload endpoint to upload a FreeCAD file."

# ✅ **File Upload & AI Optimization**
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded!", 400
    
    file = request.files["file"]
    file_path = f"uploaded_{file.filename}"
    file.save(file_path)

    # Load CAD model
    doc = FreeCAD.open(file_path)

    # Predict HVAC placement
    new_room = torch.tensor([[10, 8, 3]], dtype=torch.float32)
    predicted_hvac = generator(new_room).detach().numpy()[0]

    # Find and update HVAC unit
    for obj in doc.Objects:
        if obj.Name == "AC_Unit":
            obj.Placement.Base = FreeCAD.Vector(*predicted_hvac)
            break

    # Save updated file
    updated_file = f"optimized_{file.filename}"
    doc.recompute()
    doc.saveAs(updated_file)
    FreeCAD.closeDocument(doc.Name)

    return send_file(updated_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

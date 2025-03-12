import torch
import torch.nn as nn
import json
import numpy as np
import FreeCAD

# Define File Paths
MODEL_FILE = "hvac_model.pth"
CAD_FILE = "HVAC_Building.FCStd"
UPDATED_CAD_FILE = "HVAC_Building_Optimized.FCStd"
DATA_FILE = "cad_features.json"

# ✅ Ensure HVACModel Matches Train Model
class HVACModel(torch.nn.Module):
    def __init__(self):
        super(HVACModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),  # Increased neurons from 16 → 32
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # Output: Predicted HVAC Placement (x, y, z)
        )

    def forward(self, x):
        return self.model(x)

def load_trained_model(model_file):
    """ Load the trained AI model with the updated architecture """
    model = HVACModel()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print("✅ Loaded updated AI model.")
    return model

def load_room_dimensions(data_file):
    """ Load room dimensions from extracted JSON file """
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
        room = data["room"]
        return np.array([[room["length"], room["width"], room["height"]]])
    except Exception as e:
        print(f"❌ Error loading room data: {e}")
        return None

def predict_hvac_placement(model, room_dimensions):
    """ Predict optimal HVAC position using AI """
    room_tensor = torch.tensor(room_dimensions, dtype=torch.float32)
    predicted_hvac = model(room_tensor).detach().numpy()[0]
    print(f"🔮 AI-Predicted HVAC Placement (x, y, z): {predicted_hvac}")
    return predicted_hvac

def update_cad_with_hvac(cad_file, updated_file, hvac_position):
    """ Update the HVAC unit's placement in the FreeCAD CAD model """
    try:
        doc = FreeCAD.open(cad_file)

        # Find the HVAC unit in the CAD model
        hvac_unit = None
        for obj in doc.Objects:
            if obj.Name == "AC_Unit":
                hvac_unit = obj
                break

        if hvac_unit:
            # ✅ Print Before HVAC Placement
            current_position = hvac_unit.Placement.Base
            print(f"📍 Current HVAC Position (Before Optimization): ({current_position.x}, {current_position.y}, {current_position.z})")

            # ✅ Apply AI Predicted HVAC Placement
            hvac_unit.Placement.Base = FreeCAD.Vector(*hvac_position)
            print(f"✅ Updated HVAC Position (After AI Optimization): {hvac_position}")

        # Save updated CAD file
        doc.recompute()
        doc.saveAs(updated_file)
        FreeCAD.closeDocument(doc.Name)
        print(f"💾 Updated CAD model saved as {updated_file}")

    except Exception as e:
        print(f"❌ Error updating CAD model: {e}")

def main():
    # Load AI Model
    model = load_trained_model(MODEL_FILE)

    # Load room dimensions
    room_dimensions = load_room_dimensions(DATA_FILE)
    if room_dimensions is None:
        print("⚠️ Aborting: Room dimensions not available.")
        return

    # Predict HVAC placement
    hvac_position = predict_hvac_placement(model, room_dimensions)

    # Update the CAD model with the predicted HVAC placement
    update_cad_with_hvac(CAD_FILE, UPDATED_CAD_FILE, hvac_position)

if __name__ == "__main__":
    main()

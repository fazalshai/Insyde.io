import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

# Load extracted CAD features from JSON
DATA_FILE = "cad_features.json"
MODEL_FILE = "hvac_model.pth"

def load_training_data():
    """ Load room dimensions and HVAC placements from extracted JSON data & augment dataset """
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        room = data["room"]
        hvac_objects = data["hvac_objects"]

        if "AC_Unit" not in hvac_objects:
            print("Error: No AC unit found in the dataset.")
            return None, None

        # Base Training Data (Extracted from CAD)
        X = np.array([[room["length"], room["width"], room["height"]]])
        y = np.array([[hvac_objects["AC_Unit"]["x"], hvac_objects["AC_Unit"]["y"], hvac_objects["AC_Unit"]["z"]]])

        # 🔄 **Data Augmentation (Generating Synthetic Room Variations)**
        np.random.seed(42)  # Ensure reproducibility
        for _ in range(50):  # Generate 50 more samples
            length = room["length"] + np.random.uniform(-2, 2)  # Small variations
            width = room["width"] + np.random.uniform(-2, 2)
            height = room["height"] + np.random.uniform(-0.5, 0.5)

            hvac_x = hvac_objects["AC_Unit"]["x"] + np.random.uniform(-1, 1)
            hvac_y = hvac_objects["AC_Unit"]["y"] + np.random.uniform(-1, 1)
            hvac_z = hvac_objects["AC_Unit"]["z"] + np.random.uniform(-0.2, 0.2)

            X = np.vstack((X, [length, width, height]))
            y = np.vstack((y, [hvac_x, hvac_y, hvac_z]))

        print(f"✅ Training dataset expanded: {len(X)} samples")
        return X, y

    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None, None

# Load AI Model (HVAC Placement Predictor)
class HVACModel(torch.nn.Module):
    def __init__(self):
        super(HVACModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),  # Matches new training model
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: HVAC Placement (x, y, z)
        )

    def forward(self, x):
        return self.model(x)


def train_model(X_train, y_train, epochs=2000, learning_rate=0.005):
    """ Train the AI model with augmented data """
    model = HVACModel()
    criterion = nn.MSELoss()  # Mean Squared Error for regression task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"🟢 Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"💾 Model saved as {MODEL_FILE}")

    return model

def evaluate_model(model, X_test):
    """ Predict HVAC placement for new room dimensions """
    X_test = torch.tensor(X_test, dtype=torch.float32)
    predicted_hvac = model(X_test).detach().numpy()[0]
    print(f"🔍 AI Predicted HVAC Placement (x, y, z): {predicted_hvac}")
    return predicted_hvac

def main():
    # Load Data
    X_train, y_train = load_training_data()
    if X_train is None or y_train is None:
        print("⚠️ Training aborted due to data issues.")
        return

    # Train the model
    model = train_model(X_train, y_train)

    # Test with different room dimensions (Validation)
    validation_samples = np.array([
        [10, 8, 3],  # Existing room size
        [12, 10, 3.5],  # Slightly larger room
        [8, 6, 2.5],  # Smaller room
        [15, 12, 4]   # Large space
    ])
    
    for i, room in enumerate(validation_samples):
        print(f"\n🛠️ **Validation Sample {i+1}: Room {room}**")
        evaluate_model(model, room)

if __name__ == "__main__":
    main()

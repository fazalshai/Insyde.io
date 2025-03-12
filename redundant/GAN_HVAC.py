import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dummy dataset: (Room Length, Room Width, Room Height) → (HVAC X, HVAC Y, HVAC Z)
data = np.array([
    [10, 8, 3, 5, 4, 2.5],  # Example room size with ideal HVAC placement
    [12, 10, 3, 6, 5, 2.5],
    [8, 6, 2.5, 4, 3, 2],
    [15, 12, 3.5, 7, 6, 3],
])

# Normalize data
X_train = torch.tensor(data[:, :3], dtype=torch.float32)  # Room dimensions
y_train = torch.tensor(data[:, 3:], dtype=torch.float32)  # Suggested HVAC placements

# Define Generator (Neural Network to predict HVAC placement)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Predict HVAC (x, y, z)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
generator = Generator()
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = generator(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

# Predict optimal HVAC placement for a new room
new_room = torch.tensor([[11, 9, 3]], dtype=torch.float32)  # Room (L, W, H)
predicted_hvac = generator(new_room).detach().numpy()
print(f"Predicted HVAC Placement (x, y, z): {predicted_hvac[0]}")

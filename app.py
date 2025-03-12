import streamlit as st
import FreeCAD
import Mesh
import torch
import torch.nn as nn
import json
import numpy as np
import open3d as o3d
import os
import asyncio
import sys
import matplotlib.pyplot as plt


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


UPLOADED_CAD_FILE = "uploaded_model.FCStd"
OPTIMIZED_CAD_FILE = "optimized_model.FCStd"
STL_FILE_BEFORE = "before.stl"
STL_FILE_AFTER = "after.stl"
DATA_FILE = "cad_features.json"
MODEL_FILE = "hvac_model.pth"


class HVACModel(torch.nn.Module):
    def __init__(self):
        super(HVACModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),  # Matches training model
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predicts HVAC (x, y, z)
        )

    def forward(self, x):
        return self.model(x)

def load_trained_model():
   
    model = HVACModel()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model

def load_room_dimensions():
    
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        room = data["room"]
        return np.array([[room["length"], room["width"], room["height"]]])
    except Exception as e:
        st.error(f"Error loading room data: {e}")
        return None

def predict_hvac_placement(model, room_dimensions):
   
    room_tensor = torch.tensor(room_dimensions, dtype=torch.float32)
    predicted_hvac = model(room_tensor).detach().numpy()[0]
    return predicted_hvac

def visualize_cad_as_stl(cad_file, stl_file):
    
    doc = FreeCAD.open(cad_file)
    objects = [obj for obj in doc.Objects if obj.TypeId.startswith("Part::")]
    Mesh.export(objects, stl_file)
    FreeCAD.closeDocument(doc.Name)
    return stl_file

import plotly.graph_objects as go

def plot_3d_visualization(stl_file, title):
    
    try:
        mesh = o3d.io.read_triangle_mesh(stl_file)
        pcd = mesh.sample_points_uniformly(number_of_points=5000)
        points = np.asarray(pcd.points)

        # Assign colors based on component positions
        colors = []
        for point in points:
            if point[2] > 2.5:  # AC Unit (Higher Z position)
                colors.append("blue")
            elif point[1] < 1.0:  # Vents (Lower Y position)
                colors.append("green")
            elif 0.5 < point[2] < 2.5:  # Ducts
                colors.append("red")
            else:
                colors.append("gray")  # Other elements

        # Create interactive 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.8)
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Display in Streamlit
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error displaying 3D visualization: {e}")


def update_cad_with_hvac(hvac_position):
    """ Update HVAC unit's placement in the FreeCAD model """
    doc = FreeCAD.open(UPLOADED_CAD_FILE)
    hvac_unit = None
    for obj in doc.Objects:
        if obj.Name == "AC_Unit":
            hvac_unit = obj
            break
    if hvac_unit:
        hvac_unit.Placement.Base = FreeCAD.Vector(*hvac_position)
    doc.recompute()
    doc.saveAs(OPTIMIZED_CAD_FILE)
    FreeCAD.closeDocument(doc.Name)
    return OPTIMIZED_CAD_FILE

# Streamlit UI
st.title("AI-Based HVAC Optimization System")
st.write("Upload a CAD file to optimize HVAC placement.")

uploaded_file = st.file_uploader("Upload FreeCAD file (.FCStd)", type=["FCStd"])

if uploaded_file:
    # Save uploaded file
    with open(UPLOADED_CAD_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Load AI Model
    model = load_trained_model()

    # Load room dimensions
    room_dimensions = load_room_dimensions()
    if room_dimensions is not None:
        # Predict optimal HVAC placement
        predicted_hvac = predict_hvac_placement(model, room_dimensions)
        st.write(f"**Predicted HVAC Placement (x, y, z):** {predicted_hvac}")

        # Show Before Optimization 3D View
        st.subheader("Before Optimization (Original HVAC Placement)")
        visualize_cad_as_stl(UPLOADED_CAD_FILE, STL_FILE_BEFORE)
        plot_3d_visualization(STL_FILE_BEFORE, "Original HVAC Placement")

        # Apply AI Optimization
        optimized_file = update_cad_with_hvac(predicted_hvac)

        # Show After Optimization 3D View
        st.subheader("After Optimization (AI-Predicted HVAC Placement)")
        visualize_cad_as_stl(OPTIMIZED_CAD_FILE, STL_FILE_AFTER)
        plot_3d_visualization(STL_FILE_AFTER, "Optimized HVAC Placement")

        # Provide Download Link for Optimized CAD File
        with open(OPTIMIZED_CAD_FILE, "rb") as f:
            st.download_button("Download Optimized CAD File", f, file_name="optimized_model.FCStd")

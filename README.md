# AI-Based HVAC Optimization System

This project optimizes HVAC placement in a 3D building model using AI and FreeCAD.  
Users can upload a CAD file, AI predicts the best HVAC placement, and the updated model is automatically generated.

## Features

-   Extracts CAD features such as room dimensions and HVAC placements using FreeCAD.
-   Trains a neural network to predict optimal HVAC placement.
-   Applies AI-predicted HVAC placement to a FreeCAD model.
-   Provides an interactive 3D visualization of before and after HVAC placements.
-   Streamlit-based web app for easy user interaction.

## Project Structure

| File                      | Description                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `extract_cad_features.py` | Extracts room dimensions and HVAC placements from FreeCAD.                                               |
| `train_hvac_ai.py`        | Trains a neural network to predict HVAC placement.                                                       |
| `apply_hvac_ai.py`        | Uses AI to optimize HVAC placement in a FreeCAD model.                                                   |
| `app.py`                  | Streamlit web app for uploading CAD files, visualizing HVAC placements, and downloading optimized files. |

## Prerequisites

### Install Python 3.11.x

Download and install Python 3.11.x from the [official Python website](https://www.python.org/downloads/).

### Upgrade Pip

```sh
python -m pip install --upgrade pip
```

### Install Dependencies

```sh
pip install numpy pandas matplotlib scipy scikit-learn tensorflow torch torchvision torchaudio open3d streamlit flask boto3
```

### Install FreeCAD

-   Download and install FreeCAD from the [official website](https://www.freecad.org/).
-   Ensure FreeCAD is added to Python’s system path.

## Running the Project

### 1. Extract CAD Features

```sh
python extract_cad_features.py
```

This extracts room dimensions and HVAC placements from a FreeCAD file and saves the extracted data to `cad_features.json`.

### 2. Train the AI Model

```sh
python train_hvac_ai.py
```

This trains a deep learning model to predict optimal HVAC placement and saves the trained model as `hvac_model.pth`.

### 3. Apply AI to Optimize HVAC Placement

```sh
python apply_hvac_ai.py
```

This loads the trained AI model, updates the HVAC placement in the FreeCAD model, and saves the optimized FreeCAD model as `HVAC_Building_Optimized.FCStd`.

### 4. Run the Streamlit Web App

```sh
streamlit run app.py
```

-   Upload a FreeCAD `.FCStd` file.
-   View before and after HVAC placement in an interactive 3D visualization.
-   Download the optimized `.FCStd` file.
-   ![image](https://github.com/user-attachments/assets/a258da38-2f36-42de-ba46-47aeeaae1a7d)


## Output Examples



### Training Output Example

```
Epoch [0/2000], Loss: 12.5489
Epoch [200/2000], Loss: 0.6421
Epoch [400/2000], Loss: 0.1152
Model saved as hvac_model.pth
```

### AI Predicted HVAC Placement

```
AI-Predicted HVAC Placement (x, y, z): [6.2, 4.8, 2.4]
Current HVAC Position (Before Optimization): (8.0, 1.0, 2.5)
Updated HVAC Position (After AI Optimization): (6.2, 4.8, 2.4)
Updated CAD model saved as HVAC_Building_Optimized.FCStd
```

### **App Interface & Screenshots**

The **Streamlit web app** provides an interactive interface for uploading CAD files, visualizing HVAC placements, and downloading the optimized file.

#### **1. Home Page – Upload CAD File**

The homepage allows users to **upload a FreeCAD (`.FCStd`) file** for HVAC optimization.

![0c8c5958-b6a9-4d22-83b5-c010317acedc](https://github.com/user-attachments/assets/04664e74-17d5-4909-8f49-c13d31a896bd)

---

#### **2. Before Optimization – Original HVAC Placement**

After uploading the file, the app **visualizes the existing HVAC placement** in an **interactive 3D plot**.

---

#### **3. AI-Predicted HVAC Placement**

The AI model predicts an **optimized HVAC placement**, which is displayed as text in the app.

---

#### **4. After Optimization – Updated HVAC Placement**

The optimized **HVAC placement is applied to the CAD model**, and the updated 3D visualization is shown.

![43329ffd-57af-4429-b547-a185306e4ac7](https://github.com/user-attachments/assets/7ed28a71-a944-403a-a281-b58c57521a87)
---

#### **5. Download Optimized CAD File**

Users can **download the updated FreeCAD model** with the AI-optimized HVAC placement.

---


## Notes

-   Ensure FreeCAD is installed and added to Python’s path.
-   If the AI model fails to load, delete `hvac_model.pth` and retrain using `train_hvac_ai.py`.
-   The model hyperparameters can be adjusted in `train_hvac_ai.py` for better results.

## Future Improvements

-   Enhance AI model with better feature engineering.
-   Optimize FreeCAD integration for real-time CAD modifications.
-   Add cloud-based processing (AWS/GCP) for large-scale HVAC optimizations.

```

```

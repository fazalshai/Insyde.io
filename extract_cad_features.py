import FreeCAD
import Mesh
import json
import open3d as o3d
import numpy as np

# Define input and output files
CAD_FILE = "HVAC_Building.FCStd"
STL_FILE = "HVAC_Building.stl"
OUTPUT_JSON = "cad_features.json"

def load_cad_model(cad_file):
    """ Load FreeCAD document """
    try:
        doc = FreeCAD.open(cad_file)
        print(f"Loaded CAD file: {cad_file}")
        return doc
    except Exception as e:
        print(f"Error loading CAD file: {e}")
        return None

def extract_room_dimensions(doc):
    """ Extracts room dimensions assuming rooms are modeled as boxes """
    room_data = {}
    for obj in doc.Objects:
        if obj.TypeId == "Part::Box" and "Room" in obj.Name:
            room_data["name"] = obj.Name
            room_data["length"] = obj.Shape.BoundBox.XLength
            room_data["width"] = obj.Shape.BoundBox.YLength
            room_data["height"] = obj.Shape.BoundBox.ZLength
            break
    return room_data

def extract_object_placements(doc):
    """ Extract placements of HVAC-related objects (AC Unit, Vent, Duct) """
    object_positions = {}
    for obj in doc.Objects:
        pos = obj.Placement.Base  # Get object position
        object_positions[obj.Name] = {
            "x": pos.x,
            "y": pos.y,
            "z": pos.z
        }
    return object_positions

def convert_to_stl(doc, stl_file):
    """ Exports FreeCAD model as STL for airflow analysis """
    objects = [obj for obj in doc.Objects if obj.TypeId.startswith("Part::")]
    Mesh.export(objects, stl_file)
    print(f"Exported {CAD_FILE} to {stl_file}")

def analyze_airflow(stl_file, vent_position):
    """ Extract airflow paths using Open3D by detecting empty spaces and vents """
    try:
        mesh = o3d.io.read_triangle_mesh(stl_file)
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        points = np.asarray(pcd.points)

        # Airflow paths: points above the room's mid-height
        avg_z = np.mean(points[:, 2])
        empty_space_points = points[np.where(points[:, 2] > avg_z)]

        # Detect potential air outflow regions near vents
        vent_z_range = (vent_position["z"] - 0.2, vent_position["z"] + 0.2)
        vent_outflow_points = points[
            (points[:, 0] > vent_position["x"] - 0.5) & (points[:, 0] < vent_position["x"] + 0.5) &
            (points[:, 1] > vent_position["y"] - 0.5) & (points[:, 1] < vent_position["y"] + 0.5) &
            (points[:, 2] > vent_z_range[0]) & (points[:, 2] < vent_z_range[1])
        ]

        # Visualize airflow paths
        o3d.visualization.draw_geometries([pcd])

        return {
            "total_points": len(points),
            "empty_space_points": len(empty_space_points),
            "vent_outflow_points": len(vent_outflow_points),
            "empty_space_coordinates": empty_space_points.tolist(),
            "vent_outflow_coordinates": vent_outflow_points.tolist()
        }
    except Exception as e:
        print(f"Error analyzing airflow: {e}")
        return {}

def save_to_json(room_data, object_positions, airflow_data, output_json):
    """ Save extracted data to a structured JSON file """
    data = {
        "room": room_data,
        "hvac_objects": object_positions,
        "airflow_analysis": airflow_data
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved extracted data to {output_json}")

def main():
    doc = load_cad_model(CAD_FILE)
    if not doc:
        return
    
    room_data = extract_room_dimensions(doc)
    object_positions = extract_object_placements(doc)
    
    convert_to_stl(doc, STL_FILE)

    # Identify vent position for airflow analysis
    vent_position = object_positions.get("Vent", {"x": 0, "y": 0, "z": 0})
    airflow_data = analyze_airflow(STL_FILE, vent_position)

    save_to_json(room_data, object_positions, airflow_data, OUTPUT_JSON)

    # Close FreeCAD document
    FreeCAD.closeDocument(doc.Name)
    print("Closed CAD document.")

if __name__ == "__main__":
    main()

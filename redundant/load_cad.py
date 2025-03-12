import FreeCAD
import FreeCADGui
import Part

# Load the CAD file (replace with your actual file path)
cad_file = "C:\Fazal\AI HVAC\HVAC_Building.FCStd"

# Open the FreeCAD document
doc = FreeCAD.open(cad_file)
print(f"Loaded CAD File: {cad_file}")

# Get all objects in the CAD model
objects = doc.Objects
print(f"Total Objects in the Model: {len(objects)}")

# Print object details
for obj in objects:
    print(f"Object: {obj.Name}, Type: {obj.TypeId}")

# Close document after extraction
FreeCAD.closeDocument(doc.Name)

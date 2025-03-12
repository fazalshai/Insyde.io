import FreeCAD
import Part

# Create a new FreeCAD document
doc = FreeCAD.newDocument("HVAC_Building")

# Define a room (a simple box)
room = doc.addObject("Part::Box", "Room")
room.Length = 10  # Length in meters
room.Width = 8    # Width in meters
room.Height = 3   # Height in meters

# Define an AC unit (small box)
ac_unit = doc.addObject("Part::Box", "AC_Unit")
ac_unit.Length = 1
ac_unit.Width = 0.5
ac_unit.Height = 0.5
ac_unit.Placement.Base = FreeCAD.Vector(8, 1, 2.5)  # Position near the top

# Define a vent (small rectangle)
vent = doc.addObject("Part::Box", "Vent")
vent.Length = 0.5
vent.Width = 0.2
vent.Height = 0.1
vent.Placement.Base = FreeCAD.Vector(5, 0, 2)  # Position near the bottom

# Define a duct (connecting AC to vent)
duct = doc.addObject("Part::Box", "Duct")
duct.Length = 3
duct.Width = 0.5
duct.Height = 0.5
duct.Placement.Base = FreeCAD.Vector(6, 0.25, 2.5)

# Recompute and save the CAD file
doc.recompute()
cad_file = "HVAC_Building.FCStd"
FreeCAD.ActiveDocument.saveAs(cad_file)
print(f"Sample CAD model saved as {cad_file}")

# Close FreeCAD document
FreeCAD.closeDocument(doc.Name)

import wildmeshing as wm

input_file = "../assets/fingertip/fingertip.obj"
output_file = "../assets/fingertip/fingertip_tet"
edge_length = 0.05

print(f"Meshing {input_file} with target edge length {edge_length}...")

wm.tetrahedralize(input_file, output_file, edge_length_r=edge_length)

print("Meshing complete! Saved to", output_file)
import trimesh

def save_results_mesh(vertices, faces, filename):
  mesh = trimesh.Trimesh(vertices, faces, process=False)
  mesh.export(filename)
  print(f'save results to {filename}')
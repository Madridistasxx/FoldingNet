from pathlib import Path
import json
import trimesh

ROOT = Path("/mnt/data/datasets/shapenet_core")
PARENT_ROOT = Path("/home/RUS_CIP/st189459/FoldingNet")
TARGET_CLASS = "03001627"   # chair

# 读取 taxonomy
with open(PARENT_ROOT / "taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

name_map = {x["synsetId"]: x["name"] for x in taxonomy}
print("class name:", name_map[TARGET_CLASS])

# 找一个模型
model_dirs = [p for p in (ROOT / TARGET_CLASS).iterdir() if p.is_dir()]
model_dir = model_dirs[0]
obj_path = model_dir / "models" / "model_normalized.obj"

print("loading:", obj_path)
mesh = trimesh.load(obj_path, force="mesh")

print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))

# 采样成点云
points, _ = trimesh.sample.sample_surface(mesh, 1024)
print("sampled points:", points.shape)

mesh.show()
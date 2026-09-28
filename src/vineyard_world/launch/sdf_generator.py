from __future__ import annotations
import argparse
import pathlib
import random 
from typing import Tuple
from xml.etree import ElementTree as ET
from PIL import Image

# ───────────────────────────── CONFIG ──────────────────────────────

PKG_DIR = pathlib.Path(__file__).resolve().parents[1]  # src/vineyard_world
# Assets are referenced as model://vineyard_world/... in the SDF; launch_sim.launch.py puts
# the package's share folder on GZ_SIM_RESOURCE_PATH so Gazebo can resolve them.
MODEL_URI: str = "model://vineyard_world"

HEIGHTMAP_NAME: str = "vineyard_arc_slope_heightmap_bumpy_smooth_512.png"
HEIGHTMAP_PNG: str = str(PKG_DIR / "materials" / "png" / HEIGHTMAP_NAME) # 512×512 grayscale PNG (read locally)
HEIGHTMAP_SIZE_M: Tuple[float, float] = (50.0, 50.0) # (width, depth) in m
HEIGHTMAP_MAX_Z: float = 12.0 # Max elevation difference (m)

PLANT_MESH_BASE_PATH: str = f"{MODEL_URI}/mesh"  # Base URI for meshes
PLANT_MESH_COUNT: int = 4  # Number of available mesh files (Tree_1.obj to Tree_50.obj)
PLANT_SCALE: Tuple[float, float, float] = (1, 1, 1)
PLANT_ROLL: float = 1.57 # roll (rad) applied to every tree

PLANTS_PER_ROW: int = 12
NUM_ROWS: int = 5
PLANT_SPACING: float = 3.0 # Distance along X between trees
ROW_SPACING: float = 10.0 # Distance along Y between rows

PHYSICS_ENGINE: str = "ode" # "ode" or "bullet"
MAX_STEP_SIZE: float = 0.001 # seconds
REAL_TIME_FACTOR: float = 1.0

SDF_VERSION: str = "1.10"
WORLD_NAME: str = "heightmap_world"

# ────────────────────────── Helper utilities ───────────────────────

def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty‑print ElementTree in place."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def get_random_mesh_path() -> str:
    """
    Generate a random mesh file path from the available mesh files.
    
    Returns:
        Full path to a randomly selected mesh file (Tree_1.obj to Tree_50.obj)
    """
    # Generate random number between 1 and PLANT_MESH_COUNT (inclusive)
    random_num = random.randint(1, PLANT_MESH_COUNT)
    
    # Construct the mesh filename
    mesh_filename = f"Tree_{random_num}.obj"
    
    # Return the full URI
    return f"{PLANT_MESH_BASE_PATH}/{mesh_filename}"

class HeightmapSampler:
    """Convert (x, y) world coords ➔ height using the grayscale PNG."""
    
    def __init__(self, img_path: str, size_m: Tuple[float, float], max_z: float):
        self.img = Image.open(img_path).convert("L")
        self.px = self.img.load()
        self.w_px, self.h_px = self.img.size
        self.size_x, self.size_y = size_m
        self.max_z = max_z

    def height_at(self, x_m: float, y_m: float) -> float:
        u = int(((x_m + self.size_x / 2) / self.size_x) * (self.w_px - 1))
        v = int(((y_m + self.size_y / 2) / self.size_y) * (self.h_px - 1))
        u = max(0, min(self.w_px - 1, u))
        v = max(0, min(self.h_px - 1, v))
        intensity = self.px[u, (self.h_px - 1) - v]
        return (intensity / 255.0) * self.max_z

# ────────────────────────── SDF construction ───────────────────────

def build_world() -> ET.Element:
    """Build the complete SDF world with individual tree models using random meshes."""
    root = ET.Element("sdf", version=SDF_VERSION)
    world = ET.SubElement(root, "world", name=WORLD_NAME)

    # Add lighting
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "cast_shadows").text = "1"
    ET.SubElement(light, "pose").text = "-5 2 8 0 -0 0"
    ET.SubElement(light, "diffuse").text = "1.0 0.6 0.3 1"
    ET.SubElement(light, "specular").text = "0.8 0.4 0.2 1"
    attenuation = ET.SubElement(light, "attenuation")
    ET.SubElement(attenuation, "range").text = "1000"
    ET.SubElement(attenuation, "constant").text = "0.5"
    ET.SubElement(attenuation, "linear").text = "0.01"
    ET.SubElement(attenuation, "quadratic").text = "0.001"
    ET.SubElement(light, "direction").text = "-0.8 0.2 -0.3"

    # Physics configuration
    physics = ET.SubElement(world, "physics", name="1ms", type="ode")
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"

    # Plugins
    ET.SubElement(world, "plugin", name="gz::sim::systems::Physics", filename="gz-sim-physics-system")
    ET.SubElement(world, "plugin", name="gz::sim::systems::UserCommands", filename="gz-sim-user-commands-system")
    ET.SubElement(world, "plugin", name="gz::sim::systems::SceneBroadcaster", filename="gz-sim-scene-broadcaster-system")
    ET.SubElement(world, "plugin", name="gz::sim::systems::Contact", filename="gz-sim-contact-system")
    plugin_sensors = ET.SubElement(world, "plugin", {
    "filename": "gz-sim-sensors-system",
    "name": "gz::sim::systems::Sensors"
    })
    render_engine = ET.SubElement(plugin_sensors, "render_engine")
    render_engine.text = "ogre2"
    # Needed for the robot's IMU sensor to publish.
    ET.SubElement(world, "plugin", name="gz::sim::systems::Imu", filename="gz-sim-imu-system")


    # Environment settings
    ET.SubElement(world, "gravity").text = "0 0 -9.8"
    ET.SubElement(world, "magnetic_field").text = "5.5644999999999998e-06 2.2875799999999999e-05 -4.2388400000000002e-05"
    
    # Atmosphere
    ET.SubElement(world, "atmosphere", type="adiabatic")

    # Scene settings
    scene = ET.SubElement(world, "scene")
    ET.SubElement(scene, "ambient").text = "0.4 0.4 0.4 1"
    ET.SubElement(scene, "background").text = "0.7 0.7 0.7 1"
    ET.SubElement(scene, "shadows").text = "false"

    # Ground plane model
    add_ground_plane(world)
    
    # Add physics, heightmap, and individual trees with random meshes
    add_physics(world)
    add_heightmap(world)
    add_individual_trees_with_random_meshes(world)  # Updated function name

    return root

def add_ground_plane(parent: ET.Element) -> None:
    """Add ground plane model."""
    ground = ET.SubElement(parent, "model", name="ground_plane")
    ET.SubElement(ground, "static").text = "true"
    
    link = ET.SubElement(ground, "link", name="link")
    
    # Collision
    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    ET.SubElement(plane, "normal").text = "0 0 1"
    ET.SubElement(plane, "size").text = "100 100"
    
    surface = ET.SubElement(collision, "surface")
    friction = ET.SubElement(surface, "friction")
    ET.SubElement(friction, "ode")
    ET.SubElement(surface, "bounce")
    ET.SubElement(surface, "contact")
    
    # Visual
    visual = ET.SubElement(link, "visual", name="visual")
    geometry_v = ET.SubElement(visual, "geometry")
    plane_v = ET.SubElement(geometry_v, "plane")
    ET.SubElement(plane_v, "normal").text = "0 0 1"
    ET.SubElement(plane_v, "size").text = "100 100"
    
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = "0.8 0.8 0.8 1"
    ET.SubElement(material, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(material, "specular").text = "0.8 0.8 0.8 1"
    
    # Link properties
    ET.SubElement(link, "pose").text = "0 0 0 0 0 0"
    
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "pose").text = "0 0 0 0 0 0"
    ET.SubElement(inertial, "mass").text = "1"
    inertia = ET.SubElement(inertial, "inertia")
    ET.SubElement(inertia, "ixx").text = "1"
    ET.SubElement(inertia, "ixy").text = "0"
    ET.SubElement(inertia, "ixz").text = "0"
    ET.SubElement(inertia, "iyy").text = "1"
    ET.SubElement(inertia, "iyz").text = "0"
    ET.SubElement(inertia, "izz").text = "1"
    
    ET.SubElement(link, "enable_wind").text = "false"
    ET.SubElement(ground, "pose").text = "0 0 0 0 0 0"
    ET.SubElement(ground, "self_collide").text = "false"

def add_physics(parent: ET.Element) -> None:
    """Add physics configuration."""
    phy = ET.SubElement(parent, "physics", name=f"{PHYSICS_ENGINE}_physics", type=PHYSICS_ENGINE)
    ET.SubElement(phy, "max_step_size").text = str(MAX_STEP_SIZE)
    ET.SubElement(phy, "real_time_factor").text = str(REAL_TIME_FACTOR)
    
    if PHYSICS_ENGINE == "ode":
        # Simple ODE defaults; tweak as needed
        ode = ET.SubElement(phy, "ode")
        solver = ET.SubElement(ode, "solver")
        ET.SubElement(solver, "type").text = "quick"
    elif PHYSICS_ENGINE == "bullet":
        ET.SubElement(phy, "bullet") 

def add_heightmap(parent: ET.Element) -> None:
    """Add heightmap terrain."""
    hm = ET.SubElement(parent, "model", name="terrain")
    ET.SubElement(hm, "static").text = "true"
    
    link = ET.SubElement(hm, "link", name="link")
    
    # Collision geometry
    coll = ET.SubElement(link, "collision", name="collision")
    coll_geom = ET.SubElement(coll, "geometry")
    coll_geom.append(build_heightmap_geom())
    
    # Visual geometry
    vis = ET.SubElement(link, "visual", name="visual")
    vis_geom = ET.SubElement(vis, "geometry")
    vis_geom.append(build_heightmap_geom(textured=True))

def build_heightmap_geom(textured: bool = False) -> ET.Element:
    """Build heightmap geometry element."""
    geom = ET.Element("heightmap")
    ET.SubElement(geom, "uri").text = f"{MODEL_URI}/materials/png/{HEIGHTMAP_NAME}"
    ET.SubElement(geom, "size").text = f"{HEIGHTMAP_SIZE_M[0]} {HEIGHTMAP_SIZE_M[1]} {HEIGHTMAP_MAX_Z}"
    ET.SubElement(geom, "pos").text = "0 0 0"
    
    if textured:
        tex = ET.SubElement(geom, "texture")
        ET.SubElement(tex, "diffuse").text = f"{MODEL_URI}/materials/textures/dirt4.png"
        ET.SubElement(tex, "normal").text = f"{MODEL_URI}/materials/textures/dirt4.png"
        ET.SubElement(tex, "size").text = "1"
    
    return geom

def add_individual_trees_with_random_meshes(parent: ET.Element) -> None:
    """Generate individual tree models arranged in rows with random mesh selection."""
    sampler = HeightmapSampler(HEIGHTMAP_PNG, HEIGHTMAP_SIZE_M, HEIGHTMAP_MAX_Z)
    
    total_width = (PLANTS_PER_ROW - 1) * PLANT_SPACING
    x0 = -total_width / 2
    y0 = -((NUM_ROWS - 1) * ROW_SPACING) / 2
    
    tree_count = 0
    mesh_usage = {}  # Track which meshes are used
    
    for row in range(NUM_ROWS):
        y_world = y0 + row * ROW_SPACING
        
        for col in range(PLANTS_PER_ROW):
            x_world = x0 + col * PLANT_SPACING
            z_world = sampler.height_at(x_world, y_world)
            
            # Get random mesh for this tree
            random_mesh_path = get_random_mesh_path()
            mesh_name = random_mesh_path.rsplit("/", 1)[-1]
            
            # Track mesh usage for statistics
            if mesh_name not in mesh_usage:
                mesh_usage[mesh_name] = 0
            mesh_usage[mesh_name] += 1
            
            # Create individual tree model
            tree_model = ET.SubElement(parent, "model", name=f"tree_{row}_{col}")
            ET.SubElement(tree_model, "static").text = "true"
            ET.SubElement(tree_model, "pose").text = f"{x_world} {y_world} {z_world} {PLANT_ROLL} 0 0"
            
            # Add single link to this tree model with random mesh
            add_single_tree_link_with_mesh(tree_model, "tree_link", 0, 0, 0, random_mesh_path)
            
            tree_count += 1
    
    print(f"Generated {tree_count} individual tree models in {NUM_ROWS} rows")
    print(f"Used {len(mesh_usage)} different mesh variants:")
    for mesh, count in sorted(mesh_usage.items()):
        print(f"  {mesh}: {count} times")

def add_single_tree_link_with_mesh(parent: ET.Element, name: str, x: float, y: float, z: float, mesh_path: str) -> None:
    """Add a single tree link to a tree model with specified mesh."""
    link = ET.SubElement(parent, "link", name=name)
    
    # Visual element
    visual = ET.SubElement(link, "visual", name="visual")
    visual_geom = ET.SubElement(visual, "geometry")
    visual_mesh = ET.SubElement(visual_geom, "mesh")
    ET.SubElement(visual_mesh, "uri").text = mesh_path
    ET.SubElement(visual_mesh, "scale").text = f"{PLANT_SCALE[0]} {PLANT_SCALE[1]} {PLANT_SCALE[2]}"
    
    # Collision element
    collision = ET.SubElement(link, "collision", name="collision")
    coll_geom = ET.SubElement(collision, "geometry")
    coll_mesh = ET.SubElement(coll_geom, "mesh")
    ET.SubElement(coll_mesh, "uri").text = mesh_path
    ET.SubElement(coll_mesh, "scale").text = f"{PLANT_SCALE[0]} {PLANT_SCALE[1]} {PLANT_SCALE[2]}"
    
    # Pose relative to model (usually 0,0,0 since model pose handles positioning)
    ET.SubElement(link, "pose").text = f"{x} {y} {z} 0 0 0"
    
    # Add basic inertial properties (minimal since trees are static)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").text = "1.0"
    inertia = ET.SubElement(inertial, "inertia")
    ET.SubElement(inertia, "ixx").text = "0.1"
    ET.SubElement(inertia, "iyy").text = "0.1"
    ET.SubElement(inertia, "izz").text = "0.1"
    ET.SubElement(inertia, "ixy").text = "0"
    ET.SubElement(inertia, "ixz").text = "0"
    ET.SubElement(inertia, "iyz").text = "0"

# ─────────────────────────────── main ──────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an SDF world file with individual tree models using random meshes.")
    parser.add_argument("--out", "-o", type=str, 
                       default=str(PKG_DIR / "world" / "vineyard_for_result.sdf"),
                       help="Output SDF file (default: vineyard_for_result.sdf)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible results (optional)")
    
    args = parser.parse_args()
    
    # Set random seed if provided for reproducible results
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    root = build_world()
    indent(root)
    
    ET.ElementTree(root).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"✓ Wrote {args.out}")
    print(f"✓ Each tree uses a randomly selected mesh from Tree_1.obj to Tree_{PLANT_MESH_COUNT}.obj")

if __name__ == "__main__":
    main()

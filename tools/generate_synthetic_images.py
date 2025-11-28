import bpy
import math
import os

# ==================== CONFIGURATION ====================
# Output settings
OUTPUT_DIR = "/tmp/mirror_captures"  # Change this to your desired output directory
IMAGE_FORMAT = "PNG"
IMAGE_NAME_PREFIX = "capture"

# Mirror settings
MIRROR_RADIUS = 0.15  # 15cm mirror
MIRROR_SEGMENTS = 128  # High resolution

# Cylinder settings
CYLINDER_RADIUS = 0.5  # 50cm diameter
CYLINDER_HEIGHT = 1.5  # 150cm height

# Camera and capture settings
CAMERA_DISTANCE_FROM_MIRROR = 0.3  # 30cm below mirror
LENS_FOCAL_LENGTH = 35  # mm

# Vertical scanning parameters
START_HEIGHT = 0.1  # Start 10cm from bottom
END_HEIGHT = CYLINDER_HEIGHT - 0.2  # End 20cm from top
NUM_CAPTURES = 10  # Number of vertical positions

# Render settings
RENDER_WIDTH = 2048
RENDER_HEIGHT = 2048
SAMPLES = 256

# Defect generation (for testing)
ADD_RANDOM_DEFECTS = True
NUM_DEFECTS = 15

# ==================== SETUP SCENE ====================

def clear_scene():
    """Remove all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("✓ Scene cleared")

def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

def create_hemispherical_mirror(position=(0, 0, 0)):
    """Create the hemispherical mirror"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=MIRROR_SEGMENTS,
        ring_count=MIRROR_SEGMENTS // 2,
        radius=MIRROR_RADIUS,
        location=position
    )
    mirror = bpy.context.object
    mirror.name = "Mirror_Hemisphere"
    
    # Remove bottom half to create hemisphere
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    for vertex in mirror.data.vertices:
        if vertex.co.z < 0.001:
            vertex.select = True
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Flip normals for proper reflection
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create mirror material
    mirror_mat = bpy.data.materials.new(name="Mirror_Material")
    mirror_mat.use_nodes = True
    nodes = mirror_mat.node_tree.nodes
    nodes.clear()
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 0)
    
    node_glossy = nodes.new(type='ShaderNodeBsdfGlossy')
    node_glossy.location = (0, 0)
    node_glossy.inputs['Roughness'].default_value = 0.001
    
    mirror_mat.node_tree.links.new(node_glossy.outputs['BSDF'], node_output.inputs['Surface'])
    
    mirror.data.materials.append(mirror_mat)
    print(f"✓ Hemispherical mirror created (radius: {MIRROR_RADIUS}m)")
    
    return mirror

def create_camera(mirror_position=(0, 0, 0)):
    """Create inspection camera"""
    cam_z = mirror_position[2] - CAMERA_DISTANCE_FROM_MIRROR
    bpy.ops.object.camera_add(
        location=(mirror_position[0], mirror_position[1], cam_z),
        rotation=(0, 0, 0)
    )
    camera = bpy.context.object
    camera.name = "Inspection_Camera"
    camera.data.lens = LENS_FOCAL_LENGTH
    
    bpy.context.scene.camera = camera
    
    # Add camera light
    bpy.ops.object.light_add(type='POINT', location=(0, 0, cam_z - 0.05))
    camera_light = bpy.context.object
    camera_light.name = "Camera_Light"
    camera_light.data.energy = 100
    camera_light.parent = camera
    
    print(f"✓ Camera created at z={cam_z:.3f}m")
    
    return camera, camera_light

def create_cylinder():
    """Create engine cylinder"""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=128,
        radius=CYLINDER_RADIUS,
        depth=CYLINDER_HEIGHT,
        location=(0, 0, CYLINDER_HEIGHT / 2)
    )
    cylinder = bpy.context.object
    cylinder.name = "Engine_Cylinder"
    
    # Create cylinder material
    cyl_mat = bpy.data.materials.new(name="Cylinder_Material")
    cyl_mat.use_nodes = True
    nodes = cyl_mat.node_tree.nodes
    nodes.clear()
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (800, 0)
    
    node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled.location = (400, 0)
    node_principled.inputs['Base Color'].default_value = (0.4, 0.4, 0.45, 1.0)
    node_principled.inputs['Metallic'].default_value = 0.85
    node_principled.inputs['Roughness'].default_value = 0.25
    
    # Add noise texture for surface detail
    node_noise = nodes.new(type='ShaderNodeTexNoise')
    node_noise.location = (0, -100)
    node_noise.inputs['Scale'].default_value = 100.0
    node_noise.inputs['Detail'].default_value = 15.0
    
    node_colorramp = nodes.new(type='ShaderNodeValToRGB')
    node_colorramp.location = (200, -100)
    node_colorramp.color_ramp.elements[0].position = 0.4
    node_colorramp.color_ramp.elements[1].position = 0.6
    
    # Add coordinate system for proper texture mapping
    node_texcoord = nodes.new(type='ShaderNodeTexCoord')
    node_texcoord.location = (-200, -100)
    
    links = cyl_mat.node_tree.links
    links.new(node_texcoord.outputs['UV'], node_noise.inputs['Vector'])
    links.new(node_noise.outputs['Fac'], node_colorramp.inputs['Fac'])
    links.new(node_colorramp.outputs['Color'], node_principled.inputs['Roughness'])
    links.new(node_principled.outputs['BSDF'], node_output.inputs['Surface'])
    
    cylinder.data.materials.append(cyl_mat)
    
    print(f"✓ Cylinder created (radius: {CYLINDER_RADIUS}m, height: {CYLINDER_HEIGHT}m)")
    
    return cylinder

def create_defect_material():
    """Create material for defects (scratches, pits)"""
    defect_mat = bpy.data.materials.new(name="Defect_Material")
    defect_mat.use_nodes = True
    nodes = defect_mat.node_tree.nodes
    nodes.clear()
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs['Color'].default_value = (0.9, 0.2, 0.1, 1.0)
    node_emission.inputs['Strength'].default_value = 3.0
    
    defect_mat.node_tree.links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
    
    return defect_mat

def add_defects():
    """Add random defects to cylinder walls"""
    import random
    
    defect_mat = create_defect_material()
    
    for i in range(NUM_DEFECTS):
        # Random position on cylinder
        angle = random.uniform(0, 2 * math.pi)
        height = random.uniform(0.1, CYLINDER_HEIGHT - 0.1)
        
        x = (CYLINDER_RADIUS - 0.01) * math.cos(angle)
        y = (CYLINDER_RADIUS - 0.01) * math.sin(angle)
        z = height
        
        # Choose defect type
        defect_type = random.choice(['scratch_v', 'scratch_h', 'pit'])
        
        if defect_type == 'scratch_v':
            # Vertical scratch
            bpy.ops.mesh.primitive_cube_add(size=0.008, location=(x, y, z))
            defect = bpy.context.object
            defect.scale = (1, 0.3, random.uniform(10, 30))
        
        elif defect_type == 'scratch_h':
            # Horizontal scratch
            bpy.ops.mesh.primitive_cube_add(size=0.008, location=(x, y, z))
            defect = bpy.context.object
            defect.scale = (0.3, 1, random.uniform(10, 30))
            defect.rotation_euler = (0, 0, angle)
        
        else:  # pit
            # Small pit or dent
            bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.015, 0.035), location=(x, y, z))
            defect = bpy.context.object
            defect.scale = (random.uniform(0.3, 0.8), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
        
        defect.name = f"Defect_{i+1}"
        defect.data.materials.append(defect_mat)
    
    print(f"✓ Added {NUM_DEFECTS} random defects")

def create_ring_lights():
    """Create ring lights around the cylinder"""
    num_lights = 8
    
    for i in range(num_lights):
        angle = i * (2 * math.pi / num_lights)
        x = math.cos(angle) * (CYLINDER_RADIUS + 0.2)
        y = math.sin(angle) * (CYLINDER_RADIUS + 0.2)
        z = CYLINDER_HEIGHT / 2
        
        bpy.ops.object.light_add(type='AREA', location=(x, y, z))
        light = bpy.context.object
        light.name = f"Ring_Light_{i+1}"
        light.data.energy = 20
        light.data.size = 0.15
        
        # Point toward center
        direction = bpy.context.object.location
        light.rotation_euler = (0, math.atan2(direction.x, -direction.y), 0)
    
    print(f"✓ Created {num_lights} ring lights")

def setup_render_settings():
    """Configure render settings"""
    scene = bpy.context.scene
    
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    
    scene.render.image_settings.file_format = IMAGE_FORMAT
    scene.render.image_settings.color_mode = 'RGB'
    
    # Set background to dark
    world = bpy.data.worlds.get('World')
    if world:
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0.02, 0.02, 0.02, 1.0)
            bg_node.inputs['Strength'].default_value = 0.0
    
    print(f"✓ Render settings configured ({RENDER_WIDTH}x{RENDER_HEIGHT}, {SAMPLES} samples)")

def render_at_height(height, index):
    """Move mirror and camera to specific height and render"""
    mirror = bpy.data.objects.get("Mirror_Hemisphere")
    camera = bpy.data.objects.get("Inspection_Camera")
    camera_light = bpy.data.objects.get("Camera_Light")
    
    if not mirror or not camera:
        print("Error: Mirror or camera not found!")
        return
    
    # Move mirror to new height
    mirror.location.z = height
    
    # Move camera (maintaining distance from mirror)
    camera.location.z = height - CAMERA_DISTANCE_FROM_MIRROR
    
    # Update scene
    bpy.context.view_layer.update()
    
    # Set output path
    output_path = os.path.join(OUTPUT_DIR, f"{IMAGE_NAME_PREFIX}_{index:03d}_h{height:.3f}.png")
    bpy.context.scene.render.filepath = output_path
    
    # Render
    print(f"  Rendering at height {height:.3f}m... ", end='', flush=True)
    bpy.ops.render.render(write_still=True)
    print(f"✓ Saved: {output_path}")

def generate_capture_positions():
    """Generate list of vertical positions for captures"""
    positions = []
    step = (END_HEIGHT - START_HEIGHT) / (NUM_CAPTURES - 1) if NUM_CAPTURES > 1 else 0
    
    for i in range(NUM_CAPTURES):
        height = START_HEIGHT + (i * step)
        positions.append(height)
    
    return positions

# ==================== MAIN EXECUTION ====================

def main():
    print("="*70)
    print("MIRROR BALL CAMERA - SYNTHETIC IMAGE GENERATION")
    print("="*70)
    
    # Setup
    clear_scene()
    create_output_directory()
    
    # Create scene components
    create_cylinder()
    
    if ADD_RANDOM_DEFECTS:
        add_defects()
    
    create_ring_lights()
    
    # Create mirror at initial position
    mirror = create_hemispherical_mirror(position=(0, 0, START_HEIGHT))
    
    # Create camera
    camera, camera_light = create_camera(mirror_position=(0, 0, START_HEIGHT))
    
    # Setup rendering
    setup_render_settings()
    
    # Generate captures at different heights
    positions = generate_capture_positions()
    
    print("\n" + "="*70)
    print(f"CAPTURING {NUM_CAPTURES} IMAGES")
    print(f"Height range: {START_HEIGHT:.3f}m to {END_HEIGHT:.3f}m")
    print("="*70 + "\n")
    
    for idx, height in enumerate(positions):
        render_at_height(height, idx)
    
    print("\n" + "="*70)
    print("✓ ALL CAPTURES COMPLETE")
    print("="*70)
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print(f"Total captures: {NUM_CAPTURES}")
    print(f"\nNext step: Run the unwrapping script on these images")
    print("  python unwrap_and_crop.py")
    print("="*70)

if __name__ == "__main__":
    main()

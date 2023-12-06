---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: cv_torch
    language: python
    name: cv_torch
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
import drjit as dr

#mi.set_variant('cuda_ad_spectral')
mi.set_variant('cuda_ad_rgb')
print("Is spectral", mi.is_spectral)

from mitsuba import set_log_level,LogLevel
set_log_level(LogLevel.Warn)
```

```python
grid_teeth_wta_ply_path ='assets/grid_wta.ply'
grid_teeth_tvel2_ply_path = 'assets/grid_tvel2.ply'
```

```python
def render_scene(scene_dict):
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)
    return mi.util.convert_to_bitmap(img)
```

```python
import mypyply

teeth_paths = [grid_teeth_wta_ply_path,grid_teeth_tvel2_ply_path]
grid_teeth_types_mesh_data = [mypyply.read_ply(p) for p in teeth_paths]
```

```python
import scene_definition

frame_width = 400
frame_height = 400
camera_x = 0
camera_distance = 2
camera_z = 0
fov = 20
samples_per_pass = 8
cam_light_intensity = 47100
light_offset = 228

sensor = scene_definition.get_ring_camera(
    frame_width,
    frame_height,
    camera_x,
    camera_distance, 
    camera_z, 
    fov,
    spp=samples_per_pass
)
left_light = scene_definition.get_ring_light(camera_x+light_offset//2, camera_distance,camera_z, cam_light_intensity)
right_light = scene_definition.get_ring_light(camera_x-light_offset//2, camera_distance,camera_z, cam_light_intensity)
    
scene_dict = {
    "type" : "scene",
    # "rods_material":rods_material,
    # "grids_material":grids_material,
    "myintegrator" : {
        "type" : "path",
 #       "type":"volpath",
        "samples_per_pass":samples_per_pass,
        "max_depth": 8,
    },
    "light":{
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': 1.0,
        }
    },
    # "left_light": left_light,
    # "right_light": right_light,
    
    "sensor":sensor,
    
    "test_obj":{
        'type': 'ply',
        #'filename': grid_teeth_tvel2_ply_path,
        'filename': grid_teeth_wta_ply_path,
        'flip_normals': True
    },
    
}

img = render_scene(scene_dict)
plt.imshow(img)
```

```python

video_config = {
    "width": 720,
    "height": 576,
}

measurements = {
    "rod_width_mm" : 9.1, 
    "gap_between_rods_width_mm": 3.65,
    
    #based on diagram, not blueprint -> approximate values
    "camera_distance_mm" : 430,
    "light_offset_mm": 228    
}

fuel_material={
    "rods":{
        "material_alpha_u": .05,
        "material_alpha_v": .12,
        "diffuse_gray_intensity": 0.25,
        "zircon_to_bsdf_blend_factor": .25,
    },
    "grids":{
        "material_alpha_u": .06,
        "material_alpha_v": .06,
        "diffuse_gray_intensity": 0.25,
        "metal_to_bsdf_blend_factor": .25, #.0.25
    }    
}

grid_detection = {
    "mods": [
      {
        "name": "mod1",
        "spacing_mm": [485,510,510,510,510,510,510,261],
        "grid_heights_mm": [45, 65, 65, 65, 65, 65, 65],
      },
      {
        "name": "mod2",
        "spacing_mm": [237.5, 255,120,220,180,160,240,100,340,340,340,340,340,340,316],
        "grid_heights_mm": [45, 45, 15, 45, 15, 45, 15, 45, 45, 45, 45, 45, 45, 45],
        
      },
      {
        "name": "mod3",
        "spacing_mm": [200, 255, 340, 340, 340, 340, 340, 340, 340, 340, 340, 240],
        "grid_heights_mm": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], 
      }
    ]
  }

config = {
    "measurements":measurements,
    "fuel_material": fuel_material,
    "grid_detection": grid_detection
    
}
```

```python
import scene_definition

rods_material = config['fuel_material']['rods']
rods_material_alpha_u = rods_material['material_alpha_u']
rods_material_alpha_v = rods_material['material_alpha_v']
rods_gray_intensity =rods_material['diffuse_gray_intensity']
rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']

camera_distance= config['measurements']['camera_distance_mm'] #given by blueprints (approx)
light_offset = config['measurements']['light_offset_mm'] #given by blueprints (approx)

samples_per_pass = 32
#samples_per_pass = 128
cam_light_intensity = 47100
rod_count = 11

rods_bdsf = scene_definition.get_rods_bsdf(
    rods_gray_intensity, 
    rods_bsdf_blend_factor, 
    rods_material_alpha_u,
    rods_material_alpha_v
)


nfa_curve,rods_group = scene_definition.generate_rods_group(
    config, 
    #has_rod_divergence= has_rod_divergence, 
    bsdf=rods_bdsf,
    rod_count = rod_count
)
rods_dict = dict((f"rod_{i}",rod) for i,rod in enumerate(rods_group))
```

```python
grid_material = config['fuel_material']['grids']
alpha_u = grid_material['material_alpha_u']
alpha_v = grid_material['material_alpha_v']
gray_intensity =grid_material['diffuse_gray_intensity']
bsdf_blend_factor = grid_material['metal_to_bsdf_blend_factor']
bsdf_blend_factor = .6
grid_material = scene_definition.get_grid_bsdf(gray_intensity,bsdf_blend_factor,alpha_u,alpha_v)
```

```python
# import drjit as dr

# N = 100
# frequency = 12.0
# amplitude = 0.4

# # Generate the vertex positions
# theta = dr.linspace(mi.Float, 0.0, dr.two_pi, N)
# x, y = dr.sincos(theta)
# z = amplitude * dr.sin(theta * frequency)
# vertex_pos = mi.Point3f(x, y, z)

# # Move the last vertex to the center
# vertex_pos[dr.eq(dr.arange(mi.UInt32, N), N - 1)] = 0.0

# # Generate the face indices
# idx = dr.arange(mi.UInt32, N - 1)
# face_indices = mi.Vector3u(N - 1, (idx + 1) % (N - 2), idx % (N - 2))

# mesh = mi.Mesh(
#     "wavydisk",
#     vertex_count=N,
#     face_count=N - 1,
#     has_vertex_normals=False,
#     has_vertex_texcoords=False,
# )

```

```python
import pathlib
import tempfile
```

```python
def grid_dist_from_center(
    visible_rod_count, 
    rod_width_mm,
    gap_between_rods_width_mm
):
    rod_w_gap_mm = rod_width_mm + gap_between_rods_width_mm
    total_mm = rod_w_gap_mm * visible_rod_count - rod_width_mm
    
    return np.sqrt(total_mm**2-(total_mm/2)**2) + 3


```

```python

grid_detection = {
    "mods": [
      {
        "name": "mod1",
        "spacing_mm": [485,510,510,510,510,510,510,261],
        "grid_heights_mm": [45, 65, 65, 65, 65, 65, 65],
      },
      {
        "name": "mod2",
        "spacing_mm": [237.5, 255,120,220,180,160,240,100,340,340,340,340,340,340,316],
        "grid_heights_mm": [45, 45, 15, 45, 15, 45, 15, 45, 45, 45, 45, 45, 45, 45],
        
      },
      {
        "name": "mod3",
        "spacing_mm": [200, 255, 340, 340, 340, 340, 340, 340, 340, 340, 340, 240],
        "grid_heights_mm": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], 
      }
    ]
  }
grid_mod = grid_detection['mods'][0]
```

```python

```

```python
rod_height = np.sum(grid_mod['spacing_mm'])  
camera_z_ration = 1-np.abs(camera_z/rod_height)
#camera_x,_,_ = nfa_curve.evaluate_multi(np.array([camera_z_ration]))

samples = np.cumsum(grid_mod['spacing_mm'])/rod_height
sampled = nfa_curve.evaluate_multi(
    samples
)

pos_x,pos_y,pos_z = sampled[:,3]
pos_x,pos_y,pos_z
```

```python
import grid_mesh as gm    

mesh_data = grid_teeth_types_mesh_data[1]

points = mesh_data['points']
face = mesh_data['mesh']
p_x = points.x
p_y = points.y
p_z = points.z

v_1 = face.v1
v_2 = face.v2
v_3 = face.v3

p_x,p_y,p_z,v_1,v_2,v_3 = gm.mirror_down(p_x,p_y,p_z,v_1,v_2,v_3)
p_x,p_y,p_z,v_1,v_2,v_3 = gm.array_left(p_x,p_y,p_z,v_1,v_2,v_3,n=rod_count-2)



distance = -grid_dist_from_center(11,9.1,3.65)  #mm
p_x,p_y,p_z,v_1,v_2,v_3 = gm.array_hexagon(distance,p_x,p_y,p_z,v_1,v_2,v_3)

vertex_pos = mi.Point3f(
    np.float32(p_x),
    np.float32(p_y),
    np.float32(p_z)
)
face_indices = mi.Vector3u([
    np.float32(v_1),
    np.float32(v_2),
    np.float32(v_3)
])
```

```python
from matplotlib import pyplot as plt


cam_distance = 600
cam_z = -1000
cam_z = -pos_z

mesh = mi.Mesh(
    "grid_teeth",
    vertex_count=len(vertex_pos[0]),
    face_count=len(face_indices[0]),
    has_vertex_normals=False,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
mesh_params["faces"] = dr.ravel(face_indices)

temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
temp_dir.mkdir(parents=True,exist_ok=True)
temp_path = str(temp_dir/'grid.ply')
mesh.write_ply(temp_path)


scene_dict = {
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {"type": "constant"},
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, -cam_distance, cam_z + 50], 
            target=[0, 0, cam_z], 
            up=[0, 0, 1]
        ),
    },
    "test_grid": {        
        'type': 'ply',
        'filename': temp_path,
        #'flip_normals': True
        "material":grid_material,
        "to_world" :mi.ScalarTransform4f.translate([pos_x,pos_y,cam_z]),
        
        # "material": {
        #     'type': 'diffuse',
        #     'reflectance': {
        #         'type': 'rgb',
        #         'value': [1, 0, 1]
        #     }
        # }
    }
    
}

scene_dict.update(rods_dict)

img = render_scene(scene_dict)

plt.imshow(img)
```

```python

```

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
import mypyply
import pathlib
assets_path = pathlib.Path('assets')
grid_teeth_models_paths = list(assets_path.glob("grid_*.ply"))
grid_teeth_types_mesh_data = [mypyply.read_ply(p) for p in grid_teeth_models_paths]
grid_teeth_models_paths
```

```python
# def render_scene(scene_dict):
#     scene = mi.load_dict(scene_dict)
#     img = mi.render(scene)
#     return mi.util.convert_to_bitmap(img)

def render_scene(scene_dict):
    
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)
    bitmap = mi.Bitmap(img).convert(
        mi.Bitmap.PixelFormat.RGB, 
        mi.Struct.Type.Float32, 
        srgb_gamma=True
    )
    
    arr_bitmap= np.array(bitmap)
    # clipping
    # arr_bitmap[arr_bitmap>1] = 1
    return arr_bitmap
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


nfa_curve,curves, rods_group = scene_definition.generate_rods_group(
    config, 
    #has_rod_divergence= has_rod_divergence, 
    bsdf=rods_bdsf,
    rod_count = rod_count
)
rods_dict = dict((f"rod_{i}",rod) for i,rod in enumerate(rods_group))
```

```python
grid_material_config = config['fuel_material']['grids']
alpha_u = grid_material_config['material_alpha_u']
alpha_v = grid_material_config['material_alpha_v']
gray_intensity =grid_material_config['diffuse_gray_intensity']
bsdf_blend_factor = grid_material_config['metal_to_bsdf_blend_factor']
bsdf_blend_factor = .6
grid_material = scene_definition.get_grid_bsdf(gray_intensity,bsdf_blend_factor,alpha_u,alpha_v)
```

```python

```

```python
plt.plot(rod_tops[:,0],rod_tops[:,1],'.')
```

```python
import pathlib
import tempfile
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

```

```python
import scene_definition
tips_scenes = scene_definition.get_tips(curves, config)
```

```python
from matplotlib import pyplot as plt

cam_light_intensity = 100_000 *5
camera_distance = 100 * 4


camera_z = -pos_z
camera_z = 200
camera_x = pos_x

mesh_data = grid_teeth_types_mesh_data[2]
mesh = gm.create_grid_mesh_from_tooth(mesh_data,rod_count)

temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
temp_dir.mkdir(parents=True,exist_ok=True)
temp_path = str(temp_dir/'grid.ply')
mesh.write_ply(temp_path)

samples_per_pass = 256
sensor = scene_definition.get_ring_camera(
    frame_width,
    frame_height,
    camera_x,
    camera_distance, 
    camera_z, 
    fov,
    spp=samples_per_pass
)

sensor_my = {
    "type": "perspective",
    "to_world": mi.ScalarTransform4f.look_at(
        origin=[0, camera_distance, camera_z], target=[0, 0, 0], up=[0, 0, 1]
    ),
}

sensor = sensor_my
left_light = scene_definition.get_ring_light(camera_x+light_offset//2, camera_distance,camera_z, cam_light_intensity)
right_light = scene_definition.get_ring_light(camera_x-light_offset//2, camera_distance,camera_z, cam_light_intensity)

scene_dict = {
    "type": "scene",
    "integrator": {"type": "path"},
    "sensor":sensor,
    #"light": {"type": "constant"},
    "left_light": left_light,
    "right_light": right_light,
#     "test_grid": {        
#         'type': 'ply',
#         'filename': temp_path,
#         #'filename': 'assets/grid_wta_dented.ply',
#         #'flip_normals': True
        
#         "to_world" :mi.ScalarTransform4f.translate([pos_x,pos_y,camera_z]),
        
#         "material":grid_material,
#         # "material": {
#         #     'type': 'diffuse',
#         #     'reflectance': {
#         #         'type': 'rgb',
#         #         'value': [1, 0, 1]
#         #     }
#         # }
#     }
    
}



scene_dict.update(rods_dict)
scene_dict.update(tips_scenes)

img = render_scene(scene_dict)

plt.imshow(img,vmax = 1)
```

```python

```

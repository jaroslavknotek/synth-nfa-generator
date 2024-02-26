---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
from tqdm.auto import tqdm
import time
```

```python
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt

#mi.set_variant('cuda_ad_spectral')
mi.set_variant('cuda_ad_rgb')
print("Is spectral", mi.is_spectral)

from mitsuba import set_log_level,LogLevel
set_log_level(LogLevel.Warn)
```

```python
import logging 
logging.basicConfig()
logger = logging.getLogger("synt")
logger.setLevel(logging.DEBUG)
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
import oxides 

grid_mod = config['grid_detection']["mods"][2]
rod_height = np.sum(grid_mod['spacing_mm'])  
    
cylinder_radius = config['measurements']['rod_width_mm']/2
px_per_unit = 5
w = int(cylinder_radius * np.pi * px_per_unit)
h = int(rod_height * px_per_unit)
n = 10
h,w

#spots_texture = oxides.get_spots_texture(h,w,n,spot_radius_mean_px=50,spot_radius_var_px=15,seed = 124)
```

```python
seed = 124
n_textures = 20
spots_textures = [oxides.get_spots_texture(h,w,n,spot_radius_mean_px=30,spot_radius_var_px=15,seed = seed + i) for i in range(n_textures)]
spots_texture = np.hstack(spots_textures)
```

```python
plt.figure(figsize = (20,100))
plt.imshow(spots_texture)
```

```python
# pip install git+https://github.com/pvigier/perlin-numpy

import perlin_numpy

#noise = perlin_numpy.generate_fractal_noise_2d((256, 512), (1, 1), 1)
grid_ratio = np.array((1,4))
shape = np.array((1,6))
res_px =  128
noise = perlin_numpy.generate_fractal_noise_2d(
    shape *res_px * grid_ratio, 
    grid_ratio * shape *[2,1], 
    1,
    tileable = (False,True)
)
noise = (noise - np.min(noise))/(np.max(noise) - np.min(noise))
plt.imshow(np.tile(noise,(2,2)), cmap='gray', interpolation='lanczos')

import imageio
imageio.imwrite("test_bump.png",np.uint8( noise *255))
```

```python
import oxides
import rods_hexagon
import scene_definition
import grid_mesh as gm


def get_nfa_scene_dict(config, fov, frame_height, frame_width, camera_z, z_displacement, has_rod_divergence = False,aov= False):
    
    #get_rods_bsdf(gray_intensity,bsdf_blend_factor,material_alpha_u,material_alpha_v,  spectral_params = None)
    
    rods_material = config['fuel_material']['rods']
    rods_material_alpha_u = rods_material['material_alpha_u']
    rods_material_alpha_v = rods_material['material_alpha_v']
    rods_gray_intensity =rods_material['diffuse_gray_intensity']
    rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']
    
    camera_distance= config['measurements']['camera_distance_mm'] #given by blueprints (approx)
    light_offset = config['measurements']['light_offset_mm'] #given by blueprints (approx)
    
    samples_per_pass = 32
    #samples_per_pass = 128
    #cam_light_intensity = 47100
    cam_light_intensity = 100
    rods_bdsf = scene_definition.get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
    
    nfa_curve,curves,rods_group = scene_definition.generate_rods_group(
        config, 
        #has_rod_divergence= has_rod_divergence, 
        bsdf=rods_bdsf,
        z_displacement = z_displacement
    )

    grid_mod = config['grid_detection']["mods"][2]
    rod_height = np.sum(grid_mod['spacing_mm'])  
    camera_z_ration = 1-np.abs(camera_z/rod_height)
    camera_x,_,_ = nfa_curve.evaluate_multi(np.array([camera_z_ration]))
       
    left_light = scene_definition.get_ring_light(camera_x+light_offset//2, camera_distance,camera_z, cam_light_intensity)
    right_light = scene_definition.get_ring_light(camera_x-light_offset//2, camera_distance,camera_z, cam_light_intensity)
    sensor = scene_definition.get_ring_camera(frame_width,frame_height,camera_x,camera_distance, camera_z, fov,spp=samples_per_pass)
    
    integrator = {
        "type":"volpath",
        "max_depth": 8,
    }
    if aov:
        integrator = {
            'type': 'aov',
            #'aovs': 'dd.y:depth,nn:sh_normal',
            'aovs': 'uvx:uv,uvy:uv',
            'my_image': {
                'type': 'path',
            }
        }
        
    scene_dict = {
        "type" : "scene",
        # "rods_material":rods_material,
        # "grids_material":grids_material,
        "myintegrator" : integrator,    
        "left_light": left_light,
        "right_light": right_light,
        "sensor":sensor,
        # "constant":{
        #     'type': 'constant',
        #     'radiance': {
        #         'type': 'rgb',
        #         'value': .3,
        #     }
        # }
    }
    
    
    
    tips_scenes = scene_definition.get_tips(curves, config)
#     rand_high =  np.random.random(size=len(rods_dict))
    
    for rh,(k,v) in zip(z_displacement, tips_scenes.items()):    
        v["to_world"] = v["to_world"].translate([0,0,rh])
    
    # TEXTURE
    logger.info("Generating Texture")
    
    cylinder_radius = config['measurements']['rod_width_mm']/2
    
    px_per_unit = 5
    w = int(cylinder_radius * np.pi * px_per_unit)
    h = int(rod_height)
    n = 400
    
    #spots_texture = oxides.get_spots_texture(h,w,n,spot_radius_mean_px=80,spot_radius_var_px=50)
    
    spots = (1-spots_texture)/10 + .75
    oxide_texture = {
        'type': 'bitmap',
        'data': np.expand_dims(spots,axis=2),
        'raw':True,
        #'bitmap': oxides,
        #'wrap_mode': 'clamp',
        #'to_uv':mi.ScalarTransform4f.translate([0,0,0]).rotate([0,0,1],0).scale([1,1,1])
    }
    
    for rod in rods_group:
        rod['material'] = {
            'type': 'blendbsdf',
            "weight": oxide_texture,
            'bsdf_0': scene_definition.get_gray_diffuse(.95),
            'bsdf_1': rod['material']
        }
        
    rods_dict = dict((f"rod_{i}",rod) for i,rod in enumerate(rods_group))
    scene_dict.update(rods_dict)
    scene_dict.update(tips_scenes)
    
    mod3_spacing_mm = grid_mod['spacing_mm']
    logger.info("Adding Grids")
    
    add_grid(scene_dict,nfa_curve,rod_height, np.cumsum(mod3_spacing_mm),config)

    
    scene_dict['containment'] = {
        "type":"cylinder",
        "flip_normals":True,
        "p0":[0,0,-5000],
        "p1":[0,0,500],
        "radius":500,
        #"filename":rod_ply_path,
        "mat":{
            'type': 'diffuse',
            'reflectance': {
                'type': 'checkerboard',
                'to_uv': mi.ScalarTransform4f.scale(100),
                'color0':.3,
                'color1':{
                    'type': 'srgb',
                    'color': [.6, .6, .6]
                    #'color': [1, 0, 1]
                }
            }
        }
    }
    
    return scene_dict,nfa_curve

def render_scene(scene_dict,spp = 8,frame_random_state = None, include_raw = False):
    logger.debug("Preparing Scene")
    scene = mi.load_dict(scene_dict)
    if frame_random_state is None:
        seed = 0
    else:
        seed = frame_random_state.randint(0,high =  2**31 - 1)
        
    logger.debug("Rendering Scene")
    img = mi.render(scene,spp = spp,seed = seed)
    bitmap = mi.Bitmap(img[:,:,:4]).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
    
    arr_bitmap= np.array(bitmap)
    # clipping
    # arr_bitmap[arr_bitmap>1] = 1
    if include_raw:
        return arr_bitmap, img
    else:
        return arr_bitmap


import mypyply
import pathlib


assets_path = pathlib.Path('assets')
grid_teeth_models_paths = list(assets_path.glob("*.ply"))
grid_teeth_types_mesh_data = [mypyply.read_ply(p) for p in grid_teeth_models_paths]
grid_teeth_models_paths
mesh_data = grid_teeth_types_mesh_data[1]

def add_grid(scene_dict, nfa_curve, nfa_height, grid_positions, config):    
    rod_count = 11
    mesh = gm.create_grid_mesh_from_tooth(mesh_data,rod_count)

    #temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    temp_dir = pathlib.Path(f'/home/knotek/tmp/synt/grid_{nfa_height}')
    temp_dir.mkdir(parents=True,exist_ok=True)
    temp_path = str(temp_dir/'grid.ply')
    mesh.write_ply(temp_path)
    
    
    grid_relative_height = 1-np.abs(grid_positions/nfa_height)
    grid_curved_pos = nfa_curve.evaluate_multi(grid_relative_height)
    grid_xs = grid_curved_pos[0]
    
    
    grid_x_above,_,grid_z_above = nfa_curve.evaluate_multi(grid_relative_height - .01)
    grid_x_below,_,grid_z_below = nfa_curve.evaluate_multi(grid_relative_height + .01)
    
    dist_x = grid_x_above - grid_x_below
    dist_y = grid_z_above - grid_z_below
    
    angles = np.rad2deg(np.arctan(dist_x/dist_y))
    
    grid_material_cfg = config['fuel_material']['grids']
    alpha_u = grid_material_cfg['material_alpha_u']
    alpha_v = grid_material_cfg['material_alpha_v']
    gray_intensity =grid_material_cfg['diffuse_gray_intensity']
    bsdf_blend_factor = grid_material_cfg['metal_to_bsdf_blend_factor']
    
    bsdf_blend_factor = .6
    
    grid_material = grid_material_cfg
    # grid_material = {
    #     'type': 'diffuse',
    #     'reflectance': {
    #         'type': 'rgb',
    #         'value': [1, 0, 1]
    #     }
    # }
    # grid_material = {
    #     'type': 'area',
    #     'radiance': {
    #         'type': 'rgb',
    #         #'value': 1.0,
    #         'value': [1, 0, 1]
    #     }
    # }

    # grid_material = scene_definition.get_grid_bsdf(
    #     gray_intensity,
    #     bsdf_blend_factor,
    #     alpha_u,
    #     alpha_v
    # )
    
    for (grid_z,grid_x,a) in zip(grid_positions,grid_xs,angles):    
        grid_material = scene_definition.get_grid_bsdf(gray_intensity,bsdf_blend_factor,alpha_u,alpha_v)
        gm_id = grid_material['id']
        # HACK
        grid_material['id'] = f"{gm_id}_{grid_z}"
        
        bump = {
            'type': 'bumpmap',
            'arbitrary': {
                'type':'bitmap',
                'raw': True,
                'data': np.expand_dims(np.roll(noise,(0,int(grid_z))),axis=2)
            },
            'material': grid_material
        }
        
        grid = {
            "type": "ply",
            "filename": temp_path,
            "to_world": mi.ScalarTransform4f.translate([grid_x,0,-grid_z]),
            #"material":grid_material
            "material":bump
        }
        scene_dict[f'grid_obj_{grid_z}'] = grid

    

fov,crop_shift = 5.85,0

fov = 28
width=video_config['width']
height=video_config['height']

#TODo bump mapa

# Video frame scrop
# fov,height,width,crop_shift = 22,video_config['height'],video_config['width'],59
# height,width = 4000,ref_img.shape[1]


camera_z= -2830

render_intensity_factor = 2

z_displacement = (np.random.random(size = 331) * 5)[:,None]
scene_dict,nfa_curve = get_nfa_scene_dict(config,fov,height,width,camera_z,z_displacement,has_rod_divergence=True,aov=False)


# scene_dict['constant'] = {
#             'type': 'constant',
#             'radiance': {
#                 'type': 'rgb',
#                 'value': .5,
#             }
#         }

# TODO
#https://github.com/mitsuba-renderer/mitsuba-tutorials/blob/09d49632bde18c5a414f870e356f1c7545e670b5/how_to_guides/image_io_and_manipulation.ipynb

bitmap = render_scene(scene_dict,spp = 128,include_raw=False)
plt.imshow(bitmap)
plt.show()

# plt.imshow(raw[:,:,4])
# plt.show()
# plt.imshow(raw[:,:,5])
```

```python
assert False
```

```python
scene_dict.keys()
```

```python
n = 4000
x = np.linspace(0,1,n)
x,y,z = nfa_curve.evaluate_multi(x)

plt.plot(x,z)
```

```python
xyz = np.stack([x,y,z])
#np.savetxt('curve.npz',xyz)
```

```python
assert False
```

```python
import imageio
import pandas as pd
import imageio
import cv2
import time 

from tqdm.auto import tqdm

from datetime import datetime

def get_frames(
    frames_count,
    frame_width,
    frame_height,
    z_displacement = None,
    is_top_down = True,
    has_rod_divergence = False,
    frames_from = 0,
    seed = 456):
    
    # Video frame scrop
    frame_random_state = np.random.RandomState(seed)
    fov = 28
    
    # TODO calc the param automatically
    render_intensity_factor = 2 # scales intensities so the output looks as the ref video 
    
    mod3_spacing_mm = grid_detection["mods"][-1]['spacing_mm']
    grid_positions = np.cumsum(mod3_spacing_mm)

    # TODO find real value
    bottom_segment_mm = 100
    from_mm,to_mm = 0,grid_positions[-1] + bottom_segment_mm
    cam_zs = np.linspace(from_mm,to_mm,frames_count).astype(int)
    
    cam_zs = cam_zs[frames_from:]
    
    if not is_top_down:
        cam_zs = np.flip(cam_zs)
    
    for frame_num,cam_z in zip(range(frames_from,frames_count),cam_zs):
        camera_z = -cam_z
        logger.debug(f"Creating scene {frame_num=} {camera_z=}")
        scene_dict,_ = get_nfa_scene_dict(
            config,
            fov,
            frame_height,
            frame_width,
            camera_z,
            z_displacement,
            has_rod_divergence=has_rod_divergence
        )

        frame = render_scene(
            scene_dict,
            spp = 128,
            frame_random_state=frame_random_state
        )
        logger.debug(f"Yielding frame {frame_num=} {camera_z=}")
        yield frame_num, frame
        

# def create_video(
#     vid_dir,
#     frame_number,
#     frame_width,
#     frame_height,
#     fps = 25,
#     has_rod_divergence=False,
#     seed = 456):
#     # disabled because I don't have the right coded on my local
#     #fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     vid_path = vid_dir/"video.mp4"
    
#     top_down = seed%2==0
#     frames = get_frames(
#         frame_number,
#         frame_width,
#         frame_height,
#         is_top_down = top_down,
#         has_rod_divergence=has_rod_divergence,
#         seed = seed)
    
#     frames = tqdm(frames,desc ='generating frames',total=frame_number)
#     writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (frame_width,frame_height))
#     for i,(_,frame) in enumerate(frames):
#         #f = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         #print(datetime.now(), 'processed', f"{i/frame_number}")
#         f = np.uint8(frame *255)
#         writer.write(f)
        
#         if i % 50 ==0:
#             imageio.imwrite(vid_path.parent/f"frame_{i:04}.png", f)
        
#     sips_df = _get_sips(frame_number, top_down)
#     sips_df.to_csv(vid_dir/"sips.csv", index=False,header=True)
    
    

#     writer.release()


def create_video(
    vid_dir,
    frames_number,
    frame_width,
    frame_height,
    top_down = True,
    fps = 25,
    frames_from = 0,
    has_rod_divergence=False,
    seed = 456):
    # disabled because I don't have the right coded on my local
    #fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print(f"writing to {vid_dir}")
    
    
    np.random.seed = seed
    z_displacement = (np.random.random(size =331) * 5)[:,None]
    frames = get_frames(
        frames_number,
        frame_width,
        frame_height,
        z_displacement = z_displacement,
        is_top_down = top_down,
        frames_from = frames_from,
        has_rod_divergence=has_rod_divergence,
        seed = seed)
    frames_total = frames_number-frames_from
    frames = tqdm(frames,desc ='generating frames',total=frames_total)
    for i,frame in frames:
        f = np.uint8(frame *255)
        frame_name =f"frame_{i:05}.png"
        logger.info(f"saving frame {frame_name=}")
        imageio.imwrite(vid_dir/frame_name, f)

import os
import pathlib
import shutil

output_video_dir = pathlib.Path("output")
os.makedirs(output_video_dir, exist_ok = True)
    
dt = datetime.fromtimestamp(time.time())
timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S').replace(':','-')
#timestamp = '2023-12-18T10-22-34'

from tqdm.contrib.logging import logging_redirect_tqdm
with logging_redirect_tqdm():
    for i in range(0,1):
        assert i<6
        video_dir = output_video_dir/timestamp/"JKSY"/f"F{i+1}"
        video_dir.mkdir(parents=True,exist_ok=True)

        fps = 25
        frames_from = 0 #+ 1269 +2642
        frames_count = 100
        create_video(
            video_dir,
            frames_count,
            video_config['width'],
            video_config['height'],
            frames_from = frames_from, 
            fps=fps,
            top_down = (i % 2==0),
            #has_rod_divergence=True,
            seed = i+123)

print('done')
```

```python
exit()
```

```python
mod3_spacing_mm = grid_detection["mods"][-1]['spacing_mm']
grid_positions = np.cumsum(mod3_spacing_mm)
grid_positions[-1]
```

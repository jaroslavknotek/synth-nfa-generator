---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: palivo_general
    language: python
    name: palivo_general
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
#mi.set_variant('scalar_rgb')
mi.set_variant('scalar_spectral')
#mi.set_variant('cuda_ad_rgb')
print("Is spectral", mi.is_spectral)
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

```

```python
import warnings
import h5py

def load_ref_rods_image(ref_fuel_matrix_path, band_num = None):
    
    
    try:
        with h5py.File(ref_fuel_matrix_path, "r") as matrix_object:
            bands_total = matrix_object["bands"][()]
            band_num_resolved = band_num if band_num else bands_total//2
            ref_img_full = matrix_object[f'band-{band_num_resolved:03}'][:].astype(float)


        t,b,l,r, = 650,800, 100,700

        plt.imshow(ref_img_full[:1200])
        plt.axvline(100,)
        plt.axvline(700)
        plt.show()

        return ref_img_full[t:b,l:r] /255
    except FileNotFoundError as fe:
        warnings.warn("Couldn't find file:{ref_fuel_matrix_path}")
        return np.zeros((100,100))
    

ref_fuel_matrix = "/disk/knotek/video_matrices/1GO22-WTA6-F1-matrix.hdf5"
ref_img = load_ref_rods_image(ref_fuel_matrix)

print("Taking 20th band should take stuff 'above' camera. However, this assumption doesn't hold for videos with bottom-up camera movemet. \nAlways check the images if you see the header and notice the reflections on the grid")
ref_img_band_10 = load_ref_rods_image(ref_fuel_matrix,20)

print('shape', ref_img.shape)
plt.imshow(ref_img,cmap='gray',vmin=0,vmax=1)
plt.show()

plt.imshow(ref_img_band_10,cmap='gray',vmin=0,vmax=1)
plt.show()
```

```python

```

```python
plt.figure(figsize = (16,100))
plt.imshow(ref_img)
```

```python
import rods_hexagon
import scene_definition



def get_nfa_scene_dict(config, fov, frame_height, frame_width, camera_z, has_rod_divergence = False):
    
    #get_rods_bsdf(gray_intensity,bsdf_blend_factor,material_alpha_u,material_alpha_v,  spectral_params = None)
    
    rods_material = config['fuel_material']['rods']
    rods_material_alpha_u = rods_material['material_alpha_u']
    rods_material_alpha_v = rods_material['material_alpha_v']
    rods_gray_intensity =rods_material['diffuse_gray_intensity']
    rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']
    
    camera_distance= config['measurements']['camera_distance_mm'] #given by blueprints (approx)
    light_offset = config['measurements']['light_offset_mm'] #given by blueprints (approx)
    
    samples_per_pass = 64
    samples_per_pass = 32
    cam_light_intensity = 47100
    camera_x= 0    
    
    rods_bdsf = scene_definition.get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
    rods_group = scene_definition.generate_rods_group(config, has_rod_divergence= has_rod_divergence, bsdf=rods_bdsf)
    
    
    
    left_light = scene_definition.get_ring_light(camera_x+light_offset//2, camera_distance,camera_z, cam_light_intensity)
    right_light = scene_definition.get_ring_light(camera_x-light_offset//2, camera_distance,camera_z, cam_light_intensity)
    sensor = scene_definition.get_ring_camera(frame_width,frame_height,camera_x,camera_distance, camera_z, fov,spp=samples_per_pass)
    
    
    rods_dict = dict((f"rod_{i}",rod) for i,rod in enumerate(rods_group))
    scene_dict = {
        "type" : "scene",
        # "rods_material":rods_material,
        # "grids_material":grids_material,
        "myintegrator" : {
            #"type" : "path",
            "type":"volpath",
            "samples_per_pass":samples_per_pass,
            "max_depth": 8,
        },
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
    scene_dict.update(rods_dict)
    return scene_dict

def render_scene(scene_dict):

    scene = mi.load_dict(scene_dict)

    img = mi.render(scene)
    bitmap = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
    
    arr_bitmap= np.array(bitmap)
    # clipping
    # arr_bitmap[arr_bitmap>1] = 1
    return arr_bitmap


def add_grid(scene_dict, grid_positions, config):
    
    grid_material = config['fuel_material']['grids']
    alpha_u = grid_material['material_alpha_u']
    alpha_v = grid_material['material_alpha_v']
    gray_intensity =grid_material['diffuse_gray_intensity']
    bsdf_blend_factor = grid_material['metal_to_bsdf_blend_factor']

    grid_material = scene_definition.get_grid_bsdf(gray_intensity,bsdf_blend_factor,alpha_u,alpha_v)

    for grid_z in grid_positions:    
        grid = {
            "type": "obj",
            "filename": "assets/wta-grid.obj",
            "to_world": mi.ScalarTransform4f.translate([-1,118,-grid_z]).scale(93).rotate([1, 0, 0], angle=-90),
            "material":grid_material
        }
        scene_dict[f'grid_obj_{grid_z}'] = grid

fov,height,width,crop_shift = 5.85,*ref_img.shape[:2],0
# Video frame scrop
fov,height,width,crop_shift = 22,video_config['height'],video_config['width'],59
camera_z= -180
render_intensity_factor = 2

scene_dict = get_nfa_scene_dict(config,fov,height,width,camera_z,has_rod_divergence=True)
mod3_spacing_mm = grid_detection["mods"][-1]['spacing_mm']
np.cumsum(mod3_spacing_mm)

add_grid(scene_dict,np.cumsum(mod3_spacing_mm),config)

# scene_dict['constant'] = {
#             'type': 'constant',
#             'radiance': {
#                 'type': 'rgb',
#                 'value': .3,
#             }
#         }

bitmap = render_scene(scene_dict)

plt.imshow(bitmap)


            
```

```python

# Slim crop
fov,height,width,crop_shift = 5.85,*ref_img.shape[:2],0
# Video frame scrop
fov,height,width,crop_shift = 22,video_config['height'],video_config['width'],59
camera_z= -4480
render_intensity_factor = 2

scene_dict = get_nfa_scene_dict(config,fov,height,width,camera_z)
bitmap = render_scene(scene_dict)

arr = bitmap *render_intensity_factor
ref_img_rgb = np.dstack([ref_img,ref_img,ref_img])
fig,(ax1,ax2,ax3,ax2_plot,ax3_plot) = plt.subplots(5,1,figsize=(16,20))

arr_crop = arr[:,crop_shift:crop_shift+ref_img_rgb.shape[1]]
concat = np.concatenate([ arr_crop, ref_img_rgb],axis=0)

ax1.imshow(concat,vmin=0,vmax=1)
ax2.imshow(arr,vmin=0,vmax=1)
ax3.imshow(ref_img_rgb,vmin=0,vmax=1)

ax2_plot.plot(np.mean( arr, axis=0))
ax3_plot.plot(np.mean( ref_img_rgb, axis=0))
```

```python
import imageio
import cv2
import time 

from datetime import datetime

def get_frames(frames_count,frame_width,frame_height,has_rod_divergence = False,seed = 456):
    # Video frame scrop
    fov = 22
    render_intensity_factor = 2 # scales intensities so the output looks as the ref video 
    # TODO calc the param automatically
    mod3_spacing_mm = grid_detection["mods"][-1]['spacing_mm']
    grid_positions = np.cumsum(mod3_spacing_mm)

    from_i,to_i = 0,4500
    np.random.seed(seed)
    frames_nums = np.linspace(from_i,to_i,frames_count).astype(int)
    for i in frames_nums:
        camera_z = -i
        scene_dict = get_nfa_scene_dict(config,fov,frame_height,frame_width,camera_z,has_rod_divergence=has_rod_divergence)
        add_grid(scene_dict,grid_positions,config)
        
        scene = mi.load_dict(scene_dict)
        
        render_random = np.random.randint(0,high =  2**31 - 1)
        img = mi.render(scene,seed = render_random )
        bitmap = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
        bitmap_arr = np.array(bitmap) *render_intensity_factor
        bitmap_arr[bitmap_arr>1] = 1
                
        yield i, bitmap_arr
        

# imageio.imwrite(f"test_frame_{i:05}.png",arr)

def create_video(file_path,frame_number,frame_width,frame_height,fps = 25,has_rod_divergence=False):
    fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")

    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (frame_width,frame_height))

    for i,frame in  get_frames(frame_number,frame_width,frame_height,has_rod_divergence=has_rod_divergence):
        #f = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        print(datetime.now(), 'processed', f"{i/frame_number}")
        f = (frame *255).astype(np.uint8) 
        writer.write(f)

    writer.release()

timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S').replace(':','-')

import os
import pathlib

output_video_dir = pathlib.Path("output")
os.makedirs(output_video_dir, exist_ok = True)
video_path = output_video_dir/ f'test-{timestamp}.avi'
fps = 25
frames_count = 1000
create_video(
    video_path,
    frames_count, 
    video_config['width'],
    video_config['height'],
    fps=fps,
    has_rod_divergence=True)

import shutil
shutil.copy(video_path, output_video_dir/'test.avi')

print('done')
```

```python

```

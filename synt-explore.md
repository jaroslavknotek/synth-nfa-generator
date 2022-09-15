---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: synthdata-Cub-fwJX-py3.9
    language: python
    name: synthdata-cub-fwjx-py3.9
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
# Image has been removed
ref_img = np.zeros((10,10))

```

```python
plt.figure(figsize = (16,100))
plt.imshow(ref_img)
```

```python
import rods_hexagon
import scene_definition

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
}
config = {
    "measurements":measurements,
    "fuel_material": fuel_material
}


def get_nfa_scene_dict(config):
    rods_group_id = "my_rods_group"
    
    #get_rods_bsdf(gray_intensity,bsdf_blend_factor,material_alpha_u,material_alpha_v,  spectral_params = None)
    
    rods_material = config['fuel_material']['rods']
    rods_material_alpha_u = rods_material['material_alpha_u']
    rods_material_alpha_v = rods_material['material_alpha_v']
    rods_gray_intensity =rods_material['diffuse_gray_intensity']
    rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']
    
    camera_distance= config['measurements']['camera_distance_mm'] #given by blueprints (approx)
    light_offset = config['measurements']['light_offset_mm'] #given by blueprints (approx)
    
    samples_per_pass = 64
    samples_per_pass = 256
    cam_light_intensity = 47100
    camera_x= 0
    camera_z= -4480
    fov = 5.85
        
    rods_bdsf = scene_definition.get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
    rods_group = scene_definition.generate_rods_group(rods_group_id,config, bsdf=rods_bdsf)

    rods_instance = {
        "type": "instance",
        "my_ref": {
            "type": "ref",
            "id": rods_group_id,
        },
        #"to_world":mi.ScalarTransform4f.rotate([0,0,1],4)
    }
    
    left_light = scene_definition.get_ring_light(camera_x+light_offset//2, camera_distance,camera_z, cam_light_intensity)
    right_light = scene_definition.get_ring_light(camera_x-light_offset//2, camera_distance,camera_z, cam_light_intensity)

    height,width = ref_img.shape[:2]
    sensor = scene_definition.get_ring_camera(width,height,camera_x,camera_distance, camera_z, fov,spp=samples_per_pass)

    return {
        "type" : "scene",
        # "rods_material":rods_material,
        # "grids_material":grids_material,
        "myintegrator" : {
            #"type" : "path",
            "type":"volpath",
            "samples_per_pass":samples_per_pass,
            "max_depth": 8,
        },
        "group1":rods_group,
        "rods_instance": rods_instance,
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
    

scene_dict = get_nfa_scene_dict(config)
scene = mi.load_dict(scene_dict)

img = mi.render(scene)
bitmap = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)

render_intensity_factor = 2
arr = np.array(bitmap)

print(np.min(arr))
arr = arr *render_intensity_factor

ref_img_rgb = np.dstack([ref_img,ref_img,ref_img])

fig,(ax1,ax2,ax3,ax2_plot,ax3_plot) = plt.subplots(5,1,figsize=(16,20))
concat = np.concatenate([ arr, ref_img_rgb],axis=0)
ax1.imshow(concat,vmin=0,vmax=1)
ax2.imshow(arr,vmin=0,vmax=1)
ax3.imshow(ref_img_rgb,vmin=0,vmax=1)

ax2_plot.plot(np.mean( arr, axis=0))
ax3_plot.plot(np.mean( ref_img_rgb, axis=0))
```

```python
arr_focused =arr.copy()
```

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
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
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
```

```python
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import tempfile


import synthnf.io as io
import synthnf.config.defaults as defaults
import synthnf.config.assets as assets
import synthnf.scene as scene
```

```python
temp_root=io.create_temp_folder().parent

import shutil
shutil.rmtree(temp_root)
```

```python
#mi.set_variant('cuda_ad_spectral') # needs to be build manually

#mi.set_variant('scalar_spectral')

plt.rcParams.update({'font.size': 14})
```

```python
rnd = np.random.RandomState(seed = 123)
```

```python
def fig_imwrite(filename,img):
    fig_path = figs_out/filename
    io.imwrite(fig_path,img)
    print("Figure saved to ", str(fig_path))
```

```python
default_camera_distance = 300
default_camera_z = 50

default_sensor = scene.cam_perspective_lookat(
    [0, default_camera_distance, default_camera_z],
    res_y=1080,
    res_x=1920
)

default_high_contrast_material = {
    'type': 'diffuse',
    'reflectance': {
        'type': 'rgb',
        'value': [1, 0, 1]
    }
}

default_scene = {
    "type": "scene",
    "integrator": {"type": "path"},
    "sensor":default_sensor,
    "light": {
        "type": "constant",
        'radiance': {
            'type': 'rgb',
            'value': 1
        }
     },
}
```

```python
import imageio

figs_out = pathlib.Path('../../figs')
figs_out.mkdir(exist_ok=True,parents=False)
figs_out.absolute()
```

```python
import synthnf.simulation as simu

params = simu.RandomParameters(
    seed = 123,
    n_textures = 11,
    max_divergence_mm = 10,
    max_bow_mm = 100,
    max_z_displacement_mm = 10
)
```

# Geometry


## Fuel Rods

```python
import synthnf.materials as mat

import synthnf.geometry.curves as curves
import synthnf.geometry.fuel_rods_mesh as frm

import synthnf.report as rep
import bezier


curve_fa, curves_rod = params.bow_divergence()
curves_rod = curves_rod[:10]

rep.plot_bow(curve_fa,curves_rod)


for span in params.grid_tops_mm:
    plt.axhline(span,c='gray',alpha = .5)
plt.savefig(figs_out/'fig_fa_bow_div.png')
plt.show()
# create curve
```

```python
tip_path = assets.get_asset_path('tip_wta.ply')
tip_scene = default_scene.copy()
tip = {
    'type':'ply',
    'filename':str(tip_path),
}
tip_scene['model'] = mat.color_model_faces(tip)

tip_scene['sensor'] = scene.cam_orto_lookat(
    np.array([0,11,5])/4,
    target = np.array([0,0,1])/2,
    res_x=1024,
    res_y=1024,
    orto_scale_xy=[1,1]
)

tip_render = scene.render_scene(tip_scene,denoise=True,alpha=True)
plt.imshow(tip_render)
fig_imwrite('fig_tip_mesh.png', tip_render)
```

## Spacer Grid

```python
import synthnf.geometry.spacer_grid_mesh as sgm

tooth_pth =  str(assets.get_asset_path('grid_wta.ply'))

scene_dict = default_scene.copy()

test_tooth = {        
    'type': 'ply',
    'filename': tooth_pth,
    "to_world": mi.ScalarTransform4f.rotate([0, 0, 1], angle=180),
}

scene_dict['model'] = mat.color_model_faces(test_tooth)
scene_dict['sensor'] = scene.cam_orto_lookat(
    [0,1,0], 
    res_y=1080,
    res_x=1920,
    orto_scale_xy=[4,4]
)

tooth_img = scene.render_scene(scene_dict)

## Mirror

tooth_ply = io.read_ply(tooth_pth)
p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = io.decompose_ply(tooth_ply)
p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = sgm.mirror_down(p_x,p_y,p_z,v_1,v_2,v_3, n_x,n_y,n_z)
tooth_mirrored_mesh = sgm.compose_mesh(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z)

test_tooth['filename'] = io.save_mesh_to_temp(tooth_mirrored_mesh)
scene_dict['model']= mat.color_model_faces(test_tooth)

tooth_mirrored_img = scene.render_scene(scene_dict)

## Array
p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = sgm.array_left(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z,n=5)
tooth_arrayed_mesh = sgm.compose_mesh(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z)

test_tooth["to_world"] = mi.ScalarTransform4f.rotate([0, 0, 1], angle=180).translate([2.5,0,0])
test_tooth['filename'] = io.save_mesh_to_temp(tooth_arrayed_mesh)
scene_dict['model']= mat.color_model_faces(test_tooth)


tooth_arrayed_img = scene.render_scene(scene_dict)

plt.imshow(tooth_img)
plt.show()
plt.imshow(tooth_mirrored_img)
plt.show()
plt.imshow(tooth_arrayed_img)
```

```python
import synthnf.geometry.spacer_grid_mesh as sgm
import synthnf.models as models

tooth_ply = io.read_ply(tooth_pth)
grid_model = models.spacer_grid()

grid_model['to_world'] = mi.ScalarTransform4f.scale(.5)
grid_scene = default_scene.copy()
grid_scene['model']= mat.color_model_faces(grid_model)

full_grid = scene.render_scene(grid_scene)
plt.imshow(full_grid)
```

```python
tooth_figs = [
    tooth_img[100:600,780:1130],
    tooth_mirrored_img[100:-100,780:1130],
    tooth_arrayed_img[50:-50,100:1750],
    full_grid[200:1000]
]
tooth_figs_names = [
    "tooth_single",
    "tooth_mirrored_down",
    "teeth_arrayed",
    "full"
]
for img,name in zip(tooth_figs,tooth_figs_names):
    fig_imwrite(f"fig_grid_{name}.png",img)
```

# Material

```python
import pandas as pd

def plot_spectrum(df,title=None):
    cols = ["n","k"]
    lbls = ["Refractive","Exctinction"]
    rs = [ df[c] for c in cols]
    wv = df['wavelength']
    [plt.plot(wv*1000,r,label=c) for (c,r) in zip(lbls,rs)]

    plt.ylabel("Refractive/Exctintion coefs")
    plt.xlabel("Wavelength[nm]")
    if title:
        plt.title(title)
    plt.xlim(400,700)
    plt.legend()


df_zirconium = pd.read_csv( assets.get_asset_path('spectrum_zirconium.csv'))
plot_spectrum(df_zirconium,title="Zirconium")
plt.savefig(figs_out/'fig_spectrum_zr.png')
plt.show()

df_inconel = pd.read_csv(assets.get_asset_path('spectrum_inconel.csv'))
plot_spectrum(df_inconel,title="Inconel")
plt.savefig(figs_out/'fig_spectrum_inconel.png')
```

## Rods

```python
# import synthnf.materials.textures as textures

# import perlin_numpy
# shape = np.array([100,100])
# noise_low = perlin_numpy.generate_fractal_noise_2d(
#     [128*2,128], 
#     [2,2], 
#     2,
#     persistence=.5,
#     lacunarity=2,
#     tileable = (True,True)
# )
# noise_high = perlin_numpy.generate_fractal_noise_2d(
#     [128*2,128], 
#     [16,16], 
#     2,
#     persistence=.5,
#     lacunarity=2,
#     tileable = (True,True)
# )

# alpha = .9
# noise =  alpha*noise_low + (1-alpha)* noise_high

# plt.imshow(np.tile(noise,(2,2)))
```

```python
# import synthnf.scene as scene

# def compose_scene_rod_piece(
#     cam_ligth_intensity = 20,
#     light_height = 100,
#     blend_ratio = .5,
#     cylinder_count = 1
# ):
#     scene_dict = default_scene.copy()

#     # make sure it adds up to at most 1
#     blend_texture = np.clip(0,1,-.1 + blend_ratio*1.1 + (_norm(noise)-.5)*.2)

#     bsdf = mat.create_gray_diffuse(1)
#     zircon_conductor = mat.create_zirconium()

#     material=  mat.blend_materials(
#         bsdf,
#         zircon_conductor,
#         blend_texture
#     )
    
#     radius = 10
#     radius_padd = (radius *2)*1.3
#     xs = np.arange(cylinder_count)*radius_padd
#     xs = xs - np.mean(xs)
#     for i,x in enumerate(xs):
#         fr = {        
#             'type': 'cylinder',
#             'radius': radius,
#             'p0':[x,0,-15],
#             'p1':[x,0,15],
#             'material': material
#         }

#         scene_dict[f'model_{i}']= fr
        
#     scene_dict['sensor'] =scene.create_lookat_sensor([0,100,0],resolution_width = 1920,resolution_height  = 1080)
#     scene_dict['light'] = scene.create_mount_emmiter([-20,100,0],cam_ligth_intensity,light_height_mm=light_height)
#     scene_dict['light_1'] = scene.create_mount_emmiter([20,100,0],cam_ligth_intensity,light_height_mm=light_height)
#     return scene_dict

# rod_imgs = []
# blends = np.flip(np.linspace(.4,1.2,5))
# for i in blends:
#     scene_dict = compose_scene_rod_piece(blend_ratio=i)
#     scene_img = scene.render_scene(scene_dict,denoise = True)
#     rod_imgs.append(scene_img[:,600:1300])
    
# fig,axs = plt.subplots(1,len(rod_imgs),figsize=(4*len(rod_imgs),4))

# for ax,img,i in zip(axs,rod_imgs,blends):
#     ax.imshow(img)
#     ax.set_title(f"Blend factor:{i:.2f}")
    
# for img,i in zip(rod_imgs,blends):
#     fig_imwrite(f'fig_rod_piece_{i}.png',img)
```

```python
import synthnf.materials.textures as textures
import synthnf.scene as scene

def compose_scene_rod_piece_real_param(
    cam_ligth_intensity = 20,
    light_height = 100,
    cloudiness = .5,
    cylinder_count = 1
):
    scene_dict = default_scene.copy()
    radius = 9.2
    radius_padd = (radius *2)*1.3
    
    # make sure it adds up to at most 1
    if cloudiness == 0:
        blend_texture = np.zeros((2,2))
    else:
        noise = textures.rod_noise(resolution_factor=4,rod_height_mm=60,rod_width_mm=radius*2,seed=123)
        blend_texture = textures.blend(noise,.2,cloudiness)

    material = mat.zirconium_blend(blend_texture=blend_texture)
    
    tip_path = assets.get_asset_path('tip_wta.ply')
    
    
    xs = np.arange(cylinder_count)*radius_padd
    xs = xs - np.mean(xs)
    for i,x in enumerate(xs):
        fr = {        
            'type': 'cylinder',
            'radius': radius,
            'p0':[x,0,30],
            'p1':[x,0,-30],
            'material': material
        }

        scene_dict[f'model_{i}']= fr
        
        scene_dict[f'tip_{i}'] = {
            'type':'ply',
            'filename':str(tip_path),
            'to_world':mi.ScalarTransform4f.translate([x,0,29]).scale(radius*2),
            'material': material
        }
        
    cam_distance = defaults.blueprints.camera_ring.diameter_mm/2
    light_offset = defaults.blueprints.camera_ring.light_offset_mm
    
    # TODO not perspective?
    
    scene_dict['sensor'] =scene.cam_perspective_lookat([0,cam_distance,0],res_x = 1920*2,res_y  = 1080 *2)

    scene_dict['light'] = scene.create_mount_emmiter([-light_offset/2,cam_distance,0],cam_ligth_intensity,light_height_mm=light_height)
    scene_dict['light_1'] = scene.create_mount_emmiter([light_offset/2,cam_distance,0],cam_ligth_intensity,light_height_mm=light_height)
    return scene_dict

rod_imgs = []
cloudiness = np.linspace(0.2,.8,5)
for i in cloudiness:
    scene_dict = compose_scene_rod_piece_real_param(
        cloudiness=i,
        light_height=500,
        cam_ligth_intensity=100
    )
    scene_img = scene.render_scene(scene_dict,denoise = True)
    #rod_imgs.append(scene_img[600:-900,1600:-1600])
    rod_imgs.append(scene_img[:-600,1600:-1600])
    #rod_imgs.append(scene_img)
    
    
fig,axs = plt.subplots(1,len(rod_imgs),figsize=(4*len(rod_imgs),4))

for ax,img,i in zip(axs,rod_imgs,cloudiness):
    ax.imshow(img)
    ax.set_title(f"Cloudiness:{i:.2f}")
    
for img,i in zip(rod_imgs,cloudiness):
    fig_imwrite(f'fig_rod_piece_{i:.2f}.png',img)
plt.show()
```

## Spacer Grid

```python
import synthnf.utils as utils

def compose_grid_scene(cloudiness):
    grid_mat_scene = default_scene.copy()
    
    g_params = simu.RandomParameters(
        seed = 123,
        cloudiness=cloudiness
    )

    model_mat_grid = models.spacer_grid()
    model_mat_grid['to_world'] = mi.ScalarTransform4f.scale(.75)
    model_mat_grid['material'] = g_params.grid_material()
    grid_mat_scene['model'] = model_mat_grid

    cam_ligth_intensity = 50
    cam_distance = 220
    cam_z = 0
    grid_mat_scene['sensor'] = scene.cam_perspective_lookat([0,cam_distance,cam_z],target=[0,0,cam_z], res_y=1080,res_x=1920 )
    grid_mat_scene['light'] = scene.create_mount_emmiter([-60,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=1000)
    grid_mat_scene['light_1'] = scene.create_mount_emmiter([60,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=1000)

    return grid_mat_scene


grid_imgs = []
cloudiness = np.linspace(0.2,.90,5)

for i in cloudiness:
    scene_dict = compose_grid_scene(cloudiness=i)
    scene_img = scene.render_scene(scene_dict,denoise = True)
    grid_imgs.append(scene_img)
```

```python
n = len(grid_imgs)
width = grid_imgs[0].shape[1]
starts = np.linspace(0,width,n +1 +2 ).astype(int)

w = starts[1]
pcs = []
for img,i in zip(grid_imgs,starts[1:]):
    i = starts[3]
    crop = np.array(img[:,i:i+w])
 #   crop[:,0:10,0:3] = 0
    pcs.append(crop)

fig,axs =plt.subplots(1,n,figsize=(n*3,4))
for ax,img in zip(axs,pcs):
    ax.imshow(img)
    
for img,i in zip(pcs,cloudiness):
    fig_imwrite(f'fig_grid_piece_{i:.2f}.png',img)
    
# plt.imshow(grid_mat_img)
```

# Scene and Illumination


## Camera Stand

```python
# 1 camera even thouhgh real one has three

bl = defaults.blueprints
diameter = bl.camera_ring.diameter_mm
diameter_padded =  diameter/2 * 1.2
camera_position = [0,diameter/2,0]

plt.figure(figsize = (6,6))
cam_ring = plt.Circle((0, 0), diameter/2, color='black', fill=False,alpha = .2,label='Camera Ring Mount')

cam_poly = np.array([[-60,60],[60,60],[30,-50],[-30,-50]]) - [0, diameter/2]
cam_box = plt.Polygon(cam_poly,edgecolor ='black',facecolor = 'black',label='Camera')


light_radius = bl.camera_ring.light_diameter_mm/2 *3
cam_light_left = plt.Circle(
    [-bl.camera_ring.light_offset_mm/2,-diameter/2],
    radius = light_radius,
    color = 'orange',
    label = "Camera Light"
)

cam_light_right = plt.Circle(
    [bl.camera_ring.light_offset_mm/2,-diameter/2],
    radius = light_radius,
    color = 'orange',
)


ax = plt.gca()
rcs = frm.generate_rod_centers(11,9.1,3.65)
ax.plot(rcs[:,0],rcs[:,1],'o',label='FA')


ax.set_xlim((-diameter_padded, diameter_padded))
ax.set_ylim((-diameter_padded, diameter_padded))

# ax.set_xlim((-50, 50))
# ax.set_ylim((-diameter_padded, (diameter_padded - diameter)))


ax.add_patch(cam_ring)
ax.add_patch(cam_box)
ax.add_patch(cam_light_left)
ax.add_patch(cam_light_right)
ax.set_xlabel("X[mm]")
ax.set_ylabel("Y[mm]")
plt.legend()
plt.savefig(figs_out/'fig_scene.png')
```

## Illumination

```python
tubus = {
    'type':'cylinder',
    "radius":defaults.blueprints.camera_ring.diameter_mm/1.9,
    "flip_normals":True,
    "p0":[0,0,200],
    "p1":[0,0,-200],
    'material': {
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': .2
        }
    }
}

local_scene_dict = compose_scene_rod_piece_real_param(cloudiness=.4,cylinder_count=3)
local_scene_dict['tubus'] = tubus
local_render = scene.render_scene(
    local_scene_dict,
    clip=True,
    alpha = False,
    denoise = False
)
plt.title("local")
plt.imshow(local_render)
plt.show()

global_scene_dict = compose_scene_rod_piece_real_param(cloudiness=.4,cylinder_count=3)

del global_scene_dict['light_1']
global_scene_dict['light'] = {
    'type': 'constant',
    'radiance': {
        'type': 'rgb',
        'value': .8,
    }
}

global_scene_dict['tubus'] = tubus

global_render = scene.render_scene(
    global_scene_dict,
    clip=True,
    alpha = False,
    denoise = False
)

plt.title("global")
plt.imshow(global_render)
```

```python
ll = local_render[700:local_render.shape[0]//2,1400:-1400]
fig_imwrite('fig_ill_cam.png',ll)

gg = global_render[-global_render.shape[0]//2:-700,1400:-1400]
fig_imwrite('fig_ill_global.png',gg)

fig_imwrite('fig_ill_cmp.png',np.vstack([ll,gg]))

plt.imshow(ll)
plt.show()
plt.imshow(gg)
```

```python
side_rod_count = 11
rod_width_mm = 9.1
gap_between_rods_width_mm = 3.65
rod_centers = frm.generate_rod_centers(
    side_rod_count-2 ,
    rod_width_mm,
    gap_between_rods_width_mm,
    outern_layers = None
)

rod_centers_out = frm.generate_rod_centers(
    side_rod_count,
    rod_width_mm,
    gap_between_rods_width_mm,
    outern_layers = 2
)

plt.plot(
    rod_centers[:,0],
    rod_centers[:,1],
    'o',
    label = 'inner'
)
plt.plot(
    rod_centers_out[:,0],
    rod_centers_out[:,1],
    'o',
    label = 'outer'
)

plt.legend()
```

```python
import synthnf.geometry.spacer_grid_mesh as sgm
import synthnf.models as models
tooth_ply = io.read_ply(tooth_pth)
grid_model = models.spacer_grid()

grid_scene = default_scene.copy()
grid_scene['model']= mat.color_model_faces(grid_model)
grid_scene['sensor'] = scene.cam_perspective_lookat([0,400,100],res_x=1920,res_y = 1080)


rod_centers_peripheral = frm.generate_rod_centers(
    side_rod_count,
    rod_width_mm,
    gap_between_rods_width_mm,
    outern_layers = 2
)

for i,(x,y) in enumerate(rod_centers_peripheral):
    grid_scene[f'tip_{i}'] = {
        'type':"cylinder",
        'p0':[x,y,-20],
        'p1':[x,y,20],
        'radius':rod_width_mm/2,
        'mat':mat.debug_pink()
    }


full_grid = scene.render_scene(grid_scene)

grid_scene['sensor'] = scene.cam_perspective_lookat([0,220,450],res_x=1200,res_y = 1024)
full_grid_above = scene.render_scene(grid_scene)

fig,axs= plt.subplots(1,2,figsize = (10,10))
axs[0].imshow(full_grid)
axs[1].imshow(full_grid_above)
```

# Video - Put Together

```python
import synthnf.app as app

force =False
force = True
n_frames = 25
out_folder = pathlib.Path(f"../output/test_vid")

app.create_inspection(
    out_folder,
    n_frames,
    force
)
```

# TODO

- textures (grid/rods)
- vide
- labels
- shrunk etc

```python
import synthnf.materials.textures as textures
import synthnf.geometry.curves as curves

# TODO

```

```python
# curves instead of mesh
# check texture UV mapping
folder = io.create_temp_folder()
curves_path = folder/"curves.txt"
cvs = """-1.0 0.1 0.1 0.5
-0.3 1.0 1.2 0.1
 0.3 1.1 0.3 0.1
 1.0 1.2 1.4 0.1

-1.0 2.2 5.0 0.1
-2.3 2.3 4.0 0.1
 3.3 2.2 3.0 0.1
 4.0 2.3 2.0 0.1
 4.0 2.2 1.0 0.1
 4.0 2.3 0.0 0.1
 """
# with open(curves_path,'w') as fh:
#     fh.write(cvs)
```

```python
# TODO to add variabily - each rod can be rotated a bit
```

```python
import matplotlib.pyplot as plt
import numpy as np
import synthnf.simulation as simu

ps = [simu.RandomParameters(max_bow_mm=i) for i in (np.random.rand(12)-.5)*20]

curves = [p.bow_divergence()[0] for p in ps]

ts = np.linspace(0,1,101)
ax = plt.figure().add_subplot(projection='3d')
ax.view_init(10, 20, 0)
for curve in curves:
    xs,ys,zs = curve.evaluate_multi(ts)
    
    ax.plot(xs,ys,zs)
```

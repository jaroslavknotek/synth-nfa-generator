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
import numpy as np
import matplotlib.pyplot as plt

import mitsuba as mi

import pathlib
import tempfile


import synthnf.io as io
import synthnf.config.defaults as defaults
import synthnf.config.assets as assets
import synthnf.scene as scene
```

```python
mi.variants()
```

```python
#mi.set_variant('cuda_ad_spectral') # needs to be build manually
mi.set_variant('cuda_ad_rgb')
#mi.set_variant('scalar_spectral')

plt.rcParams.update({'font.size': 14})
```

```python
rnd = np.random.RandomState(seed = 123)
```

```python
def _norm(arr,nan_to_zero = False):
    min_arr =np.min(arr)
    max_arr = np.max(arr)
    if min_arr == max_arr and 0 <= min_arr <= 1 :
        return arr
    res = (arr - min_arr)/(max_arr - min_arr)
    if nan_to_zero:
        res = np.nan_to_num(res)
    return res

def fig_imwrite(filename,img):
    img_c = np.uint8(np.clip(0,1,img)*255)
    fig_path = figs_out/filename
    imageio.imwrite(fig_path,img_c)
    print("Figure saved to ", str(fig_path))
```

```python
default_camera_distance = 300
default_camera_z = 50

default_sensor = scene.create_lookat_sensor(
    [0, default_camera_distance, default_camera_z],
    resolution_height=1080,
    resolution_width=1920
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
# IO    
def draw_curve(curve, samples = 100,ax=None, **kwargs):
    
    if ax is None:
        ax  = plt
    
    zs = np.linspace(0,1,samples)
    points = curve.evaluate_multi(zs)
    xx = points[0]
    zz = points[2]
    ax.plot(xx,zz,**kwargs)
```

```python
import imageio

figs_out = pathlib.Path('../../figs')
figs_out.mkdir(exist_ok=True,parents=False)
figs_out.absolute()
```

# Geometry


## Fuel Rods

```python
import synthnf.materials as mat
import synthnf.simulation.artificial as simu
import synthnf.geometry.curves as curves
import synthnf.geometry.fuel_rods_mesh as frm

import bezier

fa_height_mm = 1000
spans = np.array([0,200,500,1000])
max_divergence_mm = 1
max_bow_mm = 1
span_margin_mm = 50
rod_count = 5

max_bow_mm = (rnd.rand(2)*2-1)*max_bow_mm
divergences_xy_mm = [simu.rnd_divergence_adjusted(spans,max_divergence_mm,rnd) for _ in range(rod_count)]

# THIS is not simulation
curve_fa, curves_rod = simu.create_nfa_curves( 
    max_bow_mm,
    divergences_xy_mm,
    span_margin_mm,
    spans
)

fig,ax = plt.subplots(1,1,figsize=(4,5))
draw_curve(curve_fa,ax=ax ,label = 'FA',linewidth = 3,c='red')

for i, div_curve in enumerate(curves_rod):
    draw_curve(div_curve,ax=ax,label=f'Rod {i+1}',alpha = .5)

plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.ylabel("FA Height[mm]")
plt.xlabel("Bow Divergence[mm]")
#plt.gca().invert_yaxis()
plt.subplots_adjust(right=0.6,top=.9)
plt.savefig(figs_out/'fig_fa_bow_div.png')
plt.show()
# create curve

 
scene_dict = default_scene.copy()
for i, piecewise_curve in enumerate(curves_rod):
    
    vertices,faces,uv = frm.get_rod_geometry(20,9,curve=piecewise_curve,radius= .2)
    rod_mesh = frm.get_mesh(vertices,faces,uv)

    rod_mesh_path = io.save_mesh_to_temp(rod_mesh)

    test_tooth = {        
        'type': 'ply',
        'filename': rod_mesh_path,
        "to_world": mi.ScalarTransform4f.translate([i - 2 ,0,2]).scale([1,1,.004]),
    }

    scene_dict[f'model_{i}'] = mat.color_model_faces(test_tooth)
    
scene_dict['sensor'] = scene.create_lookat_sensor([0,-10,0], type_='perspective',resolution_width=1920,resolution_height=1080)
rods_img = scene.render_scene(scene_dict)
rods_img_crop = rods_img[:,200:-200]
plt.imshow(rods_img_crop)
plt.axis('off')


fig_imwrite("fig_rod_mesh_div.png",rods_img_crop)
```

```python
tip_path = assets.get_asset_path('tip_wta.ply')
tip_scene = default_scene.copy()
tip = {
    'type':'ply',
    'filename':str(tip_path),
}
tip_scene['model'] = mat.color_model_faces(tip)

tip_scene['sensor'] = scene.create_lookat_sensor([0,11,5],resolution_width=1024,resolution_height=1024)

tip_render = scene.render_scene(tip_scene,denoise=True,alpha=True)
tip_render = tip_render[120:800,200:-200]
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
    "to_world": mi.ScalarTransform4f.rotate([0, 0, 1], angle=180).scale(.02),
}

scene_dict['model'] = mat.color_model_faces(test_tooth)
scene_dict['sensor'] = scene.create_lookat_sensor([0,1,0], type_='orthographic',resolution_height=1080,resolution_width=1920)

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

test_tooth["to_world"] = mi.ScalarTransform4f.rotate([0, 0, 1], angle=180).scale(.02).translate([30,0,0])
test_tooth['filename'] = io.save_mesh_to_temp(tooth_arrayed_mesh)
scene_dict['model']= mat.color_model_faces(test_tooth)

tooth_arrayed_img = scene.render_scene(scene_dict)
```

```python
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
grid_model['to_world'] = mi.ScalarTransform4f.scale(.75)
grid_scene = default_scene.copy()
grid_scene['model']= mat.color_model_faces(grid_model)

full_grid = scene.render_scene(grid_scene)
plt.imshow(full_grid)
```

```python
tooth_figs = [
    tooth_img[:600,780:1130],
    tooth_mirrored_img[:,780:1130],
    tooth_arrayed_img[:,100:1750],
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
    plt.xlabel("Wavelength[mm]")
    if title:
        plt.title(title)
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
import perlin_numpy
shape = np.array([100,100])
noise_low = perlin_numpy.generate_fractal_noise_2d(
    [128*2,128], 
    [2,2], 
    2,
    persistence=.5,
    lacunarity=2,
    tileable = (True,True)
)
noise_high = perlin_numpy.generate_fractal_noise_2d(
    [128*2,128], 
    [16,16], 
    2,
    persistence=.5,
    lacunarity=2,
    tileable = (True,True)
)

alpha = .9
noise =  alpha*noise_low + (1-alpha)* noise_high

plt.imshow(np.tile(noise,(2,2)))
```

```python
import synthnf.scene as scene

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
import synthnf.scene as scene

def compose_scene_rod_piece_real_param(
    cam_ligth_intensity = 20,
    light_height = 100,
    cloudiness = .5,
    cylinder_count = 1
):
    scene_dict = default_scene.copy()

    # make sure it adds up to at most 1
    if cloudiness == 0:
        blend_texture = np.zeros((2,2))
    else:
        blend_texture = np.clip(0,1,-.1 + cloudiness*1.1 + (_norm(noise)-.5)*.2)

    material = mat.zirconium_blend(blend_texture=blend_texture)
    
    tip_path = assets.get_asset_path('tip_wta.ply')
    
    radius = 9.2
    radius_padd = (radius *2)*1.3
    xs = np.arange(cylinder_count)*radius_padd
    xs = xs - np.mean(xs)
    for i,x in enumerate(xs):
        fr = {        
            'type': 'cylinder',
            'radius': radius,
            'p0':[x,0,30],
            'p1':[x,0,-50],
            'material': material
        }

        scene_dict[f'model_{i}']= fr
        
        scene_dict[f'tip_{i}'] = {
            'type':'ply',
            'filename':str(tip_path),
            'to_world':mi.ScalarTransform4f.translate([x,0,29]).scale(radius),
            'material': material
        }
        
    cam_distance = defaults.blueprints.camera_ring.diameter_mm/2
    light_offset = defaults.blueprints.camera_ring.light_offset_mm
    scene_dict['sensor'] =scene.create_lookat_sensor([0,cam_distance,0],resolution_width = 1920*2,resolution_height  = 1080 *2)

    scene_dict['light'] = scene.create_mount_emmiter([-light_offset/2,cam_distance,0],cam_ligth_intensity,light_height_mm=light_height)
    scene_dict['light_1'] = scene.create_mount_emmiter([light_offset/2,cam_distance,0],cam_ligth_intensity,light_height_mm=light_height)
    return scene_dict

rod_imgs = []
cloudiness = np.linspace(0.2,.90,5)
for i in cloudiness:
    scene_dict = compose_scene_rod_piece_real_param(
        cloudiness=i,
        light_height=500,
        cam_ligth_intensity=100
    )
    scene_img = scene.render_scene(scene_dict,denoise = True)
    #rod_imgs.append(scene_img[600:-900,1600:-1600])
    rod_imgs.append(scene_img[420:-1020,1750:-1750])
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

def compose_grid_scene(cloudiness):
    grid_mat_scene = default_scene.copy()
    noise = perlin_numpy.generate_fractal_noise_2d(
        np.array([1,24])*128*4, 
        np.array([1,24]), 
        2,
        persistence=.5,
        lacunarity=2,
        tileable = (True,True)
    )

    blend_texture = np.clip(0,1,-.1 + cloudiness*1.1 + (_norm(noise)-.5)*.2)
    material=  mat.inconel_blend(blend_texture= blend_texture,gray_diffuse=.6) 

    np.random.seed(101)
    bumpmap = perlin_numpy.generate_fractal_noise_2d(
        np.array([1,8])*128, 
        np.array([1,8]), 
        2,
        persistence=.5,
        lacunarity=2,
        tileable = (True,True)
    )

    # material = {
    #     "type":"roughconductor",
    #     "material":"Au"
    # }
    grid_material = {
        'type': 'bumpmap',
        'arbitrary': {
            'type':'bitmap',
            'raw': True,
            'data': np.expand_dims(_norm(bumpmap)*2,axis=2)
        },
        'material': material
    }

    model_mat_grid = models.spacer_grid()
    model_mat_grid['to_world'] = mi.ScalarTransform4f.scale(.75)
    model_mat_grid['material'] = grid_material
    grid_mat_scene['model'] = model_mat_grid

    cam_ligth_intensity = 50
    cam_distance = 220
    cam_z = 0
    grid_mat_scene['sensor'] = scene.create_lookat_sensor([0,cam_distance,cam_z],target=[0,0,cam_z], resolution_height=1080,resolution_width=1920 )
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
    pcs.append(crop[250:830])

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
    denoise = True
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
    denoise = True
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

# Video - Put Together

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
import synthnf.geometry.curves as curves

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
cam_z = 1000
mod = grid_detection['mods'][0]
spacing_mm = mod['spacing_mm']
grid_heights_mm = mod['grid_heights_mm']

max_divergence_mm = 20
max_twist_bow_mm= 200
rod_bow_segments = 20, 
rod_circle_segments = 32
seed = 678,
n_textures = 11
z_displacement = None
cam_distance = defaults.blueprints.camera_ring.diameter_mm//2
# body
tip_path = assets.get_asset_path('tip_wta.ply')
# only visible
rod_centers = frm.generate_rod_centers()
rod_centers = rod_centers[~(rod_centers[:,1] <0)] 


cylinder_radius = rod_width_mm/2
spacing_mm = np.concatenate([[0], spacing_mm])
bent_random = np.random.RandomState(seed)

max_bow_mm = 50
max_divergence_mm = 2
span_margin_mm = 50
spans = np.cumsum(spacing_mm)
rod_height_mm = spans[-1]

bow_mm = (rnd.rand(2)*2-1)*max_bow_mm
divergences_xy_mm = np.array([simu.rnd_divergence_adjusted(spans,max_divergence_mm,rnd) for _ in range(len(rod_centers))])
span_margin_mm = 50

curve_fa, curves_rod = simu.create_nfa_curves( 
    bow_mm,
    divergences_xy_mm,
    span_margin_mm,
    spans
)

np.random.seed(123)
h = 2**(6)
noises = []
for _ in range(n_textures):
    noise = perlin_numpy.generate_fractal_noise_2d(
        [8*h,32], 
        [2*h,2], 
        2,
        persistence=.5,
        lacunarity=2,
        tileable = (True,True)
    )
    noises.append(noise)

noise = np.hstack(noises)
# TODO 1/3 is not spotted usually (or is it)
blend_ratio = .8
blend_texture = np.clip(0,1,-.1 + blend_ratio*1.1 + (_norm(noise)-.5)*.2)

plt.figure(figsize = (8,8))
plt.imshow(blend_texture.T)
plt.show()
material= mat.zirconium_blend(blend_texture=blend_texture)
scene_dict = default_scene.copy()



cam_z = -100
cam_ligth_intensity = 20
cam_distance = defaults.blueprints.camera_ring.diameter_mm/2
light_offset_mm = defaults.blueprints.camera_ring.light_offset_mm

fa_model = models.fuel_rods(
    # curves_rod=curves_rod,
    # rod_centers=rod_centers
)

scene_dict['model'] = fa_model
scene_dict['sensor'] = scene.create_lookat_sensor([0,cam_distance,cam_z],target=[0,0,cam_z-10], resolution_height=1024,resolution_width=1024)
scene_dict['light'] = scene.create_mount_emmiter([light_offset_mm/2,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=500)
scene_dict['light_1'] = scene.create_mount_emmiter([-light_offset_mm/2,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=500)

render = scene.render_scene(scene_dict,alpha=False, denoise=True, spp = 128*2)
plt.figure(figsize=(16,16))
plt.imshow(render)
```

```python
cam_z = 0
cam_ligth_intensity = 20
cam_distance = defaults.blueprints.camera_ring.diameter_mm/2
light_offset_mm = defaults.blueprints.camera_ring.light_offset_mm

tips = models.rod_tips()
fa_model = models.fuel_rods()

scene_dict = default_scene.copy()

scene_dict['tips'] = tips
scene_dict['model'] = fa_model
scene_dict['sensor'] = scene.create_lookat_sensor([0,cam_distance,cam_z],target=[0,0,cam_z-10], resolution_height=1024,resolution_width=1024)
scene_dict['light'] = scene.create_mount_emmiter([light_offset_mm/2,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=500)
scene_dict['light_1'] = scene.create_mount_emmiter([-light_offset_mm/2,cam_distance,cam_z],cam_ligth_intensity , light_height_mm=500)

render = scene.render_scene(scene_dict,alpha=False, denoise=True, spp = 128*2)
plt.figure(figsize=(16,16))
plt.imshow(render)
```

```python
import synthnf.models as models

grid_xyz = curve_fa.evaluate_multi(np.linspace(0,1,4))
 
grid_x,grid_y,_ = grid_xyz*10
grid_z = np.linspace(0,80,4)-40

grid_scene = default_scene.copy()
for x,y,z in zip(grid_x,grid_y,grid_z):
    grid = models.spacer_grid()
    grid['to_world'] = mi.ScalarTransform4f.translate([x,y,z]).scale(.1)
    grid_scene[f'model_{int(z)}']= grid
    
full_grid = scene.render_scene(grid_scene)
plt.imshow(full_grid)
```

```python
# curves instead of mesh
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
with open(curves_path,'w') as fh:
    fh.write(cvs)
```

```python
# TODO to add variabily - each rod can be rotated a bit
```

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
def _norm(arr,nan_to_zero = False):
    min_arr =np.min(arr)
    max_arr = np.max(arr)
    if min_arr == max_arr and 0 <= min_arr <= 1 :
        return arr
    res = (arr - min_arr)/(max_arr - min_arr)
    if nan_to_zero:
        res = np.nan_to_num(res)
    return res

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
def add_face_color(mesh_dict):

    mesh = mi.load_dict(mesh_dict)

    attribute_size = mesh.vertex_count() * 3
    mesh.add_attribute(
        "face_color", 3, [0] * attribute_size
    )

    N = mesh.face_count()

    face_colors = np.repeat(np.random.rand(N),3)
    
    mesh_params = mi.traverse(mesh)
    mesh_params["face_color"] = face_colors
    mesh_params.update()
    
    return mesh

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
import synthnf.simulation.artificial as simu
import synthnf.geometry.fuel_rods_mesh as frm

import bezier

fa_height_mm = 1000
pieces_heights = [200,500]
max_divergence_mm = 70
piece_margin_mm = 100

rnd = np.random.RandomState(seed = 123)

rand_x,rand_y = rnd.rand(2)*2-1
bezier_nodes = np.array([
    [0,0,0],
    [rand_x,rand_y,fa_height_mm//2],
    [0,0,fa_height_mm] #bottom
])

curve = simu.bezier_from_nodes(bezier_nodes)
pieces = np.concatenate([[0],pieces_heights,[fa_height_mm]])

divergences_xy_mm = simu.rnd_divergence_adjusted(pieces,max_divergence_mm,rnd)
piecewise_curve = simu.construct_piecewise_curve(pieces,divergences_xy_mm,piece_margin_mm)

fig,ax = plt.subplots(1,1,figsize=(4,5))

draw_curve(curve,ax=ax ,label = 'FA',linewidth = 3,c='red')

divergence_curves = []
for i in range(5):
    divergences_xy_mm = simu.rnd_divergence_adjusted(pieces,max_divergence_mm,rnd)
    piecewise_curve = simu.construct_piecewise_curve(pieces,divergences_xy_mm,piece_margin_mm)
    divergence_curves.append(frm.merge_curves(curve,piecewise_curve))

for i, div_curve in enumerate(divergence_curves):
    draw_curve(div_curve,ax=ax,label=f'Rod {i+1}',alpha = .5)

plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.ylabel("FA Height[mm]")
plt.xlabel("Bow Divergence[mm]")
plt.gca().invert_yaxis()
plt.subplots_adjust(right=0.6,top=.9)
plt.savefig(figs_out/'fig_fa_bow_div.png')

# create curve
```

```python
scene_dict = default_scene.copy()
for i, piecewise_curve in enumerate(divergence_curves):
    
    vertices,faces,uv = frm.get_rod_geometry(20,9,curve=piecewise_curve,radius= .2)
    rod_mesh = frm.get_rod_mesh(vertices,faces,uv)

    rod_mesh_path = io.save_mesh_to_temp(rod_mesh)

    test_tooth = {        
        'type': 'ply',
        'filename': rod_mesh_path,
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "mesh_attribute",
                "name": "face_color",
            },
        },
        "to_world": mi.ScalarTransform4f.translate([i - 2 ,0,2]).scale([1,1,.004]),
    }

    scene_dict[f'model_{i}'] = add_face_color(test_tooth)
    
scene_dict['sensor'] = scene.create_lookat_sensor([0,-10,0], type_='perspective',resolution_width=1920,resolution_height=1080)
rods_img = scene.render_scene(scene_dict)
```

```python
rods_img_crop = rods_img[:,200:-200]
plt.imshow(rods_img_crop)
plt.axis('off')
imageio.imwrite(
    figs_out/f"fig_rod_mesh_div.png",
    np.uint8(np.clip(0,1,rods_img_crop)*255)
)
```

```python
# rcs = frm.generate_rod_centers(11,9.1,3.65)
# plt.figure(figsize=(3.2,3.2))
# plt.plot(rcs[:,0],rcs[:,1],'o')
# plt.axis('off')
# plt.savefig(figs_out/f"fig_rod_centers_hex.png",)
```

## Spacer Grid

```python
import synthnf.geometry.spacer_grid_mesh as sgm

tooth_pth =  str(assets.get_asset_path('grid_wta.ply'))

scene_dict = default_scene.copy()

test_tooth = {        
    'type': 'ply',
    'filename': tooth_pth,
    "material": {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "face_color",
        },
    },
    "to_world": mi.ScalarTransform4f.rotate([0, 0, 1], angle=180).scale(.02),
}

scene_dict['model'] = add_face_color(test_tooth)
scene_dict['sensor'] = scene.create_lookat_sensor([0,1,0], type_='orthographic',resolution_height=1080,resolution_width=1920)

tooth_img = scene.render_scene(scene_dict)

## Mirror

tooth_ply = io.read_ply(tooth_pth)
p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = sgm.decompose_mesh(tooth_ply)
p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = sgm.mirror_down(p_x,p_y,p_z,v_1,v_2,v_3, n_x,n_y,n_z)
tooth_mirrored_mesh = sgm.compose_mesh(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z)

test_tooth['filename'] = io.save_mesh_to_temp(tooth_mirrored_mesh)
scene_dict['model']= add_face_color(test_tooth)

tooth_mirrored_img = scene.render_scene(scene_dict)

## Array


p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z = sgm.array_left(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z,n=5)
tooth_arrayed_mesh = sgm.compose_mesh(p_x,p_y,p_z,v_1,v_2,v_3,n_x,n_y,n_z)

test_tooth["to_world"] = mi.ScalarTransform4f.rotate([0, 0, 1], angle=180).scale(.02).translate([30,0,0])
test_tooth['filename'] = io.save_mesh_to_temp(tooth_arrayed_mesh)
scene_dict['model']= add_face_color(test_tooth)

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
tooth_ply = io.read_ply(tooth_pth)

grid_mesh = sgm.create_grid_mesh_from_tooth(tooth_ply,11,rod_width_mm=9.1,rod_gap_width_mm=3.65)

grid_mesh_path = io.save_mesh_to_temp(grid_mesh)

grid_scene = default_scene.copy()

grid_model_dict = {        
    'type': 'ply',
    'filename': grid_mesh_path,
    "material": {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "face_color",  # This will be used to visualize our attribute
        },
    },
    "to_world": mi.ScalarTransform4f.scale(.75),
}

test_grid_mesh = add_face_color(grid_model_dict)
grid_scene['model']= test_grid_mesh

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
    fig_path = figs_out/f"fig_grid_{name}.png"
    img = np.uint8(np.clip(img,0,1)*255)
    print("Saving Image to", fig_path.absolute())
    imageio.imwrite(fig_path,img)
    
    
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
import synthnf.material as mat
```

```python
import perlin_numpy
shape = np.array([100,100])
noise_low = perlin_numpy.generate_fractal_noise_2d(
    [128,128], 
    [2,2], 
    2,
    persistence=.5,
    lacunarity=2,
    tileable = (True,True)
)
noise_high = perlin_numpy.generate_fractal_noise_2d(
    [128,128], 
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
def compose_scene_rod_piece(
    cam_ligth_intensity = 20,
    light_height = 100,
    blend_ratio = .5
):
    scene_dict = default_scene.copy()

    # make sure it adds up to at most 1
    blend_texture = np.clip(0,1,-.1 + blend_ratio*1.1 + (_norm(noise)-.5)*.2)

    bsdf = mat.create_gray_diffuse(1)
    zircon_conductor = mat.create_zirconium()

    material=  mat.blend_materials(
        bsdf,
        zircon_conductor,
        blend_texture
    )

    test_cylinder_1 = {        
        'type': 'cylinder',
        'radius': 10,
        'p0':[0,0,-15],
        'p1':[0,0,15],
        'material': material
    }

    scene_dict['model']= test_cylinder_1
    scene_dict['sensor'] =scene.create_lookat_sensor([0,100,0],resolution_width = 1920,resolution_height  = 1080)
    scene_dict['light'] = scene.create_mount_emmiter([-20,100,0],cam_ligth_intensity,light_height_mm=light_height)
    scene_dict['light_1'] = scene.create_mount_emmiter([20,100,0],cam_ligth_intensity,light_height_mm=light_height)
    return scene_dict

rod_imgs = []
blends = np.flip(np.linspace(.4,1.2,5))
for i in blends:
    scene_dict = compose_scene_rod_piece(blend_ratio=i)
    scene_img = scene.render_scene(scene_dict)
    rod_imgs.append(scene_img[:,600:1300])
    
fig,axs = plt.subplots(1,len(rod_imgs),figsize=(4*len(rod_imgs),4))

for ax,img,i in zip(axs,rod_imgs,blends):
    ax.imshow(img)
    ax.set_title(f"Blend factor:{i:.2f}")
    
for img,i in zip(rod_imgs,blends):
    imageio.imwrite(figs_out/f'fig_rod_piece_{i}.png',np.uint8(np.clip(0,1,img)*255))
```

## Spacer Grid

```python

def compose_grid_scene(blend_ratio):
    grid_mat_scene = default_scene.copy()
    inconel = mat.create_inconel(alpha_u=.1,alpha_v=.4)
    bsdf = mat.create_gray_diffuse(.6)

    noise = perlin_numpy.generate_fractal_noise_2d(
        np.array([1,24])*128*4, 
        np.array([1,24]), 
        2,
        persistence=.5,
        lacunarity=2,
        tileable = (True,True)
    )

    blend_texture = np.clip(0,1,-.1 + blend_ratio*1.1 + (_norm(noise)-.5)*.2)
    material=  mat.blend_materials(
        bsdf,
        inconel,
        blend_texture
    )

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

    model_mat_grid = grid_model_dict.copy()
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
blends = np.flip(np.linspace(0.4,1.2,5))

for i in blends:
    scene_dict = compose_grid_scene(blend_ratio=i)
    scene_img = scene.render_scene(scene_dict)
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
    
for img,i in zip(pcs,blends):
    imageio.imwrite(figs_out/f'fig_grid_piece_{i}.png',np.uint8(np.clip(0,1,img)*255))

    
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

## Tubus as an inspection chamber

```python
# Add texture to flip-faced cyllinder
```

## Illumination

```python
# Global local
```

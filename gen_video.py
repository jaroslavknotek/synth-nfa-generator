import pandas as pd
import imageio
import cv2
import time 

from tqdm.auto import tqdm

from datetime import datetime

import numpy as np
import pandas as pd

import mitsuba as mi





mi.set_variant('cuda_ad_spectral')
print("Is spectral", mi.is_spectral)

from mitsuba import set_log_level,LogLevel
set_log_level(LogLevel.Error)

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
    
    samples_per_pass = 32
    
    #samples_per_pass = 128
    cam_light_intensity = 47100
    
    rods_bdsf = scene_definition.get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
    
    nfa_curve,rods_group = scene_definition.generate_rods_group(config, has_rod_divergence= has_rod_divergence, bsdf=rods_bdsf)

    grid_mod = config['grid_detection']["mods"][2]
    rod_height = np.sum(grid_mod['spacing_mm'])  
    camera_z_ration = 1-np.abs(camera_z/rod_height)
    camera_x,_,_ = nfa_curve.evaluate_multi(np.array([camera_z_ration]))
       
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
    
    mod3_spacing_mm = grid_mod['spacing_mm']
    np.cumsum(mod3_spacing_mm)
    add_grid(scene_dict,nfa_curve,rod_height, np.cumsum(mod3_spacing_mm),config)

    return scene_dict

def render_scene(scene_dict):

    scene = mi.load_dict(scene_dict)

    img = mi.render(scene)
    bitmap = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
    
    arr_bitmap= np.array(bitmap)
    # clipping
    # arr_bitmap[arr_bitmap>1] = 1
    return arr_bitmap


def add_grid(scene_dict, nfa_curve, nfa_height, grid_positions, config):
    
    
    grid_relative_height = 1-np.abs(grid_positions/nfa_height)
    grid_curved_pos = nfa_curve.evaluate_multi(grid_relative_height)
    grid_xs = grid_curved_pos[0]
    
    
    grid_x_above,_,grid_z_above = nfa_curve.evaluate_multi(grid_relative_height - .01)
    grid_x_below,_,grid_z_below = nfa_curve.evaluate_multi(grid_relative_height + .01)
    
    dist_x = grid_x_above - grid_x_below
    dist_y = grid_z_above - grid_z_below
    
    angles = np.rad2deg(np.arctan(dist_x/dist_y))
    
    
    
    
    
    grid_material = config['fuel_material']['grids']
    alpha_u = grid_material['material_alpha_u']
    alpha_v = grid_material['material_alpha_v']
    gray_intensity =grid_material['diffuse_gray_intensity']
    bsdf_blend_factor = grid_material['metal_to_bsdf_blend_factor']
    bsdf_blend_factor = .6
    grid_material = scene_definition.get_grid_bsdf(gray_intensity,bsdf_blend_factor,alpha_u,alpha_v)

    for (grid_z,grid_x,a) in zip(grid_positions,grid_xs,angles):    
        grid = {
            "type": "obj",
            "filename": "assets/wta-grid.obj",
            "to_world": mi.ScalarTransform4f.translate([grid_x,118,-grid_z]).scale(93).rotate([1, 0, 0], angle=-90).rotate([0,0,1],angle=-a),
            "material":grid_material
        }
        scene_dict[f'grid_obj_{grid_z}'] = grid

def get_frames(
    frames_count,
    frame_width,
    frame_height,
    is_top_down = True,
    has_rod_divergence = False,
    seed = 456):
    # Video frame scrop
    frame_random_state = np.random.RandomState(seed)
    fov = 28
    
    # TODO calc the param automatically
    render_intensity_factor = 2 # scales intensities so the output looks as the ref video 
    
    mod3_spacing_mm = grid_detection["mods"][-1]['spacing_mm']
    grid_positions = np.cumsum(mod3_spacing_mm)

    from_i,to_i = 0,grid_positions[-1]
    
    frames_nums = np.linspace(from_i,to_i,frames_count).astype(int)
    if not is_top_down:
        frames_nums = np.flip(frames_nums)
    
    for i in frames_nums:
        camera_z = -i
        scene_dict = get_nfa_scene_dict(
            config,
            fov,
            frame_height,
            frame_width,
            camera_z,
            has_rod_divergence=has_rod_divergence)
        
        scene = mi.load_dict(scene_dict)
        
        render_random = frame_random_state.randint(0,high =  2**31 - 1)
        img = mi.render(scene,seed = render_random )
        bitmap = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
        bitmap_arr = np.array(bitmap) *render_intensity_factor
        bitmap_arr[bitmap_arr>1] = 1
                
        yield i, bitmap_arr
        

        
def _get_sips(n_frames,top_down, top_sips = 4000,bottom_sips = 400 ):
    frame_nums = np.arange(n_frames)
    fake_sips = np.int32(np.linspace(top_sips,bottom_sips, n_frames))
    probs = np.ones((n_frames),dtype=int)

    if top_down:    
        sips_data = np.vstack([frame_nums, fake_sips, probs])
    else:
        sips_data = np.vstack([
            np.flip(frame_nums), 
            fake_sips,
            probs])
        sips_data = np.flip(sips_data,axis=1)
        
    return pd.DataFrame(
        sips_data.T,
        columns=['frame_number','sips_number','probability'])



def create_video(
    vid_dir,
    frame_number,
    frame_width,
    frame_height,
    fps = 25,
    has_rod_divergence=False,
    seed = 456):
    # disabled because I don't have the right coded on my local
    #fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = vid_dir/"video.mp4"
    
    top_down = seed%2==0
    frames = get_frames(
        frame_number,
        frame_width,
        frame_height,
        is_top_down = top_down,
        has_rod_divergence=has_rod_divergence,
        seed = seed)
    
    frames = tqdm(frames,desc ='generating frames',total=frame_number)
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (frame_width,frame_height))
    for i,frame in frames:
        f = (frame *255).astype(np.uint8) 
        writer.write(f)
        
    sips_df = _get_sips(frame_number, top_down)
    sips_df.to_csv(vid_dir/"sips.csv", index=False,header=True)
    
    

    writer.release()



import os
import pathlib
import shutil

output_video_dir = pathlib.Path("output")
os.makedirs(output_video_dir, exist_ok = True)
dt = datetime.fromtimestamp(time.time())
timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S').replace(':','-')
for i in range(6):
    
    assert i<6
    video_dir = output_video_dir/timestamp/"JKSY"/f"F{i+1}"
    video_dir.mkdir(parents=True,exist_ok=True)
    
    fps = 25
    frames_count = 1000
    create_video(
        video_dir,
        frames_count, 
        video_config['width'],
        video_config['height'],
        fps=fps,
        has_rod_divergence=True,
        seed = i)


print('done')

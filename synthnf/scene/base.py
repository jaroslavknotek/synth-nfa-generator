import mitsuba as mi
import numpy as np

import synthnf.config.defaults as defaults
import synthnf.scene as scene

trans_identity = mi.ScalarTransform4f.translate([0,0,0])

def render_scene(scene_dict,spp = 128,seed =789,clip = False,alpha = True,denoise = False):
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene,spp = spp,seed = seed)
    
    if alpha:
        mode = mi.Bitmap.PixelFormat.RGBA
    else:
        mode = mi.Bitmap.PixelFormat.RGB
        
    if denoise:
        h,w = img.shape[:2]
        denoiser = mi.OptixDenoiser(
            input_size=[w,h], 
            albedo=False, 
            normals=False, 
            temporal=False
        )
        
        img = denoiser(img)
        
    
        
    bitmap = mi.Bitmap(img).convert(
        mode, 
        mi.Struct.Type.Float32, 
        srgb_gamma=True
    )

    bitmap = np.array(bitmap)
    if clip:
        bitmap = np.clip(0,1,bitmap)
        
    return bitmap

def global_light(intensity = None):
    intensity = intensity if intensity is not None else 1
    
    return {
        "type": "constant",
        'radiance': {
            'type': 'rgb',
            'value': intensity
        }
     }

def _transform(model,transformation):
    model['to_world'] = transformation @model.get('to_world',trans_identity)
    
def inspection_dict(
    integrator = None,
    light_global = None,
    cam_lights = True,
    light_offset_mm = None,
    light_height_mm = None,
    cam_ligth_intensity = None,
    cam_distance_mm = None,
    camera = None,
    cam_z = None,
    cam_res_x = None,
    cam_res_y = None,
    face_num = 1
):
    assert 1 <=face_num <= 6,f"The parameter {fuel_side_num=} must be a number between 1 and 6 inc."
    
    light_offset_mm = light_offset_mm or defaults.blueprints.camera_ring.light_offset_mm
    cam_distance_mm = cam_distance_mm or defaults.blueprints.camera_ring.diameter_mm/2
    light_height_mm = light_height_mm or defaults.blueprints.camera_ring.light_height_mm
    cam_ligth_intensity = cam_ligth_intensity or defaults.scene_params.illumination.cam_light_intensity
    
    cam_z = cam_z if cam_z is not None else 0 
    
    cam_res_x = cam_res_x or defaults.scene_params.camera.resolution_width
    cam_res_y = cam_res_y or defaults.scene_params.camera.resolution_height

    
    camera = camera or scene.cam_perspective_lookat(
        [0,cam_distance_mm,cam_z],
        target=[0,0,cam_z],
        res_x = cam_res_x,
        res_y = cam_res_y
    )
    
    integrator = integrator or {"type": "path"}
    
    scene_dict = {
        "type": "scene",
        "integrator": integrator,
        "sensor":camera,
    }
    
    if light_global:
        scene_dict['light_global'] = light_global
        
    if cam_lights:
        scene_dict['light'] = scene.create_mount_emmiter(
            [light_offset_mm/2,cam_distance_mm,cam_z],
            cam_ligth_intensity , 
            light_height_mm=light_height_mm
        )
        scene_dict['light_1'] = scene.create_mount_emmiter(
            [-light_offset_mm/2,cam_distance_mm,cam_z],
            cam_ligth_intensity , 
            light_height_mm=light_height_mm
        )    
    
    # clockwise
    trans_rotate = mi.ScalarTransform4f.rotate([0,0,1],-(face_num-1)*60)
    for obj_key in ['sensor','light_1','light']:
        _transform(scene_dict.get(obj_key,{}),trans_rotate)
    
    
    return scene_dict

def shrunk_dict(
    curve_fa,
    res_y = 2048,
    res_x = 512,
    orto_x = 200,
    orto_y = 500,
    cam_distance = 1000,
    face_num = 1,
):
    fa_center = .5
    cam_x,cam_y,cam_z=curve_fa.evaluate(fa_center)
    orto_cam = scene.cam_orto_lookat(
        [0, cam_distance, cam_z],
        target = [cam_x,cam_y,cam_z],
        res_y = res_y,
        res_x = res_x,
        orto_scale_xy = [orto_x, orto_y ]
    )
    
    # clockwise
    trans_rotate = mi.ScalarTransform4f.rotate([0,0,1],-(face_num-1)*60)
    _transform(orto_cam,trans_rotate)
    
    
    return {
        "type": "scene",
        "integrator": {"type": "path"},
        "sensor":orto_cam,
        "light": {
            "type": "constant",
            'radiance': {
                'type': 'rgb',
                'value': 1
            }
         },
    }
    
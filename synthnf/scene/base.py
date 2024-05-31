import mitsuba as mi
import numpy as np

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
import mitsuba as mi
import numpy as np

def render_scene(scene_dict,spp = 128,seed =789,clip = False):
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene,spp = spp,seed = seed)
    bitmap = mi.Bitmap(img).convert(
        mi.Bitmap.PixelFormat.RGBA, 
        mi.Struct.Type.Float32, 
        srgb_gamma=True
    )

    bitmap = np.array(bitmap)
    if clip:
        bitmap = np.clip(bitmap)
        
    return bitmap
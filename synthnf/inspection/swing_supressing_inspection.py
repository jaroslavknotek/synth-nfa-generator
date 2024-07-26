from .base_inspection import *

class SwingSupressingInspection(FAInspection):
    
    def __init__(self, params,swing_camera_relative_mm):
        super(self.__class__, self).__init__(params)
        self.swing_camera_relative_mm = swing_camera_relative_mm
        
    
    
    def render_frame(self,cam_z,face_num = 1,spp = 128):
        
        scene_dict,model_keys = self._prepare_scene_dict(cam_z,face_num)
        t = cam_z/self.fa_height_mm
        swing_deg_xz,swing_deg_xy = self.simulation.pendulum_swing_in_time(t)
        self._swing_fa(scene_dict,model_keys,swing_deg_xz,swing_deg_xy)
        
        frame_bottom = scene.render_scene(
            scene_dict,
            spp = spp,
            alpha = False,
            denoise=True
        )
        
        
        scene_dict,model_keys = self._prepare_scene_dict(
            cam_z + self.swing_camera_relative_mm,
            face_num
        )
        # same swing but does not depend on camera
        self._swing_fa(scene_dict,model_keys,swing_deg_xz,swing_deg_xy)        
        frame_top = scene.render_scene(
            scene_dict,
            spp = spp,
            alpha = False,
            denoise=True
        )
        
        return np.vstack([frame_bottom,frame_top])
        
        
    def collect_metadata(self,n_frames,**kwargs):
        
        meta = super(self.__class__, self).collect_metadata(n_frames,**kwargs)
        
        ts = np.linspace(0,1,n_frames)
        
        swing_deg_xz,swing_deg_xy = self.simulation.pendulum_swing_in_time(ts)
        meta['swing_xz'] = swing_deg_xz
        meta['swing_xy'] =swing_deg_xy
        return meta
        
    
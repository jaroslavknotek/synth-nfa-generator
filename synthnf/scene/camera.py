import synthnf.config.defaults as defaults
import mitsuba as mi

def create_mount_emmiter(
    location,
    cam_light_intensity,
    light_height_mm = 10, 
    radius_mm = 1
):
    x,y,z = location
    return {
        'type': 'cylinder',
        'radius':radius_mm,
        'p0':[x,y,z+light_height_mm/2],
        'p1':[x,y,z-light_height_mm/2],
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': cam_light_intensity,
            }
        }
    }

def create_lookat_sensor(
    origin, 
    target = None,
    up=None,
    type_= None,
    resolution_width = None,
    resolution_height=None
):
    if target is None:
        target = [0,0,0]
    if up is None:
        up = [0,0,1]
    if resolution_width is None:
        resolution_width = 600
    if resolution_height is None:
        resolution_height = 800
    if type_ is None:
        type_ = 'perspective'
    
    return {
        "type": type_,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=origin, 
            target=target, 
            up=up
        ),
        "film":{
            'type': 'hdrfilm',
            'pixel_format': 'rgba',
            'width': resolution_width,
            'height': resolution_height,
            'rfilter':{
                "type":"box"
            }
        }
    }

def create_camera_ring(
    ring_center,
    ring_diameter_mm = None,
    light_offset_mm = None,
    light_height_mm = None,
    cam_light_intensity = None
):
    bl = defaults.blueprints
    sc = defaults.scene_params
    if ring_diameter_mm is None:
        ring_diameter_mm = bl.camera_ring.diameter_mm
    if light_height_mm is None:
        light_height_mm = bl.camera_ring.light_height_mm
    if light_offset_mm is None:
        light_offset_mm = bl.camera_ring.light_offset_mm
    if light_radius_mm is None:
        light_radius_mm = bl.camera_ring.light_radius_mm
    if cam_light_intensity is None:
        cam_light_intensity = sc.illumination.cam_light_intensity
    
    c_x,c_y,c_z = ring_center
    sensor = create_lookat_sensor(
        [c_x,c_y + ring_diameter_mm/2,c_z],
        target=ring_center
    )

    ligth_left = create_mount_emmiter(
        [c_x - light_offset_mm//2,c_y + ring_diameter_mm,c_z],
        cam_ligth_intensity , 
        light_height=light_height,
        radius = light_radius_mm
    )
    ligth_left = create_mount_emmiter(
        [c_x - light_offset_mm//2,c_y + ring_diameter_mm,c_z],
        cam_ligth_intensity , 
        light_height=light_height,
        radius = light_radius_mm
    )
        
    return {
        "sensor":sensor,
        "light_left":light_left,
        "light_right":light_right
    }
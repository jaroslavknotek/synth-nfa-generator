from types import SimpleNamespace

class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)
                
materials = NestedNamespace({
    "inconel":{
        "alpha_u":.1,
        "alpha_v":.4,
    },
    "zirconium":{
        "alpha_u":.03,
        "alpha_v":.05,
    },
    
})
                
blueprints = NestedNamespace({
    "camera_ring":{
        "diameter_mm": 860,
        "light_height_mm": 200,
        "light_diameter_mm": 10,
        "light_offset_mm": 228,
        
    },
    "fuel_rod":{
        "width_mm":9.1,
        "gap_mm":3.65,
        "height_mm":3800
    },
    "fa":{
        "rods_per_face":11
    }
})

scene_params = NestedNamespace({
    "illumination":{
        "cam_light_intensity":1000
    },
    "textures_per_fuel_rods":20
})

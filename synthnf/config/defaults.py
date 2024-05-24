from types import SimpleNamespace

class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)
                
      
                
blueprints = NestedNamespace({
    "camera_ring":{
        "diameter_mm": 860,
        "light_height_mm": 200,
        "light_diameter_mm": 10,
        "light_offset_mm": 228,
        
    }
})

scene_params = NestedNamespace({
    "illumination":{
        "cam_light_intensity":1000
    }
})

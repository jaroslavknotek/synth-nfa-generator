import pandas as pd
import synthnf.io as io
import synthnf.config.assets as assets

def create_gray_diffuse(intensity):
    return {
        "type" : "diffuse",
        "reflectance" : {
            "type" : "spectrum",
            "value" : intensity,
        }
    }

def create_zirconium(alpha_u = .03,alpha_v = .05):
    df_zirconium = pd.read_csv(assets.get_asset_path('spectrum_zirconium.csv'))
    wv = df_zirconium['wavelength'] * 1000
    n = df_zirconium['n']
    eta = df_zirconium['k']
    eta_spectrum = list(zip(wv,n))
    k_spectrum = list(zip(wv,eta))
    return create_roughconductor(
        eta_spectrum,
        k_spectrum,
        alpha_u,
        alpha_v
    )
    
def create_inconel(alpha_u = .02,alpha_v = .02):
    df_zirconium = pd.read_csv(assets.get_asset_path('spectrum_inconel.csv'))
    wv = df_zirconium['wavelength'] * 1000
    n = df_zirconium['n']
    eta = df_zirconium['k']
    eta_spectrum = list(zip(wv,n))
    k_spectrum = list(zip(wv,eta))
    return create_roughconductor(
        eta_spectrum,
        k_spectrum,
        alpha_u,
        alpha_v
    )

def blend_materials(mat_1,mat_2,texture = None, float_weight= None):
    blend_dict = {
        "type": "blendbsdf",
        "bsdf_0":mat_1,
        "bsdf_1":mat_2
    }
    if texture is not None:
        blend_path = io.save_float_image_to_temp(texture)
        blend_dict['weight'] = {
            "type":"bitmap",
            "filename":str(blend_path)
        }
        return blend_dict
    
    if float_weight is None:
        float_weight = .5
        
    blend_dict['weight'] = {
        "type":"spectrum",
        "value":float_weight
    }
        
    return blend_dict
    
def create_roughconductor(eta_spectrum,k_spectrum,alpha_u,alpha_v):
    return {
        "type" : "roughconductor",
        "eta": {
            "type":"spectrum",
            "value":eta_spectrum,
        },
        "k":{
            "type":"spectrum",
            "value":k_spectrum,
        },
        "distribution":"ggx",            
        "alpha_u":alpha_u,
        "alpha_v":alpha_v,
        "sample_visible":False
    }
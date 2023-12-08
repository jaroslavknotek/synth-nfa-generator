import numpy as np
import mitsuba as mi
import rods_hexagon
import bent_rod

_pink_bsdf = {
    "type" : "diffuse",
    "reflectance" : {
        "type" : "rgb",
        "value" : [1, 0, 1],
    }
}

def get_zirconium_data():
    #wv eta k
    zr = """
    0.00236	0.99783	0.00117
    0.00316	0.99781	0.00135
    0.00447	0.99536	0.00498
    0.00676	0.9922	0.00269
    0.0114	0.9694	0.00625
    0.01355	0.955	0.0131
    0.01714	0.9188	0.022
    0.0243	0.808	0.0987
    0.02563	0.79	0.119
    0.03038	0.757	0.324
    0.0327	0.868	0.32
    0.03568	0.949	0.373
    0.03793	0.963	0.318
    0.04059	0.9	0.232
    0.04478	0.866	0.314
    0.04607	0.829	0.321
    0.04895	0.844	0.347
    0.05391	0.825	0.337
    0.05843	0.838	0.362
    0.06163	0.826	0.33
    0.06714	0.768	0.332
    0.07185	0.807	0.44
    0.07359	0.789	0.492
    0.07437	0.817	0.533
    0.08345	0.817	0.673
    0.0878	0.828	0.759
    0.09198	0.81	0.858
    0.09321	0.804	0.848
    0.09888	0.814	0.905
    0.10258	0.895	0.99
    0.10482	0.952	1.1
    0.10667	0.85	1.08
    0.10857	0.907	1.11
    0.1135	1.15	1.27
    0.12007	1.14	1.24
    0.12157	1.28	1.31
    """

    # https://refractiveindex.info/?shelf=main&book=Zr&page=Windt
    zirconium_data = np.array(zr.split(),dtype=float).reshape((-1,3))
    zirconium_data[:,0] = zirconium_data[:,0]*10000
    return zirconium_data[9:]
    
def get_inconel_data():
    
    # https://doi.org/10.1364/JOSA.63.000185
    
    # Wavelength d=22043 A d=373d2 A d=625343 A
    # (am) n k P(0) T(0) n k R(0) T(0) n k R(0) T(0)
    values = """0.3311 1.89 1.94 0.372 0.136 1.59 1.89 0.408 0.057 1.61 2.15 0.440 0.0057
0.3425 1.78 2.02 0.382 0.150 1.59 1.96 0.422 0.063 1.60 2.23 0.459 0.0059
0.3546 1.73 2.12 0.391 0.158 1.58 2.03 0.433 0.066 1.57 2.30 0.477 0.0063
0.3676 1.73 2.20 0.399 0.159 1.58 2.10 0.444 0.0675 1.52 2.36 0.495 0.0068
0.3817 1.71 2.31 0.410 0.157 1.59 2.17 0.455 0.067 1.50 2.45 0.515 0.0068
0.3968 1.71 2.41 0.421 0.155 1.64 2.24 0.465 0.067 1.52 2.54 0.530 0.0068
0.4132 1.71 2.52 0.432 0.151 1.67 2.32 0.474 0.0665 1.52 2.63 0.547 0.0068
0.4310 1.75 2.64 0.442 0.146 1.74 2.40 0.482 0.0655 1.54 2.73 0.563 0.0068
0.4505 1.78 2.76 0.453 0.141 1.79 2.48 0.491 0.0645 1.58 2.84 0.576 0.0066
0.4717 1.82 2.89 0.464 0.137 1.87 2.57 0.500 0.0635 1.63 2.96 0.588 0.0066
0.4950 1.87 3.03 0.474 0.133 1.93 2.67 0.509 0.063 1.65 3.08 0.604 0.0066
0.5208 1.92 3.18 0.485 0.128 2.00 2.77 0.518 0.0625 1.70 3.21 0.617 0.0066
0.5495 1.97 3.33 0.495 0.124 2.10 2.88 0.527 0.062 1.75 3.35 0.632 0.0066
0.5814 2.03 3.49 0.505 0.121 2.15 2.99 0.536 0.062 1.79 3.49 0.644 0.0068
0.6173 2.07 3.67 0.515 0.119 2.21 3.11 0.544 0.0625 1.79 3.62 0.659 0.0075
0.6579 2.14 3.87 0.524 0.116 2.26 3.25 0.553 0.063 1.84 3.79 0.673 0.0078
0.7042 2.20 4.08 0.534 0.114 2.30 3.41 0.562 0.0635 1.90 3.98 0.686 0.0080
0.7576 2.32 4.28 0.539 0.112 2.29 3.57 0.571 0.065 2.02 4.21 0.699 0.0076
0.8197 2.41 4.53 0.547 0.109 2.34 3.77 0.580 0.066 2.12 4.47 0.713 0.0076
0.8929 2.51 4.81 0.556 0.107 2.42 3.98 0.588 0.066 2.21 4.73 0.727 0.0078
0.9803 2.66 5.13 0.564 0.103 2.49 4.23 0.597 0.067 2.29 5.03 0.740 0.0083
1.087 2.86 5.48 0.572 0.099 2.65 4.51 0.604 0.066 2.42 5.37 0.753 0.0087
1.219 3.12 5.90 0.579 0.096 2.80 4.84 0.612 0.066 2.59 5.78 0.765 0.0091
1.389 3.40 6.40 0.588 0.092 2.99 5.24 0.620 0.066 2.75 6.26 0.778 0.0097
1.613 3.72 7.02 0.596 0.089 3.24 5.73 0.628 0.066 2.94 6.87 0.792 0.0103
1.923 4.19 7.78 0.603 0.087 3.55 6.35 0.635 0.065 3.40 7.56 0.797 0.0112
"""
    inconel_data = np.array(values.split(),dtype=float).reshape((-1,13))
    inconel_data[:,0] = inconel_data[:,0]*1000
    return np.stack([inconel_data[:,0],inconel_data[:,5],inconel_data[:,6]]).T



def get_gray_diffuse(intensity):
    return {
        "type" : "diffuse",
        "reflectance" : {
            "type" : "spectrum",
            "value" : intensity,
        }
    }

def get_grid_bsdf(gray_intensity,bsdf_blend_factor,material_alpha_u,material_alpha_v,  spectral_params = None):

    spectral_params = spectral_params if spectral_params is not None else get_inconel_data()
    
    eta_spectrum = list(zip(spectral_params[:,0],spectral_params[:,1]))
    k_spectrum = list(zip(spectral_params[:,0],spectral_params[:,2]))
    
    bsdf = get_gray_diffuse(gray_intensity)
    return {
        "type": "blendbsdf",
        "id":"grids_material",
        "weight":{
            "type":"spectrum",
            "value":bsdf_blend_factor,
        },
        "bsdf":bsdf,
        "metal": {
        "type" : "roughconductor",
            "eta": {
                "type":"spectrum",
                "value":eta_spectrum,
            },
            "k":{
                "type":"spectrum",
                "value":k_spectrum,
            },
            #"distribution":"ggx",            
            "alpha_u":material_alpha_u,
            "alpha_v":material_alpha_v,
        }
    }

def get_rods_bsdf(gray_intensity,bsdf_blend_factor,material_alpha_u,material_alpha_v,  spectral_params = None):

    spectral_params = spectral_params if spectral_params is not None else get_zirconium_data()
    
    eta_spectrum = list(zip(spectral_params[:,0],spectral_params[:,1]))
    k_spectrum = list(zip(spectral_params[:,0],spectral_params[:,2]))
    
    bsdf = get_gray_diffuse(gray_intensity)
    return {
        "type": "blendbsdf",
        "id":"rods_material",
        "weight":{
            "type":"spectrum",
            "value":bsdf_blend_factor,
        },
        "bsdf":bsdf,
        "metal": {
        "type" : "roughconductor",
            "eta": {
                "type":"spectrum",
                "value":eta_spectrum,
            },
            "k":{
                "type":"spectrum",
                "value":k_spectrum,
            },
            #"distribution":"ggx",            
            "alpha_u":material_alpha_u,
            "alpha_v":material_alpha_v,
        }
    }

def get_ring_light(x,y,z, cam_light_intensity):
    return {
        "type":"point",
        "intensity":{
            "type":"rgb",
            "value":cam_light_intensity,},
        "position":[x,y,z]
    }

def get_ring_camera(width,height,x,y,z,fov,spp=4):
  
    return {
        "type" : "perspective",
        "fov":fov,
        "fov_axis":"smaller",
        "near_clip": 0.001,
        "far_clip": 10000.0,
        "to_world" :mi.ScalarTransform4f.translate([x,y,z]).rotate([1,0,0],90),
        "myfilm" : {
            "type" : "hdrfilm",
            "rfilter" : { "type" : "tent"},
            "width" : width,
            "height" : height,
            "pixel_format": "rgb",
            "component_format":"float32"
        },
        "mysampler" : {
            "type" : "independent",
            "sample_count" : spp,
        },
    }

def get_rod(rod_center, radius, rod_height = 1):
    (rc_x,rc_y, rc_z) = rod_center     
    return {
        "type":"cylinder",
        "radius":radius,
        "p0":[rc_x,rc_y,rc_z ],
        "p1":[rc_x,rc_y,rc_z-rod_height]
    }


def generate_rods_group(config, max_twist_bow_mm = 50, max_divergence_mm = 5, bsdf=None, rod_count = 11):    
    
    spacing= config['grid_detection']["mods"][-1]['spacing_mm']
    grid_heights_mm= config['grid_detection']["mods"][-1]['grid_heights_mm']

    rod_height = np.sum(spacing)                    
    bsdf_resolved  = bsdf if bsdf is not None else _pink_bsdf    
    
    rod_width_mm = config['measurements']['rod_width_mm']
    gap_between_rods_width_mm = config['measurements']['gap_between_rods_width_mm']
    

    rod_centers = rods_hexagon.generate_rod_centers(rod_count,rod_width_mm, gap_between_rods_width_mm)
    
    if max_twist_bow_mm != 0 or max_twist_bow_mm != 0:
        return bent_rod.get_twisted_nfa_mesh(config,rod_centers,spacing,grid_heights_mm,max_divergence_mm, bsdf_resolved, max_twist_bow_mm=max_twist_bow_mm,return_curve=True)
    else:
        cylinder_radius = config['measurements']['rod_width_mm']/2
        rods = [ get_rod((*rc,0),cylinder_radius,rod_height) for rc in rod_centers]

        rod_objs = []
        for i, rod in enumerate(rods[:]):
            rod['my_bsdf'] = bsdf_resolved
            rod_objs.append(rod)

        return rod_objs
    
def get_tips(curves, config,tip_ply = 'assets/tip_model.ply'):
    rod_tops = np.array([c.evaluate(0) for c in curves])
    tips_scenes = {}
    
    rods_material = config['fuel_material']['rods']
    rods_material_alpha_u = rods_material['material_alpha_u']
    rods_material_alpha_v = rods_material['material_alpha_v']
    rods_gray_intensity =rods_material['diffuse_gray_intensity']
    rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']

    for i,(x,y,_) in enumerate(rod_tops):
        rods_bdsf = get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
        rods_bdsf['id'] = f'tip_material_{i}'
        offset = 9.1 + 3.65
        y_offset = np.sqrt(offset**2 - (offset/2)**2)
        y_shift =  offset - y_offset *12
        tips_scenes[f"tip_{i}"] = {        
            'type': 'ply',
            'filename': tip_ply,
            "to_world" :mi.ScalarTransform4f.translate([x,y+y_shift,0]),

            "material":rods_bdsf,
            # "material": {
            #     'type': 'diffuse',
            #     'reflectance': {
            #         'type': 'rgb',
            #         'value': [1, 0, 1]
            #     }
            # }
        }
    return tips_scenes
# def generate_rods_group(config, bsdf=None):
#     rod_count = 11
#     rod_height = 4500
    
#     bsdf_resolved  = bsdf if bsdf is not None else _pink_bsdf
    
#     cylinder_radius = config['measurements']['rod_width_mm']/2
#     rod_centers = rods_hexagon.generate_rod_centers(rod_count,config)
#     rods = [ get_rod((*rc,0),cylinder_radius,rod_height) for rc in rod_centers]
    
#     rods_obj = []
#     for i, rod in enumerate(rods[:]):
#         rod['my_bsdf'] = bsdf_resolved
#         rods_obj.append(rod)
        
#     return rods_obj


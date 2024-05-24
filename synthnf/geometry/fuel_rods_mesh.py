import numpy as np
import mitsuba as mi

class PiecewiseCurve():
    def __init__(self, curve_records):
        self.nfa_height=curve_records[-1][1]
        self.curve_records = curve_records
    
    def evaluate_multi(self,zs):
        found_xyz = []
        for z in zs:
            last_end = None
            last_end_point = None
            z = z*self.nfa_height
            
            for i,(start,end,curve) in enumerate(self.curve_records):
                if z < start:
                    start_point = curve.evaluate_multi(np.array([0.0])).T[0]
                    alpha = (last_end - z)/(start - z)
                    
                    x,y,_ =  last_end_point*alpha + start_point*(1-alpha)
                    found_xyz.append([x,y,z])
                    break
                elif z > end:
                    last_end_point = curve.evaluate_multi(np.array([1.0])).T[0]
                    last_end = end
                    continue
                else:
                    redone_point = (z - start)/(end-start)                    
                    x,y,_ = curve.evaluate_multi(np.array([redone_point])).T[0]
                    found_xyz.append([x,y,z])
                    break 
        
        return np.stack(found_xyz).T
    
    def evaluate(self,z):
        return self.evaluate_multi([z])

class CurveAdded():
    def __init__(self, curve_a,curve_b):
        self.curve_a = curve_a
        self.curve_b = curve_b

    def evaluate_multi(self,zs):
        
        a = self.curve_a.evaluate_multi(zs)
        b = self.curve_b.evaluate_multi(zs)
                
        res = a+b
        # z should remain the same)
        res[2] = b[2]
        return res
        
    def evaluate(self,z):
        return self.evaluate_multi([z])
    
def merge_curves(curve_a,curve_b):
    return CurveAdded(curve_a,curve_b)


def get_rod_mesh(vertex_pos, face_indices,uv,mesh_name = None):
    mesh = mi.Mesh(
        mesh_name or "nfa_rods_twisted",
        vertex_count=len(vertex_pos),
        face_count=len(face_indices),
        has_vertex_normals=False,
        has_vertex_texcoords=True,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = np.ravel(vertex_pos)
    mesh_params["faces"] = np.ravel(face_indices)
    mesh_params['vertex_texcoords'] = np.ravel(uv)
    
    return mesh

def get_rod_geometry(rod_segments, cylinder_faces, radius = 1,curve = None,height = None):
    actual_segments = rod_segments +1
    curve_eval_points = np.linspace(0,1,actual_segments)
    
    if curve is None:
        c_x = np.zeros((actual_segments,))
        c_y = np.copy(c_x)
        c_z = curve_eval_points*(height or 1)
    else:
        c_x,c_y,c_z = curve.evaluate_multi(curve_eval_points)
        
    x = c_x
    y = c_y
    z = c_z
    
    spacing = np.linspace(0,2*np.pi,cylinder_faces)
    x_circle = np.sin(spacing)* radius 
    y_circle = np.cos(spacing)* radius
    z = -z
    
    cylinder_coor_shape = (actual_segments,len(x_circle))
    cylinder_x = np.broadcast_to(x_circle ,cylinder_coor_shape) + np.broadcast_to(x[:,np.newaxis], cylinder_coor_shape)
    cylinder_y = np.broadcast_to(y_circle ,cylinder_coor_shape) + np.broadcast_to(y[:,np.newaxis], cylinder_coor_shape)
    cylinder_z = np.broadcast_to(z[:,np.newaxis],cylinder_coor_shape )
    grid_mesh = np.concatenate([
        cylinder_x[:,:,np.newaxis],
        cylinder_y[:,:,np.newaxis],
        cylinder_z[:,:,np.newaxis]],
        axis=2)
    vertex_pos = grid_mesh.reshape((-1,3))
    face_indices = []
    for i in range(actual_segments-1):
        for j in range(cylinder_faces-1):
            # a -- b
            # | -- |
            # c -- d
            idxs = np.array([(i,j),(i,j+1),(i+1,j),(i+1,j+1)]).T
            a,b,c,d= np.ravel_multi_index(idxs,(actual_segments,cylinder_faces))

            face_indices.append([a,b,c])
            face_indices.append([c,b,d])
            # inward faces
            # face_indices.append([a,c,b])
            # face_indices.append([c,d,b])
        
    # Texture coors
    y_coors = (np.linspace(0,1,actual_segments))
    uv_y = np.multiply(np.expand_dims(y_coors,axis=1),np.ones(cylinder_faces)[None]).flatten()
    uv_x = np.concatenate([np.linspace(0,1,cylinder_faces)]*(actual_segments))
    uv = np.vstack([uv_x,uv_y]).T    
    
    vertices = np.array(vertex_pos)
    faces  =np.array(face_indices,dtype=np.int32)
    return vertices,faces , uv


def add_row(start_x,start_y, n,rod_center_distance_mm):
    pts_x = np.arange(n) * rod_center_distance_mm + start_x
    pts_y = np.ones((n,)) * start_y
    return np.stack([pts_x,pts_y]).T


def generate_rod_centers(side_rod_count, rod_width_mm,gap_between_rods_width_mm):
    rod_center_distance_mm = rod_width_mm + gap_between_rods_width_mm

    pythagorean_factor = np.sqrt(3)/2
    rod_centers_lines_arrays = [add_row(0,0,side_rod_count*2 - 1,rod_center_distance_mm) ]
    x_shift = rod_centers_lines_arrays[0][side_rod_count-1]
        
    for i,current_rod_count in zip(range(1,side_rod_count),range(side_rod_count+side_rod_count - 2,side_rod_count-1,-1)):

        x = (i*rod_center_distance_mm)/2
        y = i * pythagorean_factor* rod_center_distance_mm
        points_b = add_row(x,-y,current_rod_count,rod_center_distance_mm)
        points_t = add_row(x,y,current_rod_count,rod_center_distance_mm)

        rod_centers_lines_arrays.append(points_b)
        rod_centers_lines_arrays.append(points_t)
    return np.concatenate(rod_centers_lines_arrays) - x_shift


# def generate_rods_group(
#     config, 
#     max_twist_bow_mm = 50, 
#     max_divergence_mm = 5, 
#     bsdf=None, 
#     z_displacement = None,
#     rod_count = 11
# ):    
    
#     spacing= config['grid_detection']["mods"][-1]['spacing_mm']
#     grid_heights_mm= config['grid_detection']["mods"][-1]['grid_heights_mm']

#     rod_height = np.sum(spacing)                    
#     bsdf_resolved  = bsdf if bsdf is not None else _pink_bsdf    
    
#     rod_width_mm = config['measurements']['rod_width_mm']
#     gap_between_rods_width_mm = config['measurements']['gap_between_rods_width_mm']
    

#     rod_centers = rods_hexagon.generate_rod_centers(rod_count,rod_width_mm, gap_between_rods_width_mm)
    
#     if max_twist_bow_mm != 0 or max_twist_bow_mm != 0:
#         return bent_rod.get_twisted_nfa_mesh(
#             config,
#             rod_centers,
#             spacing,
#             grid_heights_mm,
#             max_divergence_mm, 
#             bsdf_resolved, 
#             max_twist_bow_mm=max_twist_bow_mm,
#             return_curve=True,
#             z_displacement = z_displacement
#         )
#     else:
#         cylinder_radius = config['measurements']['rod_width_mm']/2
#         rods = [ get_rod((*rc,0),cylinder_radius,rod_height) for rc in rod_centers]

#         rod_objs = []
#         for i, rod in enumerate(rods[:]):
#             rod['my_bsdf'] = bsdf_resolved
            
#             rod_objs.append(rod)

#         return rod_objs
    
# def get_tips(curves, config,tip_ply = 'assets/tip_model.ply'):
#     rod_tops = np.array([c.evaluate(0) for c in curves])
#     tips_scenes = {}
    
#     rods_material = config['fuel_material']['rods']
#     rods_material_alpha_u = rods_material['material_alpha_u']
#     rods_material_alpha_v = rods_material['material_alpha_v']
#     rods_gray_intensity =rods_material['diffuse_gray_intensity']
#     rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']

#     for i,(x,y,_) in enumerate(rod_tops):
#         rods_bdsf = get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)
#         rods_bdsf['id'] = f'tip_material_{i}'
#         offset = 9.1 + 3.65
#         y_offset = np.sqrt(offset**2 - (offset/2)**2)
#         y_shift =  offset - y_offset *12
#         tips_scenes[f"tip_{i}"] = {        
#             'type': 'ply',
#             'filename': tip_ply,
#             "to_world" :mi.ScalarTransform4f.translate([x,y+y_shift,0]),

#             "material":rods_bdsf,
#             # "material": {
#             #     'type': 'diffuse',
#             #     'reflectance': {
#             #         'type': 'rgb',
#             #         'value': [1, 0, 1]
#             #     }
#             # }
#         }
#     return tips_scenes

# def add_row(start_x,start_y, n,rod_center_distance_mm):
#     pts_x = np.arange(n) * rod_center_distance_mm + start_x
#     pts_y = np.ones((n,)) * start_y
#     return np.stack([pts_x,pts_y]).T


# def generate_rod_centers(visible_rod_count, rod_width_mm,gap_between_rods_width_mm):
#     rod_center_distance_mm = rod_width_mm + gap_between_rods_width_mm

#     pythagorean_factor = np.sqrt(3)/2
#     rod_centers_lines_arrays = [add_row(0,0,visible_rod_count*2 - 1,rod_center_distance_mm) ]
#     x_shift = rod_centers_lines_arrays[0][visible_rod_count-1]
        
#     for i,current_rod_count in zip(range(1,visible_rod_count),range(visible_rod_count+visible_rod_count - 2,visible_rod_count-1,-1)):

#         x = (i*rod_center_distance_mm)/2
#         y = i * pythagorean_factor* rod_center_distance_mm
#         points_b = add_row(x,-y,current_rod_count,rod_center_distance_mm)
#         points_t = add_row(x,y,current_rod_count,rod_center_distance_mm)

#         rod_centers_lines_arrays.append(points_b)
#         rod_centers_lines_arrays.append(points_t)
#     return np.concatenate(rod_centers_lines_arrays) - x_shift

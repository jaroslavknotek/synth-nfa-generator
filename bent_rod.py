import numpy as np
import bezier

import mitsuba as mi
import os


class PiecewiseBezierCurve():
    
    def __init__(self, bezier_records,nfa_height):
        self.nfa_height=nfa_height
        self.bezier_records = bezier_records
    
    def evaluate_multi(self,zs):
        
        # Dirty solution - poc
        
        found_xyz = []
        for z in zs:
            last_end = None
            last_end_point = None
            z = z*self.nfa_height
            
            for i,(start,end,curve) in enumerate(self.bezier_records):
                if z < start:
                    start_point = curve.evaluate_multi(np.array([0.0])).T[0]
                    alpha = (last_end - z)/(start - z)
                    
                    x,y,_ =  last_end_point*alpha + start_point*(1-alpha)
                    if y>5000:
                        print(alpha, last_end_point, start_point)
                        raise Exception("here")
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
    
    def evaluate(self,zs):
        
        return self.evaluate_multi([zs])
    
def get_twisted_nfa_mesh(
    config, 
    rod_centers,
    spacing_mm,
    grid_heights_mm,
    max_divergence_mm, 
    bsdfmaterial,
    max_twist_bow_mm=0, 
    rod_bow_segments = 20, 
    rod_circle_segments = 32,
    seed = 678,
    n_textures = None,
    return_curve = False,
    z_displacement = None
):
    
    cylinder_radius = config['measurements']['rod_width_mm']/2
    spacing_mm = np.concatenate([[0], spacing_mm])
    bent_random = np.random.RandomState(seed)
    nfa_curve, curves = get_nfa_curves(spacing_mm, rod_centers,grid_heights_mm,max_divergence_mm,max_twist_bow_mm,bent_random)
    rod_vertices_faces =        [
        get_curved_rod(
            nfa_curve,
            curve,
            rod_bow_segments,
            rod_circle_segments,
            radius=cylinder_radius
        )
        for curve in curves
    ]
    rod_vertices =  np.array([r[0] for r in rod_vertices_faces])
    
    if z_displacement is None:
        z_displacement = 0
    rod_vertices[:,:,2] += z_displacement
    
    rod_faces =  [r[1] for r in rod_vertices_faces]
    
    # TODO add texture to just some rods
    # increase only x - expects the texture concat in x
    uv_list = [r[2] for r in rod_vertices_faces]
    
    #uvs =np.concatenate(uv_list,axis=0)
    n_textures = 20
    
    uvs =np.concatenate([ (uv + [i%n_textures,0]) for i,uv in enumerate(uv_list)],axis=0)
    uvs[:,0] = uvs[:,0]/n_textures
    
    v,f = bake_rod_geometry(rod_vertices,rod_faces)
    
    mesh = get_rod_mesh(v,f,uvs)
    
    
    os.makedirs("tmp",exist_ok=True)
    mesh_path = "tmp/test_rod.ply"
    mesh.write_ply(mesh_path)

    # the list brackets ('[',']') are here because of compatibility with existing solution
    
    res= [{
        "type": "ply",
        "filename": mesh_path,
        "material":bsdfmaterial
    }]
    if return_curve:
        return nfa_curve,curves,res
    else: 
        return res

def bake_rod_geometry(vertices_list,faces_list):
    
    vertices = np.concatenate(vertices_list,axis=0)   
    offsets = np.cumsum(np.concatenate([[0],[len(v) for v in vertices_list]]))             
    
    faces = np.concatenate([ np.add(f,offset) for f,offset in zip(faces_list,offsets)])
   
    return vertices, faces


def get_rod_curve_z(spacing_mm):
    fp = np.cumsum(spacing_mm)
    xp = np.arange(len(fp))
    max_val = len(xp)-1
    x = np.linspace(0,max_val, max_val*2 +1 )
    return np.interp( x, xp,fp)

def _sample_curve(curve, samples):
    x,_,z = curve.evaluate_multi(samples)
    return x,z

def plot_rod_curves(ax,nfa_bow_curve, grid_positions, rod_deflections,nfa_height):
    plot_samples = np.linspace(0,1,100)
    
    nfa_bow_curve_x,nfa_bow_curve_z = _sample_curve(nfa_bow_curve,plot_samples)

    p_grid = _plot_grids(ax, grid_positions,nfa_height)
    
    rods_points = [_sample_curve(rod_curve,plot_samples) for rod_curve in rod_deflections]
    p_rods= [ax.plot(x,y,c='orange') for x,y in rods_points ]
    p_rod, = p_rods[0]
    
    p_bow, = ax.plot(nfa_bow_curve_x,nfa_bow_curve_z)
    
    plots = [p_grid, p_bow, p_rod]
    legend_lables = ['grid_position', 'nfa bow','rods']
    ax.legend(plots,legend_lables,handler_map={tuple: HandlerTuple(ndivide=None)})
    
    
def _plot_grids(ax,grid_positions,nfa_height):
    
    # Grids distances are measured from above, therefore I need to reverse it here
    
    grid_positions_bottom_up =  np.abs(nfa_height - grid_positions)
    p_grids = [ax.axhline(gp,c='#faa') for gp in grid_positions_bottom_up]
    return p_grids[0] # assuming you treat plotted grids' position as one
    
def plot_rod_deflections(ax,rod_deflections, grid_positions_mm, rods_offset_mm,nfa_height):    
    plot_samples = np.linspace(0,1,100)
    rods_points = np.array([_sample_curve(rod_curve,plot_samples) for rod_curve in rod_deflections])
    rods_points_diffs = np.diff(rods_points[:,0,:], axis = 0) - rods_offset_mm
    
    _ = _plot_grids(ax, grid_positions_mm,nfa_height)
    ax.axvline(0)
    y = rods_points[0,1,:]
    for x in rods_points_diffs:
        ax.plot(x,y,c='gray')

    
def get_curve(start_point,end_point, max_offset_mm):
    # there should be a paramater indicating the diff
    
    h = start_point[2] - end_point[2]
    middle_points = np.array([
        #[ 0,max_offset_mm, end_point[2] + h*.25 ],
        [ -max_offset_mm,0, end_point[2] + h*.50 ],
        #[ 0,-max_offset_mm,end_point[2] + h*.75],
    ])

    nodes = np.concatenate([[start_point], middle_points,[end_point]],axis=0)
    bezier_nodes = nodes.T
    
    return bezier.Curve.from_nodes(bezier_nodes)


def get_nfa_bow_curve(nfa_height_mm, max_twist_bow_mm = 0):
    start_point = np.array([0,0,nfa_height_mm])
    end_point = np.array([0,0,0])
    return get_curve(start_point, end_point ,max_twist_bow_mm)



def get_rod_deflections_piecewise_bezier(nfa_bow_curve, rod_centers, grid_heights_mm, spacing_mm,rod_divergence_mm,bent_random):
    rods = []    
    nfa_height_mm = np.sum(spacing_mm)
    
    #None for start/end of fuel
    grid_heights_mm_per_node = np.concatenate([[0],grid_heights_mm,[0]])
    z = np.cumsum(spacing_mm)
    
    for rc_x,rc_y in rod_centers:
        beziers = []
        for z_start,z_end, grid_end_margin, grid_start_margin in zip(z,z[1:],grid_heights_mm_per_node//2,grid_heights_mm_per_node[1:]//2):
            z_start_grid = z_start + grid_end_margin
            z_end_grid = z_end - grid_start_margin
            x,y = bent_random.rand(2)*rod_divergence_mm*2 - rod_divergence_mm
            
            assert z_start_grid < z_end_grid            
            bezier_start = [rc_x,rc_y,z_start_grid]
            
            ensure_straight_start_paddig = grid_end_margin *2
            ensure_straight_end_paddig = grid_start_margin *2
            
            middle_middle_z = (z_end_grid + z_start_grid)//2
            assert z_start_grid + ensure_straight_start_paddig < middle_middle_z, f"Bezier start with padding {z_start_grid + ensure_straight_start_paddig} can't be **more** than middle point: {middle_middle_z}"
            assert middle_middle_z < z_end_grid - ensure_straight_end_paddig, f"Bezier end with padding {z_end_grid - ensure_straight_end_paddig} can't be **less** than middle point: {middle_middle_z}"
            
            bezier_middle_start = [rc_x,rc_y, z_start_grid + ensure_straight_start_paddig ]
            bezier_middle_middle = [rc_x + x, rc_y + y, middle_middle_z]
            bezier_middle_end = [rc_x,rc_y, (z_end_grid - ensure_straight_end_paddig)]
            
            bezier_end = [rc_x,rc_y,z_end_grid]
            
            
            nodes_list = [
                bezier_start,
                bezier_middle_start,
                bezier_middle_middle,
                bezier_middle_end,
                bezier_end
            ]
            nodes = np.stack(nodes_list).T
                        
            bezier_piece = bezier.Curve.from_nodes(nodes)
            bezier_record = (z_start_grid,z_end_grid,bezier_piece)
            beziers.append(bezier_record)
            
        
        
        piecewise_bezier_curve = PiecewiseBezierCurve(beziers,nfa_height_mm)
        rods.append(piecewise_bezier_curve)

    return rods


def get_nfa_curves(spacing_mm, rod_centers,grid_heights_mm,rod_divergence_mm,max_twist_bow_mm,bent_random):
    
    nfa_height_mm = np.sum(spacing_mm)
    
    nfa_bow_curve = get_nfa_bow_curve(nfa_height_mm,max_twist_bow_mm=max_twist_bow_mm)
    
    return nfa_bow_curve, get_rod_deflections_piecewise_bezier(nfa_bow_curve, rod_centers,grid_heights_mm, spacing_mm,rod_divergence_mm,bent_random)

def get_curved_rod(nfa_curve,curve, rod_segments, cylinder_faces, radius = 1):
    return get_rod_geometry(rod_segments, cylinder_faces, radius = radius, nfa_curve = nfa_curve,curve = curve)

def get_rod_geometry(rod_segments, cylinder_faces, radius = 1,nfa_curve = None,curve = None,height = None):
    actual_segments = rod_segments +1
    curve_eval_points = np.linspace(0,1,actual_segments)
    
    if curve is None:
        c_x = np.zeros((actual_segments,))
        c_y = np.copy(c_x)
        c_z = curve_eval_points*(height or 1)
    else:
        c_x,c_y,c_z = curve.evaluate_multi(curve_eval_points)
        
    if nfa_curve is None:
        n_x = np.zeros((actual_segments,))
        n_y = np.copy(c_x)
        n_z = curve_eval_points*(height or 1)
    else:
        n_x,n_y,n_z = nfa_curve.evaluate_multi(curve_eval_points)
    
    x = n_x + c_x
    y = n_y + c_y
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
            # face_indices.append([a,c,b])
            # face_indices.append([c,d,b])
        
    # Texture coors
    y_coors = (np.linspace(0,1,actual_segments))
    uv_y = np.multiply(np.expand_dims(y_coors,axis=1),np.ones(cylinder_faces)[None]).flatten()
    uv_x = np.concatenate([np.linspace(0,1,cylinder_faces)]*(actual_segments))
    uv = np.vstack([uv_x,uv_y]).T    
    return np.array(vertex_pos), np.array(face_indices,dtype=np.int32), uv


# def get_rod_geometry(rod_segments,cylinder_faces, radius = 1):
#     actual_segments = rod_segments +1
#     spacing = np.linspace(0,2*np.pi,cylinder_faces)
#     x = np.sin(spacing) *radius
#     y = np.cos(spacing) *radius
#     z = np.linspace(0,1,actual_segments)
#     vertex_pos =np.concatenate([[x,y,np.ones((len(y),))*zz] for zz in z],axis = 1).T
#     face_indices = []
#     for i in range(actual_segments-1):
#         for j in range(cylinder_faces-1):
#             # a -- b
#             # | -- |
#             # c -- d
#             idxs = np.array([(i,j),(i,j+1),(i+1,j),(i+1,j+1)]).T
#             a,b,c,d= np.ravel_multi_index(idxs,(actual_segments,cylinder_faces))

#             # face_indices.append([a,b,c])
#             # face_indices.append([c,b,d])
#             face_indices.append([d,b,a])
#             face_indices.append([d,a,c])
        
#     return vertex_pos, face_indices

def get_rod_mesh(vertex_pos, face_indices,uv):
    mesh = mi.Mesh(
        "nfa_rods_twisted",
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

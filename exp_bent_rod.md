---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: synthdata-Cub-fwJX-py3.9
    language: python
    name: synthdata-cub-fwjx-py3.9
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 12]
mi.set_variant('scalar_rgb')
#mi.set_variant('cuda_ad_rgb')
```

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_curve(x,y,z, ax = None):
    
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    
    ax.plot(x, y, z)
    return ax
```

```python
import bezier

def generate_dummy_nfa_curve(twist_offset = .1):
    start_point = [0,0,1]
    middle_points = np.array([
        [ twist_offset,twist_offset,.66],
        [ -twist_offset,twist_offset,.33],
    ])
    end_point = [0,0,0]

    nodes = np.concatenate([[start_point], middle_points,[end_point]],axis=0)
    bezier_nodes = nodes.T
    
    return bezier.Curve(bezier_nodes, degree=3)
```

```python
import bezier
from matplotlib.legend_handler import HandlerTuple

config = {
    "name": "mod3",
    "grid_heights_mm": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], 
    "spacing_mm": [200, 255, 340, 340, 340, 340, 340, 340, 340, 340, 340, 240],
    "shift_mm": [0, 200],
    "visible_rods": 11
  }


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

def get_rod_curve_z(spacing_mm):
    fp = np.cumsum(spacing_mm)
    xp = np.arange(len(fp))
    max_val = len(xp)-1
    x = np.linspace(0,max_val, max_val*2 +1 )
    return np.interp( x, xp,fp)
    

def get_nfa_bow_curve(nfa_height_mm, max_twist_bow = None):
    max_twist_bow_resolved = max_twist_bow if max_twist_bow is not None else nfa_height_mm/10
    start_point = np.array([0,0,nfa_height_mm])
    end_point = np.array([0,0,0])
    return get_curve(start_point, end_point ,max_twist_bow_resolved)
    
def get_nfa_curves(config, rod_offset_mm):
    rod_count = 10
    spacing_mm = np.concatenate([[0], config['spacing_mm']])
    nfa_height_mm = np.sum(spacing_mm)
    
    nfa_bow_curve = get_nfa_bow_curve(nfa_height_mm)
    
    return get_rod_deflections_exp(nfa_bow_curve, rod_count, spacing_mm,rod_offset_mm)

def get_rod_deflections(nfa_bow_curve, rod_count, spacing_mm,rod_offset_mm, rod_divergence_mm=None):
    rod_divergence_mm_resolved = rod_divergence_mm if rod_divergence_mm is not None else rod_offset_mm
    
    # TODO I need to refactor this so It would produce version with deflections 
    # that will correspoind to grid fixating them in a given spaceing.
    rods = []    
    nfa_height_mm = np.sum(spacing_mm)
    for i in range(rod_count):
        x_shift_mm = i*rod_offset_mm
        
        z = get_rod_curve_z(spacing_mm)
        
        nfa_eval_value = z/nfa_height_mm
        
        nodes = nfa_bow_curve.evaluate_multi(nfa_eval_value)
    
        non_grid_nodes_num = int((nodes.shape[1] -1)/2)
        
        # here I will randomize x,y 
        
        x,y = np.random.rand(2,non_grid_nodes_num)*rod_divergence_mm_resolved
        
        every_second_x = np.concatenate([x[:,np.newaxis], np.zeros((len(x),1))],axis=1).flatten()
        nodes[0,1:] += every_second_x
        every_second_y = np.concatenate([y[:,np.newaxis], np.zeros((len(y),1))],axis=1).flatten()
        nodes[2,1:] += every_second_y
        
        # then I count on projecting the x,y into the nfa face so
        # the rods wont interfere
        
        # shift rod
        nodes[0] +=x_shift_mm 
        
        # first column represent z where 
        z_segment_grid = z[1:].reshape((-1,2))
        
        
        rod_curve = bezier.Curve.from_nodes(nodes)
        rods.append(rod_curve)
        
    return rods


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
    p_grids = [ax.axhline(gp,c='r',alpha=.2) for gp in grid_positions_bottom_up]
    return p_grids[0] # assuming you treat plotted grids' position as one
    
def plot_rod_deflections(ax,rod_deflections, grid_positions_mm, rods_offset_mm,nfa_height):    
    plot_samples = np.linspace(0,1,100)
    rods_points = np.array([_sample_curve(rod_curve,plot_samples) for rod_curve in rod_deflections])
    rods_points_diffs = np.diff(rods_points[:,0,:], axis = 0) - rods_offset_mm
    
    _ = _plot_grids(ax, grid_positions_mm,nfa_height)
    ax.axvline(0)
    y = rods_points[0,1,:]
    for x in rods_points_diffs:
        ax.plot(x,y)

class CurveCompatibilityWrapper:
    
    def __init__(self,nodes,nfa_height):
        
        increasing_order = np.flip(nodes,axis = 1)
        heights = increasing_order[2] # fitted to [0,1] interval
        x_y_shifts= increasing_order[:-1]
        
        self.spline = CubicSpline(heights, x_y_shifts.T)
        self.nfa_height = nfa_height
        
    def evaluate_multi(self,zs):
        zs_scaled = (zs*self.nfa_height)
        zs_flipped = np.flip(zs_scaled)
        interp_x,interp_y = self.spline(zs_flipped).T
        interp = np.array([interp_x,interp_y,zs_flipped])
        
        return interp
        
        
        

def get_rod_deflections_exp(nfa_bow_curve, rod_count, spacing_mm,rod_offset_mm, rod_divergence_mm=None):
    rod_divergence_mm_resolved = rod_divergence_mm if rod_divergence_mm is not None else rod_offset_mm
    
    # TODO I need to refactor this so It would produce version with deflections 
    # that will correspoind to grid fixating them in a given spaceing.
    rods = []    
    nfa_height_mm = np.sum(spacing_mm)
    for i in range(rod_count):
        x_shift_mm = i*rod_offset_mm
        
        z = get_rod_curve_z(spacing_mm)
        
        nfa_eval_value = z/nfa_height_mm
        
        nodes = nfa_bow_curve.evaluate_multi(nfa_eval_value)
    
        non_grid_nodes_num = int((nodes.shape[1] -1)/2)
        
        # here I will randomize x,y 
        
        x,y = np.random.rand(2,non_grid_nodes_num)*rod_divergence_mm_resolved
        
        every_second_x = np.concatenate([x[:,np.newaxis], np.zeros((len(x),1))],axis=1).flatten()
        nodes[0,1:] += every_second_x
        every_second_y = np.concatenate([y[:,np.newaxis], np.zeros((len(y),1))],axis=1).flatten()
        nodes[2,1:] += every_second_y
        
        # then I count on projecting the x,y into the nfa face so
        # the rods wont interfere
        
        # shift rod
        nodes[0] +=x_shift_mm 
        
        # first column represent z where 
        z_segment_grid = z[1:].reshape((-1,2))
        
        # you give it height and it will return x,y shift in a plane
                
        rod_curve = CurveCompatibilityWrapper(nodes, nfa_height_mm)

        rods.append(rod_curve)
        
    return rods
 
spacing_mm = np.concatenate([[0], config['spacing_mm']])
rod_count = 11
rod_width_mm = 9.1
gap_between_rods_width_mm= 3.65
rods_offset_mm = rod_width_mm + gap_between_rods_width_mm

grid_positions_mm = np.cumsum(spacing_mm)
nfa_height_mm = np.sum(spacing_mm)
nfa_bow_curve = get_nfa_bow_curve(nfa_height_mm)

rod_deflections = get_rod_deflections(nfa_bow_curve, rod_count, spacing_mm,rods_offset_mm)

fig,(ax0,ax1,ax2) = plt.subplots(1,3)
plot_rod_curves(ax0, nfa_bow,grid_positions_mm, rod_deflections,nfa_height_mm)
plot_rod_deflections(ax1,rod_deflections, grid_positions_mm, rods_offset_mm,nfa_height_mm)

rod_deflections_exp = get_rod_deflections_exp(nfa_bow_curve, rod_count, spacing_mm,rods_offset_mm)
plot_rod_deflections(ax2,rod_deflections_exp, grid_positions_mm, rods_offset_mm,nfa_height_mm)
```

```python
def get_rod_geometry(rod_segments,cylinder_faces, radius = 1):
    actual_segments = rod_segments +1
    spacing = np.linspace(0,2*np.pi,cylinder_faces)
    x = np.sin(spacing) *radius
    y = np.cos(spacing) *radius
    z = np.linspace(0,1,actual_segments)
    vertex_pos =np.concatenate([[x,y,np.ones((len(y),))*zz] for zz in z],axis = 1).T
    face_indices = []
    for i in range(actual_segments-1):
        for j in range(cylinder_faces-1):
            # a -- b
            # | -- |
            # c -- d
            idxs = np.array([(i,j),(i,j+1),(i+1,j),(i+1,j+1)]).T
            a,b,c,d= np.ravel_multi_index(idxs,(actual_segments,cylinder_faces))

            # face_indices.append([a,b,c])
            # face_indices.append([c,b,d])
            face_indices.append([d,b,a])
            face_indices.append([d,a,c])
        
    return vertex_pos, face_indices

def get_rod_mesh(vertex_pos, face_indices):
    mesh = mi.Mesh(
        "grid",
        vertex_count=len(vertex_pos),
        face_count=len(face_indices),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = np.ravel(vertex_pos)
    mesh_params["faces"] = np.ravel(face_indices)
    return mesh

def render_object(obj,cam_origin = None):
    width,height = 1200,1000
    cam_origin = cam_origin if cam_origin is not None else [0, 10, .5]
    cam_light_intensity = 400000
    left_light =  {
        "type":"point",
        "intensity":{
            "type":"rgb",
            "value":cam_light_intensity,},
        "to_world":  mi.ScalarTransform4f.translate(cam_origin).translate([-100,0,0])
    }
    right_light = {
        "type":"point",
        "intensity":{
            "type":"rgb",
            "value":cam_light_intensity,},
        "to_world":  mi.ScalarTransform4f.translate(cam_origin).translate([100,0,0])
    }
    
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path"},
        "left_light": left_light,
        "right_light": right_light,
        # "light":{
        #     'type': 'constant',
        #     'radiance': {
        #         'type': 'rgb',
        #         'value': .5,
        #     }
        # },
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.translate(cam_origin).rotate([1,0,0],90),
            "myfilm" : {
                "type" : "hdrfilm",
                "width" : width,
                "height" : height,
            },
            # "to_world": mi.ScalarTransform4f.look_at(
            #     origin=cam_origin, target=[0, 0, 0], up=[0, 0, 1]
            # ),
        },
        "grid": obj,
    })

    img = mi.render(scene)
    return mi.util.convert_to_bitmap(img)

# rod_segments = 4
# cylinder_faces = 10

# vertex_pos, face_indices = get_rod_geometry(rod_segments,cylinder_faces,radius = .5)
# mesh = get_rod_mesh(vertex_pos, face_indices)
# mesh_render = render_object(mesh)
# plt.imshow(mesh_render)
# plt.show(),
```

```python
def shift_rod_by_curve(vertex_pos,curve):
    
    
#     assert (curve.evaluate(0.0).T == [0,0,1]).all(),"bezier should start at x=0,y=0,z=1"
#     assert (curve.evaluate(1.0).T == [0,0,0]).all(), "bezier should start at x=0,y=0,z=0"
    
    
    z= vertex_pos[:,2]
    evaluated_curve = curve.evaluate_multi(z)
    xy_shift = evaluated_curve[:2].T
    
    vertex_pos = vertex_pos.copy()
    vertex_pos[:,:2] = vertex_pos[:,:2] + xy_shift    
    return vertex_pos

def plot_curved(rod_segments = 10, cylinder_faces = 10,radius = .1):
    vertex_pos, face_indices = get_rod_geometry(rod_segments,cylinder_faces, radius = radius)
    curve = generate_dummy_nfa_curve()
    bent_vertex_pos = shift_rod_by_curve(vertex_pos, curve)

    mesh = get_rod_mesh(bent_vertex_pos, face_indices)
    mesh_render = render_object(mesh)
    plt.imshow(mesh_render)
    plt.show()

#plot_curved()
```

# Add material

```python
import os 
import scene_definition

def test_get_mesh(rod_segments = 10, cylinder_faces = 10,radius = .1):
    vertex_pos, face_indices = get_rod_geometry(rod_segments,cylinder_faces, radius = radius)
    curve = generate_dummy_nfa_curve()
    bent_vertex_pos = shift_rod_by_curve(vertex_pos, curve)

    return get_rod_mesh(bent_vertex_pos, face_indices)


def get_test_material():
    
    # return {
    #         "type": "diffuse",
    #         'reflectance': {
    #             'type': 'rgb',
    #             'value': [0.2, 0.25, 0.7]
    #         }
    # }
    
    # copies from main notebook
    
    measurements = {
        "rod_width_mm" : 9.1, 
        "gap_between_rods_width_mm": 3.65,

        #based on diagram, not blueprint -> approximate values
        "camera_distance_mm" : 430,
        "light_offset_mm": 228    
    }

    fuel_material={
        "rods":{
            "material_alpha_u": .05,
            "material_alpha_v": .12,
            "diffuse_gray_intensity": 0.25,
            "zircon_to_bsdf_blend_factor": .25,
        },
    }
    config = {
        "measurements":measurements,
        "fuel_material": fuel_material
    }
    
    
    rods_material = config['fuel_material']['rods']
    rods_material_alpha_u = rods_material['material_alpha_u']
    rods_material_alpha_v = rods_material['material_alpha_v']
    rods_gray_intensity =rods_material['diffuse_gray_intensity']
    rods_bsdf_blend_factor = rods_material['zircon_to_bsdf_blend_factor']
        
    return scene_definition.get_rods_bsdf(rods_gray_intensity, rods_bsdf_blend_factor, rods_material_alpha_u,rods_material_alpha_v)

def render_color_rod(mesh,cam_origin = None):
    
    os.makedirs("tmp",exist_ok=True)
    mesh_path = "tmp/test_rod.ply"
    mesh.write_ply(mesh_path)

    material = get_test_material()
    mesh_obj = {
        "type": "ply",
        "filename": mesh_path,
        "material": material
    }
    
    img = render_object(mesh_obj,cam_origin=cam_origin)
    plt.imshow(img)
    
# mesh = test_get_mesh(cylinder_faces = 32)
# render_color_rod(mesh)
```

# Bend on predfined positions

- create a row of 10 rods
- define nfa_twist (requires specific position of grids)
- define per-segment twist that will add up to the nfa_twist
  - the starting point of each bezier would be the nfa_twist bezier evaluated at position of each spacer grid.
 

```python
def get_curved_rod(curve, rod_segments, cylinder_faces, radius = 1):
    actual_segments = rod_segments +1
    curve_eval_points = np.linspace(0,1,actual_segments)
    
    x,y,z = curve.evaluate_multi(curve_eval_points)
    spacing = np.linspace(0,2*np.pi,cylinder_faces)
    x_circle = np.sin(spacing)* radius 
    y_circle = np.cos(spacing)* radius
    
    cylinder_coor_shape = (actual_segments,len(x_circle))
    cylinder_x = np.broadcast_to(x_circle ,cylinder_coor_shape) + np.broadcast_to(x[:,np.newaxis], cylinder_coor_shape)
    cylinder_y = np.broadcast_to(y_circle ,cylinder_coor_shape) + np.broadcast_to(x[:,np.newaxis], cylinder_coor_shape)
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
            # face_indices.append([d,b,a])
            # face_indices.append([d,a,c])
        
    return vertex_pos, face_indices


def bake_rod_geometry(vertices_list,faces_list):
    
    vertices = np.concatenate(vertices_list,axis=0)   
    offsets = np.cumsum(np.concatenate([[0],[len(v) for v in vertices_list]]))             
    
    faces = np.concatenate([ np.add(f,offset) for f,offset in zip(faces_list,offsets)])

    return vertices, faces


rod_width_mm = 9.1
gap_between_rods_width_mm= 3.65
rods_offset = rod_width_mm + gap_between_rods_width_mm
    
curves = get_nfa_curves(config, rods_offset)
rod_vertices_faces =np.array([get_curved_rod(curve,20,32,radius=rod_width_mm/2) for curve in curves],dtype=object)
v,f = bake_rod_geometry(rod_vertices_faces[:,0],rod_vertices_faces[:,1])    
mesh = get_rod_mesh(v,f)
render_color_rod(mesh, cam_origin =[50,2000,1100])

```

```python

```

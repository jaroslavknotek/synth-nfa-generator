---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
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
import bent_rod

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
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import CubicSpline

def get_rod_curve_z(*args,**kwargs):
    return bent_rod.get_rod_curve_z(*args,**kwargs)


class CurveCompatibilityWrapper:
    
    def __init__(self,nodes,nfa_height):
        increasing_order = np.flip(nodes,axis = 1)
        heights = increasing_order[2] # fitted to [0,1] interval
        x_y_shifts= increasing_order[:-1]
        
        self.spline = CubicSpline(heights, x_y_shifts.T)
        self.nfa_height = nfa_height
        
    def evaluate_multi(self,zs):
        zs_scaled = ((1-zs)*self.nfa_height)
        interp_x,interp_y = self.spline(zs_scaled).T
        interp = np.array([interp_x,interp_y,np.flip(zs_scaled)])
        
        return interp

def get_rod_deflections_exp(nfa_bow_curve, rod_centers,grid_heights_mm,spacing_mm,rod_divergence_mm):
    rods = []    
    nfa_height_mm = np.sum(spacing_mm)
    for rc_x,rc_y in rod_centers:
        z = get_rod_curve_z(spacing_mm)
        
        nfa_eval_value = z/nfa_height_mm
        
        nodes = nfa_bow_curve.evaluate_multi(nfa_eval_value)
    
        non_grid_nodes_num = int((nodes.shape[1] -1)/2)
        
        # here I will randomize x,y 
        
        x,y = np.random.rand(2,non_grid_nodes_num)*rod_divergence_mm*2 - rod_divergence_mm
        
        every_second_x = np.concatenate([x[:,np.newaxis], np.zeros((len(x),1))],axis=1).flatten()
        nodes[0,1:] += every_second_x
        every_second_y = np.concatenate([y[:,np.newaxis], np.zeros((len(y),1))],axis=1).flatten()
        nodes[2,1:] += every_second_y
        
        # then I count on projecting the x,y into the nfa face so
        # the rods wont interfere
        
        # shift rod
        nodes[0] +=rc_x 
        nodes[1] +=rc_y 
        
        # first column represent z where 
        z_segment_grid = z[1:].reshape((-1,2))
        
        # you give it height and it will return x,y shift in a plane
                
        rod_curve = CurveCompatibilityWrapper(nodes, nfa_height_mm)

        rods.append(rod_curve)
        
    return rods

def get_rod_deflections(nfa_bow_curve, rod_centers, grid_heights_mm, spacing_mm,rod_divergence_mm):
    rod_divergence_mm_resolved = rod_divergence_mm if rod_divergence_mm is not None else rod_offset_mm
    
    # TODO I need to refactor this so It would produce version with deflections 
    # that will correspoind to grid fixating them in a given spaceing.
    rods = []    
    nfa_height_mm = np.sum(spacing_mm)
    for rc_x,rc_y in rod_centers:
        z = get_rod_curve_z(spacing_mm)
        
        nfa_eval_value = z/nfa_height_mm
        
        nodes = nfa_bow_curve.evaluate_multi(nfa_eval_value)
    
        non_grid_nodes_num = int((nodes.shape[1] -1)/2)
        
        # here I will randomize x,y 
        
        x,y = np.random.rand(2,non_grid_nodes_num)*rod_divergence_mm*2 - rod_divergence_mm
        
        every_second_x = np.concatenate([x[:,np.newaxis], np.zeros((len(x),1))],axis=1).flatten()
        nodes[0,1:] += every_second_x
        every_second_y = np.concatenate([y[:,np.newaxis], np.zeros((len(y),1))],axis=1).flatten()
        nodes[2,1:] += every_second_y
        
        # then I count on projecting the x,y into the nfa face so
        # the rods wont interfere
        
        # shift rod
        nodes[0] +=rc_x 
        nodes[1] +=rc_y 
        
#         # first column represent z where 
#         z_segment_grid = z[1:].reshape((-1,2))
        
        
        rod_curve = bezier.Curve.from_nodes(nodes)
        rods.append(rod_curve)
        
    return rods

    
def plot_rod_deflections(ax,rod_deflections, grid_positions_mm, grid_heights_mm, rods_offset_mm,nfa_height_mm,title):
    eval_at_z = np.linspace(0,1,1000)
    curves = [ curve.evaluate_multi(eval_at_z) for curve in rod_deflections]
    
    ax.set_title(title)
    for i, curve in enumerate(curves):
        x = curve[0] - rods_offset_mm *i
        
        y = curve[-1] # eval_at_z*nfa_height_mm
        ax.plot(x ,y)
        
    grid_heights_mm_add = np.concatenate([[None],grid_heights_mm,[None]])
    for gp,gh in zip(grid_positions_mm, grid_heights_mm_add):
        ax.axhline(gp,alpha=.2)
        if gh is not None:
            ax.axhline(gp - gh//2,c='r',alpha=.2)
            ax.axhline(gp + gh//2,c='r',alpha=.2)
        
        
    ax.axvline(0,alpha =.2)
    #ax.set_ylim((0,1000))

config = {
    'grid_detection':
    {"mods":[{
    "name": "mod3",
    "grid_heights_mm": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], 
    "spacing_mm": [200, 255, 340, 340, 340, 340, 340, 340, 340, 340, 340, 240],
    "shift_mm": [0, 200],
    "visible_rods": 11}
        ]
    }
  }


mod = config['grid_detection']["mods"][-1]
spacing_mm = np.concatenate([[0], mod['spacing_mm']])
grid_heights_mm = mod['grid_heights_mm']
rod_count = 11
rod_width_mm = 9.1
gap_between_rods_width_mm= 3.65
rods_offset_mm = rod_width_mm + gap_between_rods_width_mm

grid_positions_mm = np.cumsum(spacing_mm)
nfa_height_mm = np.sum(spacing_mm)
nfa_bow_curve = bent_rod.get_nfa_bow_curve(nfa_height_mm)

#rod_deflections = bent_rod.get_rod_deflections(nfa_bow_curve, rod_count, spacing_mm,rods_offset_mm)

fig,(ax0,ax1,ax2) = plt.subplots(1,3)
# plot_rod_curves(ax0, nfa_bow_curve,grid_positions_mm, rod_deflections,nfa_height_mm)
# plot_rod_deflections(ax1,rod_deflections, grid_positions_mm, rods_offset_mm,nfa_height_mm)

test_rod_centers = [(i*rods_offset_mm,0) for i in range(rod_count)]
rod_deflections = get_rod_deflections(nfa_bow_curve, test_rod_centers, grid_heights, spacing_mm,rods_offset_mm)
rod_deflections_exp = get_rod_deflections_exp(nfa_bow_curve, test_rod_centers, grid_heights, spacing_mm,rods_offset_mm)
rod_deflections_piecewise_bezier = bent_rod.get_rod_deflections_piecewise_bezier(nfa_bow_curve, test_rod_centers, grid_heights, spacing_mm,rods_offset_mm)

plot_rod_deflections(ax0,rod_deflections, grid_positions_mm, grid_heights_mm,rods_offset_mm,nfa_height_mm,"Just bezier")    
plot_rod_deflections(ax1,rod_deflections_piecewise_bezier, grid_positions_mm, grid_heights_mm,rods_offset_mm,nfa_height_mm,"Piecewise bezier")    
plot_rod_deflections(ax2,rod_deflections_exp, grid_positions_mm, grid_heights_mm,rods_offset_mm,nfa_height_mm, "Spline")
```

```python

```

```python

def render_objects(objs,cam_origin = None):
    width,height = 1200,1000
    cam_origin = cam_origin if cam_origin is not None else [0, 10, .5]
    cam_light_intensity = 400000
    
    obj_dict = dict( (f"o_{i}", o) for i,o in enumerate(objs))
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
    
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},
        "left_light": left_light,
        "right_light": right_light,
        "light":{
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 10.5,
            }
        },
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
    }
    scene_dict.update(obj_dict)
    scene = mi.load_dict(scene_dict)
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

# Bend on predefined positions

- create a row of 10 rods
- define nfa_twist (requires specific position of grids)
- define per-segment twist that will add up to the nfa_twist
  - the starting point of each bezier would be the nfa_twist bezier evaluated at position of each spacer grid.
 

```python
import rods_hexagon
import scene_definition
import bent_rod
def get_rod(*args,**kwargs):
    return scene_definition.get_rod(*args,**kwargs)


def generate_rods_group(config, has_rod_divergence, bsdf=None):
    rod_count_per_face = 11
    spacing= config['grid_detection']["mods"][-1]['spacing_mm']
    grid_heights_mm= config['grid_detection']["mods"][-1]['grid_heights_mm']
    rod_height = np.sum(spacing)
    
    bsdf_resolved  = bsdf if bsdf is not None else _pink_bsdf    
    
    rod_centers = rods_hexagon.generate_rod_centers(rod_count_per_face,rod_width_mm, gap_between_rods_width_mm)
    if has_rod_divergence:
        max_divergence_mm = 10
        return bent_rod.get_twisted_nfa_mesh(config,rod_centers,spacing,grid_heights_mm,max_divergence_mm,bsdf_resolved)
    else:
        cylinder_radius = config['measurements']['rod_width_mm']/2
        rods = [ get_rod((*rc,0),cylinder_radius,rod_height) for rc in rod_centers]

        rod_objs = []
        for i, rod in enumerate(rods[:]):
            rod['my_bsdf'] = bsdf_resolved
            rod_objs.append(rod)

        return rod_objs


config['measurements'] = {
    "rod_width_mm" : 9.1, 
    "gap_between_rods_width_mm": 3.65,
    
    #based on diagram, not blueprint -> approximate values
    "camera_distance_mm" : 430,
    "light_offset_mm": 228    
}


mesh_objs = generate_rods_group(config,True, scene_definition._pink_bsdf)

img = render_objects(mesh_objs,cam_origin =[50,2050,-500])
plt.imshow(img)


```

```python

```

```python

```

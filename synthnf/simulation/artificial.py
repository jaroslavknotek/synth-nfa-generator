import numpy as np

import synthnf.geometry.fuel_rods_mesh as frm
import synthnf.geometry.curves as curves

def create_nfa_curves(
    bow_xy_mm,
    rods_divergences_xy_mm,
    span_margin_mm,
    spans
):      
    fa_height_mm = spans[-1]
    bow_x,bow_y = bow_xy_mm
    # LIMITED to a single bent
    bezier_nodes = np.array([
        [0,0,0], #bottom
        [bow_x,bow_y,fa_height_mm*1/3],
        [0,0,fa_height_mm*2/3],[0,0,fa_height_mm*3/4],  # until 1 third there is nothing happening
        [0,0,fa_height_mm] #top
    ])
    curve_fa = curves.bezier_from_nodes(bezier_nodes)
    
    curves_rod = []
    for rod_div_xy in rods_divergences_xy_mm:
        piecewise_curve = curves.construct_piecewise_curve(spans,rod_div_xy,span_margin_mm)
        curves_rod.append(curves.merge_curves(curve_fa,piecewise_curve))
       
    return curve_fa, curves_rod


def rnd_divergence_adjusted(pieces,max_divergence,rnd = None):    
    if rnd is None:
        rnd = np.random.RandomState()
        
    heights = np.diff(pieces)
    max_div = np.max(heights)
    # ensure divergence are smaller for smaller spans
    divergence_factor =heights/max_div
    
    divergences = (rnd.rand(len(heights),2)*2-1) * max_divergence * divergence_factor[None].T
    return np.nan_to_num(divergences)
    
    
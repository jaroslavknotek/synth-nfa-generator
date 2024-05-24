import numpy as np
import bezier

import synthnf.geometry.fuel_rods_mesh as frm

def rnd_divergence_adjusted(pieces,max_divergence,rnd = None):    
    if rnd is None:
        rnd = np.random.RandomState()
        
    heights = np.diff(pieces)
    max_div = np.max(heights)
    # ensure divergence are smaller for smaller spans
    divergence_factor =heights/max_div
    
    divergences = (rnd.rand(len(heights),2)*2-1) * divergence_factor[None].T
    return np.nan_to_num(divergences)
    
def construct_piecewise_curve(pieces,divergences, piece_margin_mm):
    beziers = []

    for piece_start_mm,piece_end_mm,(div_x_mm,div_y_mm) in zip(pieces,pieces[1:],divergences):

        bezier_start_z = piece_start_mm + piece_margin_mm
        bezier_end_z = piece_end_mm - piece_margin_mm

        if bezier_start_z <= bezier_end_z:    
            middle_z = (bezier_start_z + bezier_end_z)/2

            bezier_middle_start = [0,0, bezier_start_z ]
            bezier_middle_middle = [div_x_mm, div_y_mm, middle_z]
            bezier_middle_end = [0,0, bezier_end_z]

        bezier_start = [0,0,bezier_start_z]
        bezier_end = [0,0,bezier_end_z]

        nodes_list = [
            bezier_start,
            bezier_middle_start,
            bezier_middle_middle,
            bezier_middle_end,
            bezier_end
        ]
        nodes = np.stack(nodes_list)

        bezier_piece = bezier_from_nodes(nodes)
        bezier_record = (piece_start_mm,piece_end_mm,bezier_piece)
        beziers.append(bezier_record)

    return frm.PiecewiseCurve(beziers)

def bezier_from_nodes(nodes):
    return bezier.Curve.from_nodes(nodes.T)
    
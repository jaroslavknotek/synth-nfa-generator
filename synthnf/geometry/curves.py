import numpy as np
import bezier

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
    
def straight_curve(
    from_x = None,
    from_y = None,
    from_z = None,
    to_x = None,
    to_y = None,
    to_z = None
):
    from_x = from_x if from_x is not None else 0
    from_y = from_y if from_y is not None else 0
    from_z = from_z if from_z is not None else 0
    to_x = to_x if to_x is not None else 0
    to_y = to_y if to_y is not None else 0
    to_z = to_z if to_z is not None else 1
    
    p0 = np.array([from_x,from_y,from_z])
    p1 = np.array([to_x,to_y,to_z])
    bezier_nodes = np.array([
        p0,
        (p0+p1)/2,
        p1
    ])
    return bezier_from_nodes(bezier_nodes)
    
    
def merge_curves(curve_a,curve_b):
    return CurveAdded(curve_a,curve_b)


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

    return PiecewiseCurve(beziers)

def bezier_from_nodes(nodes):
    return bezier.Curve.from_nodes(nodes.T)
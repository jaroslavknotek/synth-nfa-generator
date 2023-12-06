import numpy as np

def mirror_down(p_x,p_y,p_z,v_1,v_2,v_3):
    
    #return p_x,p_y,p_z,v_1,v_2,v_3
    #return p_x,p_y,-p_z,v_1,v_3,v_2
    n_points = len(p_x)
    p_x = np.concatenate([p_x,p_x])
    p_y = np.concatenate([p_y,p_y])
    p_z = np.concatenate([p_z,-p_z+2*np.min(p_z)])
    
    v_1_m = v_1 + n_points
    v_2_m = v_3 + n_points
    v_3_m = v_2 + n_points
    
    v_1 = np.concatenate([v_1,v_1_m])
    v_2 = np.concatenate([v_2,v_2_m])
    v_3 = np.concatenate([v_3,v_3_m])
    
    return p_x,p_y,p_z,v_1,v_2,v_3
    
def array_left(p_x,p_y,p_z,v_1,v_2,v_3,n=1):
    
    if n==0:
        return p_x,p_y,p_z,v_1,v_2,v_3

    n_points = len(p_x)
    
    x_min = np.min(p_x)
    x_max = np.max(p_x)
    width = x_max - x_min
    width_shifts = -np.arange(0,n+1) *width
    
    p_x = np.concatenate(np.stack([p_x]*(n+1)) + width_shifts[:,None])

    p_y = np.concatenate([p_y]*(n+1))
    p_z = np.concatenate([p_z]*(n+1))
    
    v_1 = np.concatenate([v_1 + n_points*i for i in range(0,n+1)])
    v_2 = np.concatenate([v_2 + n_points*i for i in range(0,n+1)])
    v_3 = np.concatenate([v_3 + n_points*i for i in range(0,n+1)])
    
    return p_x,p_y,p_z,v_1,v_2,v_3

def rotate_2d(deg, points):
    assert points.shape[0] == 2, "array should have 2 rows and n columns"
    angle = np.deg2rad(deg)
    s = np.sin(angle)
    c = np.cos(angle)

    rot = np.array([[c,s],[-s,c]])
    return np.dot(rot,points)   

def array_hexagon(dist, p_x,p_y,p_z,v_1,v_2,v_3):
    
    deg = 60
    n_points = len(p_x)
    p_x = p_x - np.mean(p_x)
    p_y = p_y - np.mean(p_y) + dist
    p_z = p_z - np.mean(p_z)
    
    for i in range(1,6):
        r_p_x,r_p_y = rotate_2d(
            deg*i,
            np.stack([p_x[:n_points],p_y[:n_points]])
        )
       
        p_x = np.concatenate([p_x,r_p_x])
        p_y = np.concatenate([p_y,r_p_y])
        
    p_z = np.concatenate([p_z]*6)
        
    v_1 = np.concatenate([v_1 + n_points*i for i in range(0,6)])
    v_2 = np.concatenate([v_2 + n_points*i for i in range(0,6)])
    v_3 = np.concatenate([v_3 + n_points*i for i in range(0,6)])
        
    return p_x,p_y,p_z,v_1,v_2,v_3


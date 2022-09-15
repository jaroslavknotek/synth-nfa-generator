import numpy as np

def add_row(start_x,start_y, n,rod_center_distance_mm):
    pts_x = np.arange(n) * rod_center_distance_mm + start_x
    pts_y = np.ones((n,)) * start_y
    return np.stack([pts_x,pts_y]).T


def generate_rod_centers(rod_count, config):
    rod_width_mm = config['measurements']['rod_width_mm']
    gap_between_rods_width_mm = config['measurements']['gap_between_rods_width_mm']

    rod_center_distance_mm = rod_width_mm + gap_between_rods_width_mm

    pythagorean_factor = np.sqrt(3)/2
    rod_centers_lines_arrays = [add_row(0,0,rod_count*2 - 1,rod_center_distance_mm) ]
    x_shift = rod_centers_lines_arrays[0][rod_count-1]
        
    for i,current_rod_count in zip(range(1,rod_count),range(rod_count+rod_count - 2,rod_count-1,-1)):

        x = (i*rod_center_distance_mm)/2
        y = i * pythagorean_factor* rod_center_distance_mm
        points_b = add_row(x,-y,current_rod_count,rod_center_distance_mm)
        points_t = add_row(x,y,current_rod_count,rod_center_distance_mm)

        rod_centers_lines_arrays.append(points_b)
        rod_centers_lines_arrays.append(points_t)
    return np.concatenate(rod_centers_lines_arrays) - x_shift

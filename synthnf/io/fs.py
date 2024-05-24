import pathlib
import tempfile
import imageio
import numpy as np

def create_temp_folder():
    temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    temp_dir.mkdir(parents=True,exist_ok=True)
    return temp_dir

def save_mesh_to_temp(mesh):
    temp_dir = create_temp_folder()
    mesh_path = str(temp_dir/'grid.ply')
    mesh.write_ply(mesh_path)
    return mesh_path

def save_float_image_to_temp(img,filename='image.png',temp_dir = None):
    if temp_dir is None:
        temp_dir = create_temp_folder()
        
    path = temp_dir/'image.png'
    imageio.imwrite(
        path,
        np.uint8(img *255)
    )
    return path

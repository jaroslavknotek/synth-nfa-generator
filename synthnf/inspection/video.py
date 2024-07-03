from tqdm.auto import tqdm
import numpy as np
import pathlib

import synthnf.io as io

import cv2
import json
import logging
import shutil
import matplotlib.pyplot as plt

logger = logging.getLogger('synthnf')


def run_inspection_all_sides(out_folder,inspection,frames_num,odd_top_down=True):
    for i in range(6):    
        out_folder_side = out_folder/f'F{i+1}'
        top_down = True
        
        if odd_top_down:
            # Side numbering starts with 1 e.g. zero based i is even numbered e.g.:
            top_down = i%2 == 0
        logger.info("Creating side %d",i+1)
        run_inspection(out_folder_side,inspection,frames_num,face_num=i+1,top_down=top_down)
        plt.close()
        
def run_inspection(directory,inspection,n_frames,top_down = True,face_num=1):
    directory = pathlib.Path(directory)
    directory.mkdir(exist_ok=True,parents=True)
    
    logger.info("Collecting metadata")
    metadata = inspection.collect_metadata(n_frames,top_down=top_down,face_num=face_num)
    metadata['top_down'] = top_down
    metadata['face_num'] = face_num
    metadata['frames_number'] = n_frames
    save_metadata(directory/'metadata',metadata)
    
    logger.info("Creating video")
    
    create_video(
        directory/'frames',
        directory/'video.mp4',
        inspection,
        n_frames,
        fps=25,
        top_down=top_down
    )

    
def create_video(frame_folder,video_path,inspection,n_frames,top_down = True,fps = None):
    if video_path.exists():
        logger.info("Video exists. Skipping")
        return
    
    # fa_height_mm = inspection.fa_height_mm
    # zs = np.linspace(0,fa_height_mm,n_frames)
    _,_,zs = inspection.curve_fa.evaluate_multi(np.linspace(0,1,n_frames))
    
    if top_down:
        zs = np.flip(zs)
    zs_i = list(enumerate(zs))
    frame_folder.mkdir(exist_ok=True,parents=True)
    idx_start=len(list(frame_folder.glob("*.png")))    
    if idx_start > 0:
        logger.warning(f"Found existing frames. Continuing from {idx_start}")
        zs_i = zs_i[idx_start:]

    for i,z in tqdm(zs_i):
        frame = inspection.render_frame(z)
        path = frame_folder/f"frame_{i:04}_{int(z):04}.png"
        io.imwrite(path,frame)
        
    images_to_video(frame_folder,video_path,fps=None)
    
        

def images_to_video(input_folder,video_filepath = None,fps = None):
    
    if not video_filepath:
        video_filepath = input_folder/'video.mp4'
        
    fps = fps or 25 
        
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    frames_filepaths = list(sorted(input_folder.glob('*.png')))
    fp = str(    frames_filepaths[0])
    frame_height,frame_width = cv2.imread(fp).shape[:2]
    
    frames = map(cv2.imread,[str(p) for p in frames_filepaths])
    
    frame_number = len(frames_filepaths)
    frames = tqdm(frames,desc ='Making a video',total=frame_number)
    writer = cv2.VideoWriter(
        str(video_filepath), 
        fourcc, 
        fps, 
        (frame_width,frame_height)
    )
    for f in frames:
        writer.write(f)
        
    return video_filepath

def save_metadata(meta_dir,metadata):
    meta_dir.mkdir(exist_ok=True,parents=True)
    metadata['shifts'].to_csv(meta_dir/'shifts.csv',index=False)
    io.imwrite(meta_dir/'shrunk.png',metadata['shrunk'])
    
    metadata['bow_fig'].savefig(meta_dir/'bow_div.png')
    
    res = {
        'cam_speed_mm_per_s':metadata['cam_speed'],
        "rod_growth_mm":metadata['displacement']
    }
    
    with open(meta_dir/'results.json','w') as f:
        json.dump(res,f)
    
    shutil.copy(metadata['model_filepath'],meta_dir/'fa_model.ply')
    
    with open(meta_dir/'simulation_parameters.json','w') as f:
        json.dump(metadata['simulation_parameters'],f)

import drjit as dr

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
mi.set_log_level(mi.LogLevel.Error)

import pathlib
import argparse
import json
import logging
import synthnf.inspection.video as vid
import synthnf.io as io
import synthnf.simulation as simu
import synthnf.utils as utils


logging.basicConfig()
logger = logging.getLogger('synthnf')

logger_re = logging.getLogger('synthnf.renderer')

import synthnf.inspection as ins


import numpy as np

def generate_random_params():
    random_seed = np.random.randint(2**31)
    
    rnd = np.random.RandomState(seed = random_seed)
    
    mean = [.7, .5]
    cov = [
        [1, .5], 
        [.5, 1 ],
    ]  # diagonal covariance

    rand_points = rnd.multivariate_normal(mean, cov, 100)

    rand_points=utils.normalize(rand_points)
    cloudiness, max_bow = rand_points[50]

    cloudiness = .5 + cloudiness/2
    max_div = 5*max_bow
    max_dis = 10*max_bow
    max_bow = 150*max_bow
    
    return simu.RandomParameters(
        max_bow_mm = max_bow,
        max_divergence_mm=max_div,
        max_z_displacement_mm=max_dis,
        n_textures=21,
        cloudiness=cloudiness,
        seed = random_seed
    )

def create_inspection(out_folder,params_dict,n_frames,force):
    if not force and out_folder.exists():
        raise Exception(f"Output folder already exists. Use '--force' flag to override behavior. Folder:{out_folder.absolute()}")

    simulation_model = simu.RandomParameters(**params_dict)
    logger.info("Generating inspection scene with seed %d", simulation_model.seed)
    inspection = ins.FAInspection(simulation_model)
    vid.run_inspection_all_sides(out_folder,inspection,n_frames)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Generate FA inspection videos')

    parser.add_argument('destination',help='Folder to which videos will be genereated.',type=pathlib.Path)
    parser.add_argument('--config',help='Folder to which videos will be genereated.',type=pathlib.Path,required=True)
    parser.add_argument('-n','--frames_number',help='The number of frames each video should have',required=True,type=int)
    parser.add_argument('-f', '--force',action='store_true',help='Generation proceeds even if the destination folder already exists')
    return parser

    
if __name__ == '__main__':

    parser = setup_argparse()
    args = parser.parse_args()
    
    # todo log level through argparse
    logger.setLevel(logging.INFO)
    mi.set_log_level(mi.LogLevel.Error)
    
    params_dict = io.load_json(args.config)
    
    create_inspection(
        args.destination,
        params_dict,
        args.frames_number,
        args.force
    )



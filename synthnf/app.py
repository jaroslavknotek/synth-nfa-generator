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

def create_inspection(out_folder,params_dict,n_frames,force):
    if not force and out_folder.exists():
        raise Exception(f"Output folder already exists. Use '--force' flag to override behavior. Folder:{out_folder.absolute()}")

    swing_cam_above_mm = params_dict.get('swing_cam_above_mm',0)
    del params_dict['swing_cam_above_mm']
    
    simulation_model = simu.RandomParameters(**params_dict)
    logger.info("Generating inspection scene with seed %d", simulation_model.seed)
    inspection = ins.SwingSupressingInspection(simulation_model,swing_cam_above_mm)
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



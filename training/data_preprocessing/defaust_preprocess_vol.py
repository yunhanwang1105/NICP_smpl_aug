# Automatic building of cython extensions
import numpy as np
import pyximport
pyximport.install(
    setup_args={
        "include_dirs": [np.get_include(), "./utils/libvoxelize"], 
        "script_args": ["--cython-cplus"]
    }, reload_support=True, language_level=3
)

import argparse
import os
import traceback
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool

import torch
import trimesh
from tqdm import tqdm

from utils.voxels import VoxelGrid
from utils.parallel_map import parallel_map
import glob
import shutil

## SET RESOLUTION OF THE VOXELIZATION
res = 64 

faces = np.load("./assets/faces.npy")


### VOXELIZATION UTILS  ####

def voxelize(path, res):
    output_file = os.path.dirname(path) + '/vox_{}.npy'.format(res)
    try:
        if os.path.exists(output_file):
            return

        mesh = trimesh.load(path , process=False)
        occupancies = VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(output_file, occupancies)

    except Exception as err:
        path = os.path.normpath(path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(path))


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix

def voxelize_distance(v, res, mesh_faces,OUT_PATH_OCC,OUT_PATH_SCAL ):      
    vertices = v[:, 0:3]
    idx = int(v[:, 3][0])
    
    mesh = trimesh.Trimesh(vertices = vertices,faces = faces)
    resolution = res # Voxel resolution
    b_min = np.array([-0.8, -0.8, -0.8]) 
    b_max = np.array([0.8, 0.8, 0.8])
    step = 5000
    
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) /2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1/total_size)
    
    vertices = mesh.vertices
    factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        points_npy = coords.reshape(3, -1).T
        iters = len(points)//step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it*step:(it+1)*step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
        signed_distance = np.concatenate(all_distances)
    del v 
    del coords 
    
    voxels = signed_distance.reshape(resolution, resolution, resolution)
    
    ## Save voxels and aligned vertices    
    torch.save(voxels,OUT_PATH_OCC / str(f'{idx:09}.pt') )
    torch.save(vertices,OUT_PATH_SCAL/ str(f'{idx:09}.pt')  )


def my_voxelize_distance(v_mesh, res, orginal_mesh):      
    
    resolution = res # Voxel resolution
    b_min = np.array([-0.8, -0.8, -0.8]) 
    b_max = np.array([0.8, 0.8, 0.8])
    step = 5000
    
    total_size = (orginal_mesh.bounds[1] - orginal_mesh.bounds[0]).max()
    centers = (orginal_mesh.bounds[1] + orginal_mesh.bounds[0]) /2


    v_mesh.apply_translation(-centers)
    v_mesh.apply_scale(1/total_size)
    
    vertices = v_mesh.vertices
    factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        iters = len(points)//step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it*step:(it+1)*step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
        signed_distance = np.concatenate(all_distances)
    del v 
    del coords 
    
    voxels = signed_distance.reshape(resolution, resolution, resolution)
    
    # ## Save voxels and aligned vertices    
    # torch.save(voxels, str('noised_voxels.pt'))
    # torch.save(vertices, str('noised_verts.pt'))
    return voxels, vertices
    
##############

def main(cfg):

    minimal_pcd_path = '/mnt/qb/work/ponsmoll/pba594/ml_proj/smpl_aug/outdir/mnt_minimal/qb/work/ponsmoll/pba594/data/amass/DFaust_67/'
    clothed_pcd_path = '/mnt/qb/work/ponsmoll/pba594/ml_proj/smpl_aug/outdir/mnt_clothed/qb/work/ponsmoll/pba594/data/amass/DFaust_67/'

    #########################################################
    smpl_aug_pcd_path_used = minimal_pcd_path if cfg.is_minimal else clothed_pcd_path
    pcd_type_flag = 'point_cloud_gt' if cfg.is_gt else 'point_cloud_noised'
    aug_flag_true = cfg.is_aug
    test_subject = '50027' # used to benchmark in smplaug paper
    #########################################################
    DFaust_path = '/mnt/qb/work/ponsmoll/pba594/data/amass/DFaust_67/*/*.npz'
    save_pcd_path = '/mnt/qb/work/ponsmoll/pba594/data/dfaust_NICP/'
    dfaust_motion_paths = glob.glob(DFaust_path)
    dfaust_motion_names = []
    for motion_path in dfaust_motion_paths: # "/mnt/qb/work/ponsmoll/pba594/data/amass/DFaust_67/50027/50027_shake_shoulders_poses.npz"
        temp_motion_name = motion_path.split("DFaust_67/")[1] # '50027/50027_shake_shoulders_poses.npz'
        motion_name = temp_motion_name.split(".npz")[0] # '50027/50027_shake_shoulders_poses'
        if motion_name.endswith('shape'):
            continue
        dfaust_motion_names.append(motion_name)
    
    dfaust_motion_names.sort()
    print("There are ", len(dfaust_motion_names), " motions.")
    print("Use minimal surface: ", str(args.is_minimal))
    print("Use gt instead of noise: ", str(args.is_gt))
    print("Use pose augmentation: ", str(args.is_aug))

    # start processing
    tqdm_bar = tqdm(cfg.datasets, total=len(cfg.datasets), ncols=80)
    for DATASET in tqdm_bar: # vald, train, test
        tqdm_bar.set_description(f"Processing {DATASET}")
        idx = 0

        test_motion_names = [s for s in dfaust_motion_names if s.startswith(test_subject)] # test set
        train_motion_names = list(set(dfaust_motion_names) - set(test_motion_names)) # train set

        motion_names = train_motion_names if DATASET == 'train' else test_motion_names # ['50027/50027_shake_shoulders_poses']

        OUT_PATH_OCC = args.input_path / cfg.exp / 'stage_III' / DATASET / 'ifnet_indi' / "occ_dist"
        OUT_PATH_SCAL = args.input_path / cfg.exp /'stage_III' / DATASET / 'ifnet_indi' / "verts_occ_dist"
        OUT_PATH_BODY_DATA = args.input_path / cfg.exp /'stage_III' / DATASET / 'ifnet_indi' / "body_data"
        OUT_PATH_SMPL_VERTS = args.input_path / cfg.exp /'stage_III' / DATASET / 'ifnet_indi' / "smpl_verts"

        OUT_PATH_OCC.mkdir(parents=True, exist_ok=True)
        OUT_PATH_SCAL.mkdir(parents=True, exist_ok=True)
        OUT_PATH_BODY_DATA.mkdir(parents=True, exist_ok=True)
        OUT_PATH_SMPL_VERTS.mkdir(parents=True, exist_ok=True)
        
        for cur_motion_name in motion_names:
            cur_motion_pcd_path = smpl_aug_pcd_path_used + cur_motion_name + '/kinect/' + pcd_type_flag + '/'
            cur_motion_mesh_path = smpl_aug_pcd_path_used + cur_motion_name + '/kinect/' + 'body_meshes/'
            cur_motion_body_data_path = smpl_aug_pcd_path_used + cur_motion_name + '/kinect/' + 'body_data/'

            directory = cur_motion_pcd_path
            ply_file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] # collect all frame file names
 
            if not aug_flag_true:
                ply_file_names = [p for p in ply_file_names if not p.endswith("_aug.ply")] # remove all _aug.ply, no augmentation

            for cur_ply_file in ply_file_names: # for each frame: 0000.ply, 0100.ply
                pcd_path = cur_motion_pcd_path + cur_ply_file
                mesh_path = cur_motion_mesh_path + cur_ply_file.replace(".ply", ".obj")
                body_data_path = cur_motion_body_data_path + cur_ply_file.replace(".ply", ".npy")

                print(pcd_path)

                vertices_mesh = trimesh.load(pcd_path)
                original_mesh = trimesh.load(mesh_path)
                voxels, vertices = my_voxelize_distance(vertices_mesh, res, original_mesh)

                torch.save(voxels, OUT_PATH_OCC / str(f'{idx:09}.pt'))
                torch.save(vertices, OUT_PATH_SCAL/ str(f'{idx:09}.pt'))
                torch.save(original_mesh.vertices, OUT_PATH_SMPL_VERTS / str(f'{idx:09}.pt'))

                # save body data
                cur_body_data = np.load(body_data_path, allow_pickle=True)
                np.save(OUT_PATH_BODY_DATA / str(f'{idx:09}.pt'), cur_body_data)
                # shutil.copy2(body_data_path, OUT_PATH_BODY_DATA)

                # del cur_body_data
                del voxels
                del vertices
                del vertices_mesh
                del original_mesh

                idx += 1





if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data voxelization")

    parser.add_argument("--exp", "-e", type=str, default='V1_SV1_T5', help="Experiment name")
    parser.add_argument("--datasets", "-d", type=str, default='vald', nargs="+", choices=["train", "vald", "test"], help="Dataset name")
    parser.add_argument("--input_path", "-i", type=Path, default=Path('/mnt/qb/work/ponsmoll/pba594/data/dfaust_NICP'), help="Path to input folder / where to save processed data")
    parser.add_argument("--is_minimal", action='store_true')
    parser.add_argument("--is_aug", action='store_true')
    parser.add_argument("--is_gt", action='store_true')

    args = parser.parse_args()
    ##########
    # args.datasets = list(set(args.datasets))
    args.datasets = ["train", "vald", "test"]
    ##########

    main(args)

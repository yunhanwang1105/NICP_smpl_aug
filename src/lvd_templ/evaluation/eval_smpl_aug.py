import os
import glob
import hydra
import numpy as np
import omegaconf
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra import compose, initialize
import time   
import tqdm
import torch
import trimesh
import gc

from utils_cop.prior import MaxMixturePrior
from utils_cop.SMPL import SMPL

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO

from lvd_templ.paths import chk_pts, home_dir, output_dir as out_folder, path_demo, path_demo_guess_rot, path_FAUST_train_reg, path_FAUST_train_scans
from lvd_templ.data.datamodule_AMASS import MetaData
from lvd_templ.evaluation.utils import vox_scan, vox_scan_smpl_aug, fit_LVD, selfsup_ref, SMPL_fitting, fit_cham, fit_plus_D

import warnings
warnings.filterwarnings("ignore")

## Device
device = torch.device('cuda')

smpl_aug_path = '/mnt/qb/work/ponsmoll/pba594/data/df_NICP/'

def export_mesh(T, r,t,s, path):
        T.apply_scale(s)
        T.apply_translation(t)
        T.apply_transform(r)
        
        k = T.export(path)
        return

## Here you can set the path for different datasets
def get_dataset(name, version):
    if name=='smpl_aug':
        return smpl_aug_path + version + '/stage_III/test/ifnet_indi/' 

    raise ValueError('this challenge does not exists')

## Function to load checkpoint
def get_model(chk):
    # Recovering the Path to the checkpoint
    chk_zip = glob.glob(chk + 'checkpoints/*.zip')[0]
    
    # Restoring the network configurations using the Hydra Settings
    tmp = hydra.core.global_hydra.GlobalHydra.instance().clear()
    tmp = initialize(config_path="../../../" + str(chk))
    cfg_model = compose(config_name="config")
    
    # Recovering the metadata
    train_data = hydra.utils.instantiate(cfg_model.nn.data.datasets.train, mode="test")
    MD = MetaData(class_vocab=train_data.class_vocab)
    
    # Instantiating the correct nentwork
    model: pl.LightningModule = hydra.utils.instantiate(cfg_model.nn.module, _recursive_=False, metadata=MD)
    
    # Restoring the old checkpoint
    old_checkpoint = NNCheckpointIO.load(path=chk_zip)
    module = model._load_model_state(checkpoint=old_checkpoint, metadata=MD).to(device)
    module.model.eval()
    
    return module, MD, train_data, cfg_model

# Main Method to register all the shapes in a folder
def run(cfg: DictConfig) -> str:
    os.chdir(home_dir)

    # Recovering the parameters of the run
    model_name = cfg['core'].checkpoint
    chk = chk_pts + model_name + '/'
    model_name = model_name

    # Create Output Folders
    if not(os.path.exists(out_folder + model_name)):
        os.mkdir(out_folder + model_name)
        
    out_dir = out_folder + model_name + '/' + cfg['core'].challenge 
     
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    
    # Recover Data Path
    path_in = get_dataset(cfg['core'].challenge, cfg['core'].version)
    
    # How the data are organized
    scans = glob.glob(path_in + 'verts_occ_dist/' + '*.pt')

    print('--------------------------------------------')
    print(f'List of target scans: {scans}')
    
    # You can add an initial rotation for the shapes to align
    # The axis.This one works for the FAUST shapes
    
    origin, xaxis = [0, 0, 0], [1, 0, 0]
    if cfg['core'].challenge in ('demo','demo_guess_rot'):
        alpha = np.pi/2 #0
    else:
        alpha = np.pi/2
      
    ### Get SMPL model
    SMPL_model = SMPL('neutral_smpl_with_cocoplus_reg.txt', obj_saveable = True).cuda()
    prior = MaxMixturePrior(prior_folder='utils_cop/prior/', num_gaussians=8) 
    prior.to(device)
    
    #### Restore Model
    module, MD, train_data, cfg_model = get_model(chk)
    module.cuda()
    
    ### Get Resolution and GT_IDXS of the experiment
    res = MD.class_vocab()['occ_res']
    gt_points = MD.class_vocab()['gt_points']
    gt_idxs = train_data.idxs
    data_type = cfg_model['nn']['data']['datasets']['type']
    grad = cfg_model['nn']['module']['grad']

    smpl_fit_v2v_list = []
    smpl_fit_j2j_list = []
    cham_fit_v2v_list = []
    cham_fit_j2j_list = []
    plusD_v2v_list = []
 
    print('--------------------------------------------')
    print('Version: ', cfg['core'].version)
    print('--------------------------------------------')               
    ### REGISTRATIONS FOR ALL THE INPUT SHAPES
    for scan in tqdm.tqdm(scans,desc="Scans:"):
        print('--------------------------------------------')
                
        ### PRELIMINARIES: LOAD MODEL, LOAD SHAPE, SET CONFIGURATIONS OF THE REGISTRATION
        print(f"Start :{scan}")
        
        # Basic Name --> You can add "tag" if you want to differentiate the runs
        out_name = 'out' + cfg['core'].tag
        
        # Scans name format
        # if(cfg['core'].challenge == 'demo'):
        #     name = os.path.basename(os.path.dirname(scan))
        # else:
        name = os.path.basename(scan)[:-3] # e.g. 000000046
         
        # If we want to use the Neural ICP Refinement    
        if cfg['core'].ss_ref:
            del module, train_data
            module, MD, train_data, cfg_model = get_model(chk)
        
        # Read input shape       
        ###############
        scan_vertices = torch.load(scan)
        scan_src_pcd = trimesh.points.PointCloud(scan_vertices)

        # Read occ dist
        occ_dist_path = scan.replace("verts_occ_dist", "occ_dist")
        occ_dist = torch.load(occ_dist_path)

        # Read body data
        body_data_path = scan + ".npy"
        body_data_path = body_data_path.replace("verts_occ_dist", "body_data")
        body_data = np.load(body_data_path, allow_pickle=True).item()

        gt_body_pose = torch.cat((body_data['global_orient'], body_data['body_pose']), dim=1).cuda()
        gt_verts, gt_joints, gt_Rs = SMPL_model.forward(theta=gt_body_pose, beta=body_data['betas'].cuda(), get_skin=True)

        gt_joints = gt_joints[0].detach().cpu().numpy()

        # Read SMPL
        smpl_verts_path = scan.replace("verts_occ_dist", "smpl_verts")
        smpl_verts = torch.load(smpl_verts_path)

        ###############
        
        # EXPERIMENTAL FEATURE: you can use NICP loss to guess the best rotation for the input shape

        # if cfg['core'].guess_rot==True:   
        #     print("Seeking for the rotation that minimizes NICP...")
        #     best_loss = np.inf
        #     set_axis = [[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 1, 1],[1, 1, 0],[1, 0, 1]]
        #     set_angles = np.linspace(0,2*np.pi,5)
        #     for ax in set_axis:
        #         for al in set_angles:
        #             Rx = trimesh.transformations.rotation_matrix(al, ax)
        #             with torch.no_grad():
        #                 mesh_src_copy = scan_src.copy()
        #                 voxel_src, mesh_src, scale, trasl = vox_scan(mesh_src_copy, res, style=data_type, grad=grad)
                        
        #                 # Extract Features
        #                 module.model(voxel_src)
        #                 input_points = torch.tensor(np.asarray(mesh_src.vertices))
        #                 factor = max(1, int(input_points.shape[0] / 20000))
        #                 input_points = input_points[torch.randperm(input_points.size()[0])]
        #                 input_points_res = input_points[1:input_points.shape[0]:factor,:].type(torch.float32).unsqueeze(0).cuda()
                                        
        #                 # Query points on the target surface
        #                 pred_dist = module.model.query(input_points_res)
                        
        #                 pred_dist = pred_dist.reshape(1, gt_points, 3, -1).permute(0, 1, 3, 2)
                        
        #                 # Collect the offset with the minimum norm for each target vertex
        #                 v, _ = torch.min(torch.sum(pred_dist**2,axis=3),axis=1)
                        
        #                 # Global loss
        #                 if  torch.sum(v)<best_loss:
        #                     best_loss = torch.sum(v)
        #                     best_alpha = al
        #                     best_axis = ax
        #     Rx = trimesh.transformations.rotation_matrix(al, ax)
        #     inv_Rx = trimesh.transformations.rotation_matrix(-al, ax)
        #     print("Done!")
        # else:
        #     Rx = trimesh.transformations.rotation_matrix(alpha, xaxis)
        #     inv_Rx = trimesh.transformations.rotation_matrix(-alpha, xaxis)
        
        # # Canonicalize the input point cloud and prepare input of IF-NET
        # scan_src.apply_transform(Rx)
        # voxel_src, mesh_src, scale, trasl = vox_scan(scan_src, res, style=data_type, grad=grad)

        voxel_src = vox_scan_smpl_aug(occ_dist, res, style=data_type, grad=grad)
                
        # Save algined mesh
        if not(os.path.exists(out_dir +'/'+ name)):
           os.mkdir(out_dir +'/'+ name)
        
        # if not(cfg['core'].scaleback): 
        trasl = np.array([0.0, 0.0, 0.0]).reshape(3)
        scale = 1
        inv_Rx = np.eye(4)
        
        scan_src_pcd.export(out_dir +'/'+ name + '/target.ply')

        # export_mesh(mesh_src.copy(), inv_Rx, trasl, scale, out_dir +'/'+ name + '/target.ply')               
        # k = mesh_src.export(out_dir +'/'+ name + '/aligned.ply')
        
        #######
        
        # IF N-ICP is requested, run it
        if cfg['core'].ss_ref:
            # We add a name to specify the NF-ICP is performed
            out_name = out_name + '_ss'
            et = time.time()
            module.train()
            selfsup_ref(module, torch.tensor(np.asarray(scan_src_pcd.vertices)), voxel_src, gt_points,steps=cfg['core'].steps_ss, lr_opt=cfg['core'].lr_ss)
            module.eval()

        # You can initialize LVD in different points in space. Default is at the origin
        if cfg['core'].init:
            picker = np.int32(np.random.uniform(0,len(scan_src_pcd.vertices),gt_points))
            init = torch.unsqueeze(torch.tensor(np.asarray(scan_src_pcd.vertices[picker]),dtype=torch.float32),0)
        else:
            init = torch.zeros(1, gt_points, 3).cuda()
        
        # Fit LVD
        reg_src =  fit_LVD(module, gt_points, voxel_src, iters=cfg['lvd'].iters, init=init)
            
        # FIT SMPL Model to the LVD Prediction
        out_s, params = SMPL_fitting(SMPL_model, reg_src, gt_idxs, prior, iterations=2000)
        params_np = {}
        for p in params.keys():
            params_np[p] = params[p].detach().cpu().numpy()

        smpl_fit_v2v = 100 * np.sqrt((np.asarray(out_s - smpl_verts)**2).sum(axis=-1)).mean()
        print("SMPL_fitting verts diff: ", smpl_fit_v2v)
        smpl_fit_v2v_list.append(smpl_fit_v2v)

        smpl_fit_j2j = 100 * np.sqrt(((params_np['fit_joint'] - gt_joints)**2).sum(axis=-1)).mean()
        print("SMPL_fitting joint diff: ", smpl_fit_j2j)
        smpl_fit_j2j_list.append(smpl_fit_j2j)
        
            
        # Save intermidiate output 
        # NOTE: You may want to remove this if you are interested only
        # in the final registration
        T = trimesh.Trimesh(vertices = out_s, faces = SMPL_model.faces) 
        export_mesh(T.copy(), inv_Rx, trasl, scale, out_dir +'/'+ name + '/' + out_name + '.ply')   
        np.save(out_dir +'/'+ name + '/loss_' + out_name + '.npy',params_np)
        
        # SMPL Refinement with Chamfer            
        if cfg['core'].cham_ref:
            # Mark the registration as Chamfer Refined
            out_name = out_name + '_cham_' + str(cfg['core'].cham_bidir)
            
            # CHAMFER REGISTRATION
            # cham_bidir = 0  -> Full and clean input
            # cham_bidir = 1  -> Partial input
            # cham_bidir = -1 -> Noise input
            out_cham_s, params = fit_cham(SMPL_model, out_s, scan_src_pcd.vertices, prior,params,cfg['core'].cham_bidir)
            
            # Save Output
            T = trimesh.Trimesh(vertices = out_cham_s, faces = SMPL_model.faces)
            export_mesh(T.copy(), inv_Rx, trasl, scale, out_dir +'/'+ name + '/' + out_name + '.ply')   
            
            # DEBUG: Save some params of the fitting to check quality of the registration          
            for p in params.keys():
                params[p] = params[p].detach().cpu().numpy()            
            # np.save(out_dir +'/'+ name + '/loss_'+ out_name + '.npy',params)
        
            cham_fit_v2v = 100 * np.sqrt((np.asarray(out_cham_s - smpl_verts)**2).sum(axis=-1)).mean()
            print("Cham_fitting verts diff: ", cham_fit_v2v)
            cham_fit_v2v_list.append(cham_fit_v2v)
            
            cham_fit_j2j = 100 * np.sqrt(((params['fit_joint'] - gt_joints)**2).sum(axis=-1)).mean()
            print("Cham_fitting joint diff: ", cham_fit_j2j)
            cham_fit_j2j_list.append(cham_fit_j2j)
            # Update the name
            out_s = out_cham_s
        
        # SMPL Refinement with +D

        # if cfg['core'].plusD:
        #     smpld_vertices, faces, params = fit_plus_D(out_s, SMPL_model, scan_src_pcd.vertices, subdiv= 1, iterations=300)
        #     T = trimesh.Trimesh(vertices = smpld_vertices, faces = faces)
        #     out_name_grid = out_name + '_+D'
        #     export_mesh(T.copy(), inv_Rx, trasl, scale, out_dir +'/'+ name + '/' + out_name_grid + '.ply') 

        #     # plusD_v2v = 100 * np.sqrt((np.asarray(smpld_vertices - np.asarray(scan_src_pcd.vertices))**2).sum(axis=-1)).mean()
        #     # print("+D fitting verts diff: ", plusD_v2v)
        #     # plusD_v2v_list.append(plusD_v2v)

        gc.collect()
    
    print()
    print('--------------------final-------------------')
    print("SMPL Fitting v2v error: ", sum(smpl_fit_v2v_list) / len(smpl_fit_v2v_list))
    print("SMPL Fitting j2j error: ", sum(smpl_fit_j2j_list) / len(smpl_fit_j2j_list))
    print("Chamfer Fitting v2v error: ", sum(cham_fit_v2v_list) / len(cham_fit_v2v_list))
    print("Chamfer Fitting j2j error: ", sum(cham_fit_j2j_list) / len(cham_fit_j2j_list))






        
@hydra.main(config_path=str(PROJECT_ROOT / "conf_test"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
    


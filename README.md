# Train NICP on SMPLAug data

## 1. Generate SMPLAug dataset
Please follow the instruction in the [SMPL AUGMENTATION LIBRARY](https://github.com/yunhanwang1105/smpl_aug) to generate data of different augmentations.

Augmentations types:

minimal -> Minimally clothed point cloud (Baseline)

minimal_aug -> Minimally clothed point cloud + pose augmentation

minimal_noise -> Minimally clothed point cloud + Kinect noise

minimal_noise_aug -> Minimally clothed point cloud + pose augmentation + Kinect noise

clothed -> Clothed point cloud 

clothed_aug -> Clothed point cloud + pose augmentation

clothed_noise -> Clothed point cloud + Kinect noise

clothed_noise_aug -> Clothed point cloud + pose augmentation + Kinect noise

## 2. Preprocess SMPLAug data for NICP
Set up the data preprocessing environment following the instruction at `training/data_preprocessing/README.md` and activate the environment.

Set SMPLAug data paths (*_pcd_path), data saving path (`--input_path`) and `DFaust_path` in `training/data_preprocessing/defaust_preprocess_vol.py`

Example commands:
minimal: `python defaust_preprocess_vol.py -e minimal --is_minimal --is_gt`

minimal_aug: `python defaust_preprocess_vol.py -e minimal_aug --is_minimal --is_aug --is_gt`

minimal_noise: `python defaust_preprocess_vol.py -e minimal_noise --is_minimal`

minimal_noise_aug: `python defaust_preprocess_vol.py -e minimal_noise_aug --is_minimal --is_aug`

clothed: `python defaust_preprocess_vol.py -e clothed --is_gt`

clothed_aug: `python defaust_preprocess_vol.py -e clothed_aug --is_aug --is_gt`

clothed_noise: `python defaust_preprocess_vol.py -e clothed_noise`

clothed_noise_aug: `python defaust_preprocess_vol.py -e clothed_noise_aug --is_aug`

## 3. Train on different settings

Set up the training environment following the instruction at `training/README.md` and activate the environment.

Edit the data, project root, and body model paths at:
```
training/src/lvd_templ/run.py
training/src/lvd_templ/data/datamodule_AMASS.py
training/src/lvd_templ/data/dataset_AMASS_occ.py
training/src/lvd_templ/paths.py
```

Change data_path and datasets.version to the augmentation type accordingly in `training/conf_ifnet/nn`

Example command:
```
export PYTHONPATH=. python ./src/lvd_templ/run.py core.tags=['clothed_aug']
```
## 4. Evaluate on different settings
Set up the evaluation environment following the [instruction](https://github.com/riccardomarin/NICP) and activate the environment.

Set the trained model checkpoint obtained from the previous step. 

Example command:
```
export PYTHONPATH=. python ./src/lvd_templ/evaluation/eval_smpl_aug.py core.version='minimal_noise' core.checkpoint='wr7o4grh' -m 
```

<p align="center">
  <h1 align="center"> NICP: Neural ICP Fields for 3D Human Registration at Scale
 </h1>
 <p align="center">
    <a href="https://riccardomarin.github.io/"><strong>Riccardo Marin</strong></a>
    ·
    <a href="https://www.iri.upc.edu/people/ecorona/"><strong>Enric Corona</strong></a>
    .
    <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html"><strong>Gerard Pons-Moll</strong></a>
  </p>
  <h2 align="center"> </h2>
  <div align="center">
    <img src="assets/myteaser3.png" alt="teaser" width="100%">
  </div>
</p> 

 [[arXiv]](https://arxiv.org/abs/2312.14024)
```bibtex
@article{marin2024nicp,
      title={NICP: Neural ICP for 3D Human Registration at Scale}, 
      author={Riccardo Marin and Enric Corona and Gerard Pons-Moll},
      journal={arXiv preprint arXiv:2312.14024},
      year={2024},
}
```
This repository contains the inference code for NSR registration pipeline. The code has been tested on Linux Ubuntu 20.04.6 LTS, using python 3.8.13, and a GPU GeForce RTX 3080 Ti.  

## Getting Started
1) Clone the repo 
  ```bash
  git clone https://github.com/riccardomarin/NICP.git NICP
  cd NICP
  ```  

2) Create the environment 
 ```
conda create -n nsr python=3.8.13
conda activate nsr
  ```

3) Run the installation script (it also contains checkpoint download)
 ```
./install.sh
  ```

4) You need to download the smplh model and place it in the ``support_data`` folder. The correct file structure is:
```
support_data
  |__ body_models
         |__smplh
             |__neutral
                 |__model.npz
```
5) Set the home directory path in ``./src/lvd_templ/paths.py``
```
home_dir                          = #e.g., '/home/ubuntu/Documents/NICP/'
```
   
You are ready to start!

## Inference 
To use NSR and fit all the scans into the ``demo`` folder, you can run the following command:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py
```

We also provide a streamlit demo to run NSR on single shapes using a GUI.
```
PYTHONPATH=. streamlit run ./src/lvd_templ/evaluation/stream_demo.py
```

## Change Settings in ``evaluation_benchmark.py``
Depending on your use case, specify different parameters for the NSR pipeline. This can be easily done my command line. For example, this command will run a unidirectional chamfer distance refinement (-1: input with oultiers; 1: Partial input):

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.cham_bidir=-1
```

For visualization and inspection purposes, by default the output is provided in a canonical frame. If you want that the output is aligned with the original input, you can specify:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.scaleback=True
```

If you don't know if the shape Y-axis is aligned, we also implemented a "rotation guessing" heuristic: it tries several different rotations for the shape and applies the one with the best NICP loss score. For example, the SCAPE shape in `demo_guess_rot` folder is not aligned, but the alignment provided by the heuristic is good enough to let NSR succeed:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.challenge='guess_rot' core.guess_rot=True
```

If you want to characterize a run (and avoid overwrite), you can specify a tag which will be added to the output filenames:

```
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.tag='greedy' core.lr_ss=0.01
```

Others parameters (like the number of NICP iterations or learning rate) can be tuned in ``conf_test/default.yaml``.

## Example Evaluation: FAUST
To obtain NSR results of Table 2, you can run:

```
# Registering all the shapes
PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.challenge='FAUST_train_reg','FAUST_train_scans' core.checkpoint='1ljjfnbx' -m

# Obtaining the p2p matching for the scans
PYTHONPATH=. python ./src/lvd_templ/evaluation/get_match.py core.regist=out_ss_cham_0 core.subdiv=2 core.challenge.name='FAUST_train_scans' core.checkpoint='1ljjfnbx'
PYTHONPATH=. python ./src/lvd_templ/evaluation/get_match.py core.regist=out_ss_cham_0 core.subdiv=0 core.challenge.name='FAUST_train_reg' core.checkpoint='1ljjfnbx'

# Evaluating the p2p matching and get avarage error
PYTHONPATH=. python ./src/lvd_templ/evaluation/get_error.py core.evaluate="_1ljjfnbx_out_ss_cham_0_0_" core.challenge.name='FAUST_train_reg'
PYTHONPATH=. python ./src/lvd_templ/evaluation/get_error.py core.evaluate="_1ljjfnbx_out_ss_cham_0_2_" core.challenge.name='FAUST_scan_reg'

```
Libraries updates might have a minor impact on the numbers.

## Other checkpoints (CAPE, SCAPE, 5K AMASS)
Other checkpoints used in our experiments can be found [here](https://drive.google.com/file/d/1h5ozzdkwg0x-EgW5T-ropNzdJwjswmyy/view?usp=sharing)
<!-- 
<img src='./assets/myteaser3.png' width=800> -->

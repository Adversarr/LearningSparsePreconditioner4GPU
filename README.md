# 🚀 Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs
[![arXiv](https://img.shields.io/badge/arXiv-2510.27517-ff0000.svg)](https://arxiv.org/abs/2510.27517)

> [Zherui Yang](https://github.com/Adversarr) and [Zhehao Li](https://zhehaoli1999.github.io/) and [Kangbo Lyu](https://github.com/combolv) and Yixuan Li and [Tao Du](https://taodu-eecs.github.io/) and [Ligang Liu](https://staff.ustc.edu.cn/~lgliu/)

📢 This repository contains the official implementation of our paper accepted to NeurIPS 2025.

## 📦 Requirements

🧪 Tested on: Python 3.10+, PyTorch 2.6.0 + CUDA **12.6**, Linux (x86_64), NVIDIA GPUs

```sh
source uv-setup.sh
```

If `pymathprim`([link](https://github.com/Adversarr/mathprim)) is not installed, refer to `mathprim`'s document.

```
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install -e .
```


To install `pymathprim` for serious timing:
```
uv pip install $PATH_TO_MATHPRIM --no-deps
```

To install `pyssim`([link](https://github.com/Adversarr/ssim)) for elasticity datagen.
```
uv pip install $PATH_TO_SSIM --no-deps
```

## 📝 Citation

🙏 If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{Yang2025gnnspai,
  title     = {Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs},
  author    = {ggZherui Yang and Zhehao Li and Kangbo Lyu and Yixuan Li and Tao Du and Ligang Liu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```

## ▶️ Usage

For heat problem on one single mesh (diffusivity varies):
```sh
# Prepare your data: datagen/config/heat.yaml
#   on a small mesh, default to bunny_low_res.
python datagen/heat.py
# #   on a larger mesh, and tetra it to get volume
# python datagen/heat.py mesh_file=data/objs/armadillo_mid_res.obj tetrahedralize.enable=true

# Training
python train.py exp_name=heat
```

If the input is tetras files (`.msh`):
```sh
# Download and unzip to data/raw/10k_tetmesh
python preprocess/msh_to_npy.py # preprocess each msh to numpy.
# original density: 1e-4~5e-4 uniform distribution.
python datagen/heat_tetmesh.py  # generate the matrix dataset.
# Out-of-distribution data:
# 3 sigma
python datagen/heat_tetmesh.py +use_all_mesh=false min_density=7e-4 random_field=false basic.prefix=generated/heat_tetmesh-7e-4
# 5 sigma
python datagen/heat_tetmesh.py +use_all_mesh=false min_density=1e-3 random_field=false basic.prefix=generated/heat_tetmesh-1e-3

# Training
# python train.py exp_name=heat_tetmesh data.is_fixed_topology=false data.load_into_memory=false data.has_shared_features=false
sh training/heat_tetmesh.sh
# Infer on validation. same distribution
python infer.py exp_name=heat_tetmesh data.is_fixed_topology=false data.load_into_memory=false data.has_shared_features=false
# Infer on out of distribution data.
python infer.py exp_name=heat_tetmesh-7e-4 data.is_fixed_topology=false data.load_into_memory=false data.has_shared_features=false
python infer.py exp_name=heat_tetmesh-1e-3 data.is_fixed_topology=false data.load_into_memory=false data.has_shared_features=false
```

For elasticity problem:
```sh
# Prepare your data: datagen/config/elast_twist.yaml
python datagen/elast_twist.py basic.max_count=500 visualize=false
python datagen/elast_bend.py  basic.max_count=500 visualize=false
# Training
# python train.py exp_name=elast_twist data.block_size=3
sh training/elast_twist.sh
# Infer
python infer.py pretrained=PATH_TO_CKPT data.block_size=3
```

For hyper elasticity problem with remeshing:

```sh
# Datagen.
sh data/objs/gen-remesh.sh
# Training. (See ./config/basic_multidata.yaml for its dataset for training.)
# python train.py --config-name=basic_multidata data.block_size=3 exp_name=elast_twist_remesh-unstructured
sh training/multidata.sh
# validation.
sh misc/infer_all_precision.sh twist-tiny-box-remesh-6e-5 PATH_TO_CKPT data.block_size=3 data.load_into_memory=false
```

For Poisson 2D:
```sh
# Prepare your data
python datagen/poisson.py
# Training
# python train.py exp_name=poisson data.use_node_features=false data.is_fixed_topology=false
sh training/poisson.sh
```

For Poisson 3D:
```sh
python datagen/poisson3d_tetmesh.py
# Training
# python train.py exp_name=poisson_tetmesh data.has_shared_features=false data.is_fixed_topology=false
sh training/poisson_tetmesh.sh
# Infer
python infer.py exp_name=poisson_tetmesh data.has_shared_features=false data.is_fixed_topology=false data.load_into_memory=false
```

For synthetic problem, we generate a matrix with sparsity $\approx 0.15\%$.

```sh
# prepare your data
python datagen/synthetic.py
# Training
sh training/synthetic.sh
# Infer
python infer.py \
    exp_name=synthetic \
    data.is_fixed_topology=false data.has_shared_features=false data.use_node_features=false data.use_edge_features_as_node_feature=mean
```

## 📊 All experiments

We consider PDEs on meshes with different topologies and dimensions. The following table summarizes the experiments we conducted.

| Experiment Name | PDE kind | Fixed Topology ? | Fixed Boundaries? | Fixed Coefficients? | Mesh Kind |
|-----------------|----------|------------------|-------------------|---------------------|-----------|
| Heat            | Heat     | No (TetWild)     | Yes               | Yes                 | FEM 3D    |
| Poisson 3D      | Poisson  | Yes              | No                | Yes                 | FEM 3D    |
| Beam Twisting   | Hyper El | No               | Yes               | No                  | FEM 3D    |
| Synthetic       | Algebra  | No               | No  (impossible)  | No (scipy random)   | Algebra   |


## 📄 License

📜 This project is released under the license specified in the LICENSE file of this repository.

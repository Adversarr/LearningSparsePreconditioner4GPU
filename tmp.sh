# sh misc/infer_all_precision.sh heat_tetmesh output/heat_tetmesh_csim/last.ckpt data.load_into_memory=false +dataloader=test data.is_fixed_topology=false data.has_shared_features=false +infer_prefix=CosSim
# sh misc/infer_all_precision.sh heat_tetmesh output/heat_tetmesh_l2_loss/last.ckpt data.load_into_memory=false +dataloader=test data.is_fixed_topology=false data.has_shared_features=false +infer_prefix=L2_ data.normalize_matrix=false
# python cond.py exp_name=heat-cond pretrained="output/cond/last2.ckpt"

# sh misc/infer_all_precision.sh heat_tetmesh output/heat_tetmesh_rl2_loss_unseen_meshes/heat_tetmesh-simple-latest.ckpt data.load_into_memory=false +dataloader=test data.is_fixed_topology=false data.has_shared_features=false +infer_prefix=amgxcg_
sh misc/infer_all_precision_multidata.sh twist_multidata output/elast_csim/last.ckpt data.load_into_memory=false +infer_prefix=amgxcg_
sh misc/infer_all_precision.sh synthetic output/synthetic_l2/last.ckpt data.has_shared_features=false data.is_fixed_topology=false data.use_node_features=false data.use_edge_features_as_node_feature=mean data.load_into_memory=false +infer_prefix=amgxcg_
import os
import pickle
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

heat_train_config = {
    # 'num_diffusivities': 5,
    # 'diffusivities': [0.2, 0.4, 0.6, 0.7],
    # 'diffusivities': [0.5, 1.5, 5.0, 10.0],
    # 'diffusivities': [1.5, 5.0, 10.0],
    # 'diffusivities': [1.5, 5.0, 20.0, 50.0, 100.0],
    "diffusivities": [100.0],
    # 'diffusivities': [1.5],
    # 'diffusivity_cap':1.5,
    "name": "eight_mid_res",
    "time_step": 1e-2,
    "num_time_steps": 100,
    "total_data_num": 4800,
    "data_features": None,
    "num_inits": 1,
    "split": "train",
}

heat_test_config = {
    "diffusivities": [100.0],
    "name": "eight_mid_res",
    "time_step": 1e-2,
    "num_time_steps": 100,
    "total_data_num": 100,
    "data_features": None,
    "num_inits": 1,
    "split": "test",
}

# Define the data type we plan to use.
np_float_dtype = np.float64
torch_float_dtype = torch.float64
np_int_dtype = np.int64
torch_int_dtype = torch.int64
torch_device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)


def to_np_float(x):
    return np.asarray(x, dtype=np_float_dtype).copy()


def to_np_int(x):
    return np.asarray(x, dtype=np_int_dtype).copy()


def to_torch_float(x, requires_grad):
    return torch.tensor(
        to_np_float(x), dtype=torch_float_dtype, requires_grad=requires_grad
    )


def to_torch_bool(x):
    return torch.tensor(x, dtype=torch.bool, requires_grad=False)


def to_torch_int(x, requires_grad):
    return torch.tensor(
        to_np_int(x), dtype=torch_int_dtype, requires_grad=requires_grad
    )


def torch_to_np_float(x):
    return to_np_float(x.clone().cpu().detach().numpy())


def torch_to_np_int(x):
    return to_np_int(x.clone().cpu().detach().numpy())


from itertools import combinations

from torch_geometric.utils import sort_edge_index


def twohop(edge_index):
    # edge_index = edge_index
    sorted_edge_index = sort_edge_index(edge_index)
    sorted_edge_index_list = list(sorted_edge_index.T)
    all_edges = [(x.item(), y.item()) for x, y in sorted_edge_index_list]
    dct = dict((x.item(), []) for x, y in sorted_edge_index_list)

    for x, y in all_edges:
        dct[x].append(y)

    two_hop_edges = []

    for x in dct:
        y_list = dct[x]
        all_hops = list(combinations(y_list, 2))
        for hop in all_hops:
            if hop[0] < x and hop[1] < x and hop not in all_edges:
                two_hop_edges.append(hop)

    two_hop_edges = torch.tensor(two_hop_edges, device=edge_index.device).long()
    return two_hop_edges.T


class FEMDataset(Dataset):
    """Base Dataset that deals only with mesh related"""

    def __init__(self, domain_files_path=None, name=None, data_features=None):
        self.domain_files = []
        if domain_files_path is None and name is None:
            self.domain_files.append(
                str(
                    Path(root_path)
                    / "asset"
                    / "mesh"
                    / "2d"
                    / "{}.obj".format("circle_low_res")
                )
            )
        elif domain_files_path is not None:
            if domain_files_path.endswith("txt"):
                with open(domain_files_path, "r") as f:
                    for line in f:
                        l = line.strip()
                        self.domain_files.append(
                            str(
                                Path(root_path)
                                / "asset"
                                / "mesh"
                                / "2d"
                                / "{}.obj".format(l)
                            )
                        )
            else:
                raise Exception(f"data set type {domain_files_path} is not defined")
        elif name is not None:
            self.domain_files.append(
                str(Path(root_path) / "asset" / "mesh" / "2d" / "{}.obj".format(name))
            )

        self.data_features = [
            "node_pos",
            "edge_len",
            "node_interior_mask",
            "edge_index",
        ]

        if data_features is not None:
            self.data_features = data_features

        self.node_pos = []
        self.edge_index = []
        self.interior_node_mask = []
        self.edge_len = []
        start_counter = 0
        for domain_file in self.domain_files:
            # nodes, edge_index, interior_node_mask, edge_len, boundary_nodes, faces
            nodes, edge_index, interior_node_mask, edge_len, _, _ = (
                self.load_finite_elements(domain_file, start_counter=start_counter)
            )
            self.node_pos.extend(nodes)
            self.edge_index.extend(edge_index)
            self.interior_node_mask.extend(interior_node_mask)
            self.edge_len.extend(edge_len)
            start_counter = len(self.node_pos)

        self.node_pos = np.array(self.node_pos)
        self.edge_index = np.array(self.edge_index)
        self.interior_node_mask = np.array(self.interior_node_mask)[..., np.newaxis]
        self.edge_len = np.array(self.edge_len)

    def to(self, int_dtype, float_dtype, device):
        for g in self.graphs:
            g.to(int_dtype, float_dtype, device)

    def load_finite_elements(self, obj_file_name, start_counter=0):
        # read obj file
        with open(obj_file_name, "r") as f:
            lines = f.readlines()

        v = []
        f = []
        boundary_edges = set()
        for l in lines:
            l = l.strip()
            if l.startswith("v "):
                words = l.split()
                if len(words) != 4:
                    print_error("[load_finite_elements]: invalid vertex line.")
                vx = float(words[1])
                vy = float(words[2])
                v.append([vx, vy])
            if l.startswith("f "):
                words = l.split()
                if len(words) != 4:
                    print_error("[load_finite_elements]: invalid face line.")
                nodes = [int(n.split("/")[0]) - 1 for n in words[1:]]
                f.append(nodes)
                for i in range(3):
                    ni = nodes[i]
                    nj = nodes[(i + 1) % 3]
                    pi, pj = ni, nj
                    if ni > nj:
                        pi, pj = nj, ni
                    if (pi, pj) in boundary_edges:
                        boundary_edges.discard((pi, pj))
                    else:
                        boundary_edges.add((pi, pj))
        nodes = to_np_float(v)
        faces = to_np_int(f)

        boundary_nodes = set()
        for pi, pj in boundary_edges:
            boundary_nodes.add(pi)
            boundary_nodes.add(pj)
        boundary_nodes = to_np_int(sorted(list(boundary_nodes)))

        interior_node_mask = np.array([True] * nodes.shape[0])
        interior_node_mask[boundary_nodes] = False

        edge_index = set()
        for e in faces:
            for i in range(faces.shape[1]):
                for j in range(i + 1, faces.shape[1]):
                    ei, ej = e[i], e[j]
                    edge_index.add((ei, ej))
                    edge_index.add((ej, ei))
        for i in range(nodes.shape[0]):
            edge_index.add((i, i))

        edge_index = np.array(list(edge_index))
        e0 = nodes[edge_index[:, 0]]
        e1 = nodes[edge_index[:, 1]]
        edge_len = np.sqrt(np.sum((e0 - e1) ** 2, axis=1))[..., np.newaxis]

        edge_index += [start_counter, start_counter]
        return nodes, edge_index, interior_node_mask, boundary_nodes, edge_len, faces

    def get_nodes(self, nodes_input, idx):
        edges = torch.from_numpy(self.edge_index[idx]).view(-1).long()
        out = torch.from_numpy(nodes_input)[edges]
        return out

    def get_edges(self, edge_input, idx):
        out = torch.from_numpy(edge_input)[idx]
        return out

    def save(self, save_dir, filename="data.npy"):
        data = np.array([x.__dict__ for x in self.graphs])
        np.save(save_dir + f"/{filename}", data)

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, idx):
        data_feats = ()
        for feat in self.data_features:
            if feat == "node_pos":
                out = self.get_nodes(self.node_pos, idx)
                data_feats = data_feats + (out,)

            elif feat == "edge_len":
                out = self.get_edges(self.edge_len, idx)
                data_feats = data_feats + (out,)

            elif feat == "node_interior_mask":
                out = self.get_nodes(self.interior_node_mask, idx)
                data_feats = data_feats + (out,)

            elif feat == "edge_index":
                out = self.get_edges(self.edge_index, idx)
                data_feats = data_feats + (out,)

            else:
                raise Exception(f" data feature {feat} is not defined! ")

        return data_feats


class HeatDatasetMultiSource(FEMDataset):
    def __init__(
        self,
        domain_files_path=None,
        name=None,
        config=None,
        use_data_num=None,
        use_high_freq=False,
        augment_edge=False,
        use_pred_x=False,
        high_freq_aug=False,
    ):
        diffusivities = config["diffusivities"] if config is not None else [100.0]
        total_data_num = config["total_data_num"] if config is not None else 5000
        file_data_num = (
            config["num_time_steps"] * config["num_inits"] if config is not None else 50
        )
        name = config["name"] if config is not None else "circle_low_res"
        split = config["split"] if config is not None else "train"
        epsilon = 0.5  # percentage of high frequency data for augmenting with high frequency data
        # base_dir = '/data/yichenl/diffusivity_{}/' + name + '/'

        if split == "train":
            base_dir = "./data/diffusivity_{}/" + name + "/"
        else:
            base_dir = "./data/diffusivity_{}/" + name + "_test/"

        self.graphs = []
        domain_files_paths = []
        num_data_to_load_per_domain = (
            total_data_num // len(diffusivities) // file_data_num
        )
        # to form training data from the training and testing config
        for diffusivity in diffusivities:
            cur_diffusivity_dir = base_dir.format(diffusivity)
            cur_diffu_domain_files = os.listdir(cur_diffusivity_dir)
            selected_domain_files = np.random.choice(
                cur_diffu_domain_files, num_data_to_load_per_domain
            )

            from tqdm import tqdm

            for cur_domain_file in tqdm(selected_domain_files, desc=f'Load {split} set'):
                data = np.load(cur_diffusivity_dir + cur_domain_file, allow_pickle=True)
                ind = 0
                data_shape = data[0]["x"].shape[0]

                for d in data:
                    if use_pred_x:
                        x = torch.cat(
                            [
                                d["x"][:, :2],
                                d["u"].reshape(-1, 1).float(),
                                d["x"][:, -1].reshape(-1, 1),
                            ],
                            dim=-1,
                        )
                    else:
                        x = d["x"]
                    edge_attr = d["edge_attr"]
                    edge_index = d["edge_index"]

                    graph_data = Data(
                        x=x,  # d['x'], \
                        edge_attr=edge_attr,
                        edge_index=edge_index,
                        y=d["rhs"],
                        u=d["u"],
                        diag=d["diag"].reshape(-1, 1),
                        # diag = d['A_diag'],
                        r=d["r"],
                        rhs=d["rhs"],
                        u_next=d["u_next"],
                    )
                    # if augment_edge: graph_data = self.transform(graph_data)
                    self.graphs.append(graph_data)

                    ind += 1

        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[
            -1
        ]  # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1
        self.dirichlet_idx = 3
        self.b_dim = self.graphs[0].x.shape[0]
        if split == "test":
            self.graphs = self.graphs[:use_data_num]
        else:
            self.graphs = self.graphs[:use_data_num]
            # self.graphs = self.graphs[:total_data_num]

        # try on only one data
        # self.graphs = [self.graphs[0]]

    def get_data(self):
        return self.graphs

    def save(self, file_name="./data2d.npy"):
        data = np.array([x.to_dict() for x in self.graphs])
        # data = np.array([x.__dict__ for x in self.graphs])
        np.save(file_name, data)

        with open(f"{file_name}.pickle", "wb") as handle:
            pickle.dump(self.meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def to(self, int_dtype, float_dtype, device):
        for g in self.graphs:
            g.to(int_dtype, float_dtype, device)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
        # return self.graphs[idx] if self.transform is None else self.transform(self.graphs[idx])

def train_dataset():
    return HeatDatasetMultiSource(name="eight_mid_res", config=heat_train_config)
def test_dataset(): # This design is stupid, why limit use_data_num to 1???
    return HeatDatasetMultiSource(name="eight_mid_res_test", config=heat_test_config, use_data_num=1)

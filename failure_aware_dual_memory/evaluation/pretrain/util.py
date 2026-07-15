import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from failure_aware_dual_memory.util.legacy_imports import install_agent4crys_aliases
from failure_aware_dual_memory.evaluation.models.util.model import retrieve_model
from failure_aware_dual_memory.evaluation.models.util.data import get_pyg_dataset


def load_model(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    file_path = model_path / "best_model.pth"
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    install_agent4crys_aliases()
    checkpoint = torch.load(
        file_path,
        map_location=map_location,
        weights_only=False,
    )
    cfg = checkpoint["cfg"]
    model = retrieve_model(cfg)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    std_train = checkpoint["std_train"]
    mean_train = checkpoint["mean_train"]
    return model, cfg, std_train, mean_train


def load_evaluation_model(eval_prop="formation_energy_per_atom"):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if eval_prop == "formation_energy_per_atom":
        model_path = current_dir / "comformer"
        model, cfg, std_train, mean_train = load_model(model_path)
        return model, cfg, std_train, mean_train
    else:
        raise ValueError(f"Model type {eval_prop} is not supported.")


def get_dataloader(mat_df, cfg, std_train, mean_train):
    dataset, _, _ = get_pyg_dataset(
        mat_df,
        target=cfg.data.target,
        neighbor_strategy=cfg.data.neighbor_strategy,
        atom_features=cfg.data.atom_features,
        use_canonize=cfg.data.use_canonize,
        line_graph=True,
        cutoff=cfg.data.cutoff,
        max_neighbors=cfg.data.max_neighbors,
        use_lattice=cfg.data.use_lattice,
        use_angle=False,
        mean_train=mean_train,
        std_train=std_train,
        eval=True,
    )
    collate_fn = dataset.collate_line_graph
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn
    )
    return loader, loader.dataset.prepare_batch

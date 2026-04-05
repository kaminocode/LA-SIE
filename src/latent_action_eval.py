import json
from argparse import Namespace
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import src.dataset as ds
import src.experience_registry as exp_registry
import src.models as m
from src.latent_action import apply_operator


_DEFAULT_ARG_VALUES = {
    "latent_action_dim": 8,
    "num_generators": 8,
    "latent_align_weight": None,
    "latent_identity_weight": 1.0,
    "latent_inverse_weight": 1.0,
    "latent_composition_weight": 1.0,
    "latent_enable_identity": False,
    "latent_enable_inverse": False,
    "latent_enable_composition": False,
    "latent_online_eval": False,
    "latent_online_eval_samples": 16,
    "size_dataset": -1,
    "predictor_type": "hypernetwork",
    "predictor": "",
    "tf_num_layers": 1,
    "simclr_temp": 0.1,
    "bias_pred": False,
    "bias_hypernet": False,
    "predictor_relu": False,
    "hypernetwork": "linear",
    "port": 52472,
    "resolution": 256,
    "wandb": False,
    "wandb_project": "",
    "wandb_entity": "",
    "wandb_name": "",
    "wandb_dir": None,
    "no_amp": True,
}


def _with_default(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _as_path(value):
    if value is None or value == "":
        return None
    return Path(value)


def load_experiment_args(exp_dir: Path) -> Namespace:
    exp_dir = Path(exp_dir)
    with open(exp_dir / "params.json", "r") as f:
        raw_args = json.load(f)
    args = Namespace(**raw_args)
    for key, value in _DEFAULT_ARG_VALUES.items():
        _with_default(args, key, value)
    args.exp_dir = exp_dir
    args.dataset_root = _as_path(getattr(args, "dataset_root", None))
    args.images_file = _as_path(getattr(args, "images_file", None))
    args.labels_file = _as_path(getattr(args, "labels_file", None))
    args.root_log_dir = _as_path(getattr(args, "root_log_dir", None))
    args.wandb_dir = _as_path(getattr(args, "wandb_dir", None))
    return args


def checkpoint_family(args: Namespace) -> str:
    if exp_registry.is_latent_action_experience(args.experience):
        return "latent_action"
    return "original_sie"


def is_latent_action_checkpoint(args: Namespace) -> bool:
    return checkpoint_family(args) == "latent_action"


def is_triplet_checkpoint(args: Namespace) -> bool:
    return exp_registry.is_triplet_experience(args.experience)


def resolve_backbone_weights_file(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    final_weights = exp_dir / "final_weights.pth"
    if final_weights.is_file():
        return final_weights
    return exp_dir / "model.pth"


def strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def default_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_eval_transform(args: Namespace):
    normalize = transforms.Normalize(
        mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    return transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_training_dataset(
    args: Namespace,
    *,
    dataset_root: Optional[Path] = None,
    images_file: Optional[Path] = None,
    labels_file: Optional[Path] = None,
    size_dataset: Optional[int] = None,
    transform=None,
):
    dataset_root = Path(dataset_root or args.dataset_root)
    images_file = Path(images_file or args.images_file)
    labels_file = Path(labels_file or args.labels_file)
    size_dataset = args.size_dataset if size_dataset is None else size_dataset
    transform = build_eval_transform(args) if transform is None else transform

    if exp_registry.is_triplet_experience(args.experience):
        return ds.Dataset3DIEBenchTriplet(
            dataset_root,
            images_file,
            labels_file,
            size_dataset=size_dataset,
            transform=transform,
        )
    if exp_registry.uses_rotcolor_dataset(args.experience):
        return ds.Dataset3DIEBenchRotColor(
            dataset_root,
            images_file,
            labels_file,
            size_dataset=size_dataset,
            transform=transform,
        )
    return ds.Dataset3DIEBench(
        dataset_root,
        images_file,
        labels_file,
        size_dataset=size_dataset,
        transform=transform,
    )


def load_checkpoint_model(
    exp_dir: Path,
    *,
    device: Optional[str] = None,
    require_latent_action: bool = False,
    strict: bool = False,
):
    args = load_experiment_args(exp_dir)
    if require_latent_action and not is_latent_action_checkpoint(args):
        raise ValueError(
            f"{args.experience} is not a latent-action checkpoint."
        )

    device = default_device(device)
    model = m.__dict__[args.experience](args).to(device)
    ckpt = torch.load(Path(exp_dir) / "model.pth", map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    msg = model.load_state_dict(strip_module_prefix(state_dict), strict=strict)
    model.eval()
    return args, model, msg


def split_embeddings(model, embeddings: Iterable[torch.Tensor]):
    embeddings = list(embeddings)
    equi_repr_size = getattr(model, "equi_repr_size", getattr(model.args, "equi"))
    inv_repr_size = embeddings[0].shape[-1] - equi_repr_size
    projector_inv = getattr(model, "projector_inv", None)
    projector_equi = getattr(model, "projector_equi", None)

    inv_views = []
    equi_views = []
    inv_proj_views = []
    equi_proj_views = []
    full_proj_views = []

    for embedding in embeddings:
        if inv_repr_size > 0:
            inv_view = embedding[..., :inv_repr_size]
            inv_proj = projector_inv(inv_view) if projector_inv is not None else inv_view
        else:
            inv_view = embedding.new_zeros(embedding.shape[0], 0)
            inv_proj = inv_view
        equi_view = embedding[..., inv_repr_size:]
        equi_proj = projector_equi(equi_view) if projector_equi is not None else equi_view
        full_proj = torch.cat((inv_proj, equi_proj), dim=1)

        inv_views.append(inv_view)
        equi_views.append(equi_view)
        inv_proj_views.append(inv_proj)
        equi_proj_views.append(equi_proj)
        full_proj_views.append(full_proj)

    return inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views


def _sample_subdir(sample) -> Path:
    sample_str = str(sample)
    if sample_str.startswith("/"):
        sample_str = sample_str[1:]
    return Path(sample_str)


def _load_rgb_image(path: Path, transform=None):
    with open(path, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
        if transform is not None:
            image = transform(image)
    return image


def make_pair_schedule(pairs_per_object: int, *, num_views: int = 50, seed: int = 0):
    if pairs_per_object <= 0:
        raise ValueError("pairs_per_object must be positive.")
    schedule = []
    seen = set()
    rng = np.random.RandomState(seed)
    max_pairs = num_views * (num_views - 1)
    target_count = min(pairs_per_object, max_pairs)
    while len(schedule) < target_count:
        start, end = rng.choice(num_views, size=2, replace=False)
        pair = (int(start), int(end))
        if pair in seen:
            continue
        seen.add(pair)
        schedule.append(pair)
    return schedule


class RelativePosePairDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        img_file,
        labels_file,
        *,
        experience="quat",
        size_dataset=-1,
        transform=None,
        pairs_per_object=16,
        pair_seed=0,
    ):
        self.dataset_root = Path(dataset_root)
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.experience = experience
        self.pair_schedule = make_pair_schedule(
            pairs_per_object,
            seed=pair_seed,
        )

    def __len__(self):
        return len(self.samples) * len(self.pair_schedule)

    def __getitem__(self, index):
        object_index = index // len(self.pair_schedule)
        pair_index = index % len(self.pair_schedule)
        view_start, view_end = self.pair_schedule[pair_index]
        sample_root = self.dataset_root / _sample_subdir(self.samples[object_index])

        x = _load_rgb_image(sample_root / f"image_{view_start}.jpg", self.transform)
        y = _load_rgb_image(sample_root / f"image_{view_end}.jpg", self.transform)

        latent_start = np.load(sample_root / f"latent_{view_start}.npy").astype(np.float32)
        latent_end = np.load(sample_root / f"latent_{view_end}.npy").astype(np.float32)
        z = ds._relative_rotation(latent_start, latent_end, experience=self.experience)
        label = int(self.labels[object_index])

        return x, y, torch.FloatTensor(z), label


def flatten_prediction_component(prediction, kind: str):
    tensor = None
    if kind == "code":
        tensor = prediction.code
    elif kind == "coefficients":
        tensor = prediction.coefficients
    elif kind == "raw_matrix":
        tensor = prediction.raw_matrix
    elif kind == "operator":
        tensor = prediction.operator
    else:
        raise ValueError(f"Unknown latent prediction component: {kind}")

    if tensor is None:
        raise ValueError(f"Prediction component {kind} is not available.")
    if tensor.ndim > 2:
        return tensor.reshape(tensor.shape[0], -1)
    return tensor


def auto_prediction_kinds(prediction) -> List[str]:
    kinds: List[str] = []
    if prediction.coefficients is not None:
        kinds.append("coefficients")
    if prediction.code is not None:
        kinds.append("code")
    if kinds:
        return kinds
    if prediction.raw_matrix is not None:
        return ["raw_matrix"]
    return ["operator"]


class LatentActionFeatureAdapter:
    def __init__(self, exp_dir: Path, *, device: Optional[str] = None):
        args, model, _ = load_checkpoint_model(
            exp_dir,
            device=device,
            require_latent_action=True,
        )
        self.args = args
        self.model = model
        self.device = default_device(device)

    @torch.no_grad()
    def pair_features(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x_emb = self.model.backbone(x)
        y_emb = self.model.backbone(y)
        inv_views, equi_views, inv_proj_views, equi_proj_views, full_proj_views = split_embeddings(
            self.model, [x_emb, y_emb]
        )
        prediction = self.model.operator(equi_views[0], equi_views[1])
        transported = apply_operator(prediction.operator, equi_views[0])
        return {
            "reprs": [x_emb, y_emb],
            "inv_views": inv_views,
            "equi_views": equi_views,
            "inv_proj_views": inv_proj_views,
            "equi_proj_views": equi_proj_views,
            "full_proj_views": full_proj_views,
            "prediction": prediction,
            "transported_equi": transported,
        }

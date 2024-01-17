from typing import Any
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import pandas as pd
import typing as tp
from torch.utils.data import Dataset
import numpy as np
import json
from dataclasses import dataclass
from PIL import Image
from abc import ABC, abstractmethod
from tqdm import tqdm
from skimage.transform import resize
from diskcache import Cache
from einops import rearrange
from matplotlib import pyplot as plt


DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("Environment variable DATA_ROOT must be set")

DATASET_KEY = "2023-12-14_bmode_dataset_1024px"

PATIENT = pd.read_csv(os.path.join(DATA_ROOT, DATASET_KEY, "patient.csv"))
CORE = pd.read_csv(os.path.join(DATA_ROOT, DATASET_KEY, "core.csv"))


def get_patient_splits(fold=0, n_folds=5):
    """returns the list of patient ids for the train, val, and test splits."""
    if n_folds == 5:
        # we manually override the this function and use the csv file
        # because the original code uses a random seed to split the data
        # and we want to be able to reproduce the splits
        table = pd.read_csv(os.path.join(DATA_ROOT, DATASET_KEY, "5fold_splits.csv"))
        train_ids = table[table[f"fold_{fold}"] == "train"].id.values.tolist()
        val_ids = table[table[f"fold_{fold}"] == "val"].id.values.tolist()
        test_ids = table[table[f"fold_{fold}"] == "test"].id.values.tolist()
        return train_ids, val_ids, test_ids

    if fold not in range(n_folds):
        raise ValueError(f"Fold must be in range {n_folds}, but got {fold}")

    table = PATIENT

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(
        kfold.split(table.index, table["center"])
    ):
        if i == fold:
            train, test = table.iloc[train_idx], table.iloc[test_idx]
            break

    train, val = train_test_split(
        train, test_size=0.2, random_state=0, stratify=train["center"]
    )

    train = train.id.values.tolist()
    val = val.id.values.tolist()
    test = test.id.values.tolist()

    return train, val, test


def get_core_ids(patient_ids):
    """returns the list of core ids for the given patient ids."""
    return CORE[CORE.patient_id.isin(patient_ids)].id.values.tolist()


def remove_benign_cores_from_positive_patients(core_ids):
    """Returns the list of cores in the given list that are either malignant or from patients with no malignant cores."""
    table = CORE.copy()
    table["positive"] = table.grade.apply(lambda g: 0 if g == "Benign" else 1)
    num_positive_for_patient = table.groupby("patient_id").positive.sum()
    num_positive_for_patient.name = "patients_positive"
    table = table.join(num_positive_for_patient, on="patient_id")
    ALLOWED = table.query("positive == 1 or patients_positive == 0").id.to_list()

    return [core for core in core_ids if core in ALLOWED]


def remove_cores_below_threshold_involvement(core_ids, threshold_pct):
    """Returns the list of cores with at least the given percentage of cancer cells."""
    table = CORE
    ALLOWED = table.query(
        "grade == 'Benign' or pct_cancer >= @threshold_pct"
    ).id.to_list()
    return [core for core in core_ids if core in ALLOWED]


def undersample_benign(cores, seed=0, benign_to_cancer_ratio=1):
    """Returns the list of cores with the same cancer cores and the benign cores undersampled to the given ratio."""

    table = CORE
    benign = table.query('grade == "Benign"').id.to_list()
    cancer = table.query('grade != "Benign"').id.to_list()
    import random

    cores_benign = [core for core in cores if core in benign]
    cores_cancer = [core for core in cores if core in cancer]
    rng = random.Random(seed)
    cores_benign = rng.sample(
        cores_benign, int(len(cores_cancer) * benign_to_cancer_ratio)
    )

    return [core for core in cores if core in cores_benign or core in cores_cancer]


@dataclass
class CohortSelectionOptions:
    fold: int = 0
    n_folds: int = 5
    min_involvement: float = None
    remove_benign_from_positive_patients: bool = False
    benign_to_cancer_ratio: float = None
    seed: int = 0


def select_cohort(
    split: str = "train",
    cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
) -> tp.List[int]:
    """Returns the list of core ids for the given split and cohort selection options."""
    if split not in ["train", "val", "test", "all"]:
        raise ValueError(
            f"Split must be 'train', 'val', 'test' or 'all, but got {split}"
        )

    train, val, test = get_patient_splits(
        cohort_selection_options.fold, cohort_selection_options.n_folds
    )
    match split:
        case "train":
            patient_ids = train
        case "val":
            patient_ids = val
        case "test":
            patient_ids = test
        case "all":
            patient_ids = train + val + test
        case _:
            raise ValueError(
                f"Split must be 'train', 'val', 'test' or 'all, but got {split}"
            )

    core_ids = get_core_ids(patient_ids)
    if cohort_selection_options.remove_benign_from_positive_patients:
        core_ids = remove_benign_cores_from_positive_patients(core_ids)
    if cohort_selection_options.min_involvement is not None:
        core_ids = remove_cores_below_threshold_involvement(
            core_ids, cohort_selection_options.min_involvement
        )
    if cohort_selection_options.benign_to_cancer_ratio is not None:
        core_ids = undersample_benign(
            core_ids,
            cohort_selection_options.seed,
            cohort_selection_options.benign_to_cancer_ratio,
        )

    return core_ids


class CoreInfo(Dataset):
    def __init__(
        self,
        split="train",
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__()
        core_ids = select_cohort(split, cohort_selection_options)
        self.core_info = (
            CORE.set_index("patient_id")
            .join(PATIENT.rename(columns={"id": "patient_id"}).set_index("patient_id"))
            .reset_index()
        )
        self.core_info = self.core_info[self.core_info.id.isin(core_ids)]

    def __getitem__(self, index):
        out = {}
        core_info = dict(self.core_info.iloc[index])
        out.update(core_info)
        return out

    def __len__(self):
        return len(self.core_info)

    def tag_for_core_id(self, core_id):
        return self.core_info[self.core_info.id == core_id].tag.values[0]


def load_prostate_mask(tag):
    from PIL import Image

    mask = Image.open(
        os.path.join(DATA_ROOT, DATASET_KEY, "prostate_masks", f"{tag}.png")
    )
    mask = np.array(mask)
    mask = np.flip(mask, axis=0).copy()
    mask = mask / 255
    return mask


_needle_mask = None


def load_needle_mask():
    global _needle_mask
    if _needle_mask is None:
        _needle_mask = np.load(os.path.join(DATA_ROOT, DATASET_KEY, "needle_mask.npy"))
    return _needle_mask


class ExactNCT2013BModeImages(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
    ):
        super().__init__()
        self.core_info = CoreInfo(split, cohort_selection_options)
        self.transform = transform

        self._bmode_data = np.load(
            os.path.join(DATA_ROOT, DATASET_KEY, "bmode_data.npy"), mmap_mode="r"
        )
        import json

        self._core_tag_to_index = json.load(
            open(os.path.join(DATA_ROOT, DATASET_KEY, "core_id_to_idx.json"))
        )
        self.needle_mask = load_needle_mask()

    def __getitem__(self, idx):
        info = self.core_info[idx]
        tag = info["tag"]

        bmode = self._bmode_data[self._core_tag_to_index[tag]]
        needle_mask = self.needle_mask
        prostate_mask = load_prostate_mask(tag)

        item = {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            **info,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.core_info)


def n_windows(image_size, patch_size, stride):
    return 1 + (image_size - patch_size) // stride


class ExactNCT2013BModePatches(Dataset):
    def __init__(
        self,
        split="train",
        cohort_selection_options: CohortSelectionOptions = CohortSelectionOptions(),
        patch_size=128,
        stride=128,
        needle_mask_threshold=-1,
        prostate_mask_threshold=-1,
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.bmode_images = ExactNCT2013BModeImages(split, cohort_selection_options=cohort_selection_options)
        self.core_info = self.bmode_images.core_info

        # computing patch positions
        position_candidates = np.mgrid[
            0 : n_windows(1024, patch_size, stride) * stride : stride,
            0 : n_windows(1024, patch_size, stride) * stride : stride,
        ]
        position_candidates = rearrange(position_candidates, "c h w -> (h w) c")
        position_candidates = np.concatenate(
            [position_candidates, position_candidates + patch_size], axis=-1
        )

        # since it is the same for every image, we can apply the needle mask threshold here
        needle_mask = load_needle_mask()
        needle_mask = resize(needle_mask, (1024, 1024), order=0)

        new_position_candidates = []
        for position_candidate in tqdm(
            position_candidates, desc="Applying needle mask"
        ):
            x1, y1, x2, y2 = position_candidate
            patch = needle_mask[x1:x2, y1:y2]
            if patch.mean() > needle_mask_threshold:
                new_position_candidates.append(position_candidate)
        position_candidates = np.array(new_position_candidates)

        # loading all prostate masks
        prostate_masks = []
        for idx in tqdm(range(len(self.core_info)), desc="Loading Prostate Masks"):
            tag = self.core_info[idx]["tag"]
            prostate_mask = load_prostate_mask(tag)
            prostate_masks.append(prostate_mask)
        prostate_masks = np.stack(prostate_masks, axis=-1)

        n_images = len(self.core_info)
        n_position_candidates = len(position_candidates)
        valid_position_candidates = np.zeros(
            (n_images, n_position_candidates), dtype=bool
        )

        for idx in tqdm(range(n_position_candidates), desc="Applying prostate mask"):
            x1, y1, x2, y2 = position_candidates[idx]
            x1 = int(x1 / 1024 * prostate_masks.shape[0])
            x2 = int(x2 / 1024 * prostate_masks.shape[0])
            y1 = int(y1 / 1024 * prostate_masks.shape[1])
            y2 = int(y2 / 1024 * prostate_masks.shape[1])

            valid_position_candidates[:, idx] = (
                prostate_masks[x1:x2, y1:y2].mean(axis=(0, 1)) > prostate_mask_threshold
            )

        self._indices = []
        self._positions = []
        for idx in tqdm(range(n_images), desc="Filtering positions"):
            positions_for_core = []
            for j in range(n_position_candidates):
                if valid_position_candidates[idx, j]:
                    position = position_candidates[j]
                    positions_for_core.append(position)
                    self._indices.append((idx, len(positions_for_core) - 1))
            self._positions.append(positions_for_core)

    def __getitem__(self, idx):
        core_idx, patch_idx = self._indices[idx]
        info = self.core_info[core_idx]

        bmode = self.bmode_images[core_idx]["bmode"]

        positions = self._positions[core_idx][patch_idx]
        x1, y1, x2, y2 = positions
        patch = bmode[x1:x2, y1:y2]

        item = {
            "patch": patch,
            "position": positions,
            **info,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self._indices)

    @property
    def num_cores(self):
        return len(self.core_info)

    def list_patches_for_core(self, core_idx):
        indices = [i for i, (idx, _) in enumerate(self._indices) if idx == core_idx]
        return [self._indices[i] for i in indices]

    def show_patch_extraction(self, ax=None):
        """Illustrates the patch extraction process for a random core

        Shows a random image from the dataset, with the extracted patches
        highlighted with red transparent rectangles.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        """
        ax = ax or plt.gca()

        import matplotlib.patches as patches
        import random

        core_idx = random.randint(0, len(self.core_info) - 1)
        info = self.core_info[core_idx]
        tag = info["tag"]
        bmode = self.bmode_images[core_idx]["bmode"]
        positions = self._positions[core_idx]
        ax.imshow(bmode, cmap="gray")
        for position in positions:
            x1, y1, x2, y2 = position
            rect = patches.Rectangle(
                (y1, x1),
                y2 - y1,
                x2 - x1,
                linewidth=1,
                edgecolor="black",
                facecolor="red",
                alpha=0.1,
            )
            ax.add_patch(rect)
        ax.axis("off")



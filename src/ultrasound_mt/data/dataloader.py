import os
import cv2
import nrrd
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader


def read_png(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_mask_nearest(mask, target_hw):
    target_h, target_w = target_hw
    mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask


def normalize_image(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def read_slicer_nrrd_mask(mask_path, image_shape=None):
    mask, header = nrrd.read(mask_path)
    mask = np.asarray(mask)

    if mask.ndim == 2:
        pass
    elif mask.ndim == 3:
        # [1, H, W]
        if mask.shape[0] == 1:
            mask = mask[0]
        # [H, W, 1]
        elif mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        else:
            if mask.shape[0] <= 10:
                mask = mask[0]

            elif mask.shape[-1] <= 10:
                mask = mask[:, :, 0]
            else:
                squeezed = np.squeeze(mask)
                if squeezed.ndim == 2:
                    mask = squeezed
                else:
                    raise ValueError(f"Unsupported 3D mask shape: {mask.shape}")
    else:
        squeezed = np.squeeze(mask)
        if squeezed.ndim == 2:
            mask = squeezed
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Final mask is not 2D: {mask.shape}")

    mask = (mask > 0).astype(np.uint8)

    if image_shape is not None:
        if mask.shape == image_shape:
            pass
        elif mask.T.shape == image_shape:
            mask = mask.T
        else:
            mask = resize_mask_nearest(mask, image_shape)

    return mask


def random_translate_image_and_mask(
    image,
    mask,
    max_shift_x=20,
    max_shift_y=20,
    p=0.8,
    border_value_img=0,
    border_value_mask=0
):
    """
    Synchronously apply a random small translation to the image and mask.
    """
    if np.random.rand() > p:
        return image, mask

    h, w = image.shape[:2]

    dx = np.random.randint(-max_shift_x, max_shift_x + 1)
    dy = np.random.randint(-max_shift_y, max_shift_y + 1)

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    image_t = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_img
    )

    mask_t = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_mask
    )

    mask_t = (mask_t > 0).astype(mask.dtype)

    return image_t, mask_t


def random_rotate_image_and_mask(
    image,
    mask,
    max_angle=10,
    p=0.8,
    border_value_img=0,
    border_value_mask=0
):
    """
    New: synchronously apply a random slight rotation to the image and mask.
    """
    if np.random.rand() > p:
        return image, mask

    h, w = image.shape[:2]
    # Random angle: -max_angle ~ +max_angle
    angle = np.random.uniform(-max_angle, max_angle)
    # Rotation center: image center
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image (linear interpolation)
    image_rot = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_img
    )

    # Rotate the mask (nearest-neighbor interpolation to preserve labels)
    mask_rot = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_mask
    )

    mask_rot = (mask_rot > 0).astype(mask.dtype)
    return image_rot, mask_rot


class MultiTaskUltrasoundDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=(768, 1024),   # (H, W)
        require_mask=True,
        use_translation_aug=False,
        use_rotation_aug=False,  # New: rotation augmentation switch
        max_shift_x=20,
        max_shift_y=20,
        max_rotate_angle=10,     # New: maximum rotation angle
        translation_p=0.8,
        rotation_p=0.8,          # New: rotation execution probability
        aug_repeat_count=1       # New: number of augmented versions per image (number of outputs)
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.require_mask = require_mask

        # Translation augmentation
        self.use_translation_aug = use_translation_aug
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.translation_p = translation_p

        # Rotation augmentation (new)
        self.use_rotation_aug = use_rotation_aug
        self.max_rotate_angle = max_rotate_angle
        self.rotation_p = rotation_p

        # Augmentation repeat count (new)
        self.aug_repeat_count = max(1, int(aug_repeat_count))

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in: {root_dir}")

    def _collect_samples(self):
        samples = []
        for label_name in ["0", "1"]:
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = int(label_name)
            for patient_name in sorted(os.listdir(label_dir)):
                patient_dir = os.path.join(label_dir, patient_name)
                if not os.path.isdir(patient_dir):
                    continue
                files = os.listdir(patient_dir)
                png_files = sorted([f for f in files if f.lower().endswith(".png")])
                for png_name in png_files:
                    base = os.path.splitext(png_name)[0]
                    png_path = os.path.join(patient_dir, png_name)
                    nrrd_path = os.path.join(patient_dir, base + ".nrrd")
                    has_mask = os.path.isfile(nrrd_path)
                    if self.require_mask and not has_mask:
                        continue
                    samples.append({
                        "image_path": png_path,
                        "mask_path": nrrd_path if has_mask else None,
                        "has_mask": has_mask,
                        "label": label,
                        "patient_id": patient_name
                    })
        return samples

    def __len__(self):
        return len(self.samples) * self.aug_repeat_count

    def __getitem__(self, idx):
        # Compute the actual sample index
        real_idx = idx // self.aug_repeat_count
        item = self.samples[real_idx]

        image_path = item["image_path"]
        mask_path = item["mask_path"]
        has_mask = False
        label = item["label"]
        patient_id = item["patient_id"]

        # Read the image
        image = read_png(image_path)
        if mask_path and os.path.exists(mask_path):
            mask = read_slicer_nrrd_mask(mask_path, image.shape)
            has_mask = True
        else:
            mask = np.zeros_like(image, dtype=np.uint8)

        # Resize to a uniform size
        target_h, target_w = self.image_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = resize_mask_nearest(mask, (target_h, target_w))

        # Translation augmentation
        if self.use_translation_aug:
            image, mask = random_translate_image_and_mask(
                image=image, mask=mask,
                max_shift_x=self.max_shift_x, max_shift_y=self.max_shift_y,
                p=self.translation_p
            )

        # Rotation augmentation (new)
        if self.use_rotation_aug:
            image, mask = random_rotate_image_and_mask(
                image=image, mask=mask,
                max_angle=self.max_rotate_angle, p=self.rotation_p
            )

        # Normalize
        image = normalize_image(image)
        mask = mask.astype(np.float32)

        # Add channel dimension and convert to tensor
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        label = torch.tensor(label, dtype=torch.long)
        has_mask = torch.tensor(has_mask, dtype=torch.bool)

        return {
            "image": image, "mask": mask, "has_mask": has_mask,
            "label": label, "patient_id": patient_id,
            "image_path": image_path, "mask_path": mask_path if mask_path else ""
        }


def create_dataloader(
    root_dir,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    image_size=(768, 1024),
    require_mask=True,
    pin_memory=True,
    drop_last=False,
    use_translation_aug=False,
    use_rotation_aug=False,    # New
    max_shift_x=20,
    max_shift_y=20,
    max_rotate_angle=10,       # New
    translation_p=0.8,
    rotation_p=0.8,            # New
    aug_repeat_count=1         # New: generate N augmented images per image
):
    dataset = MultiTaskUltrasoundDataset(
        root_dir=root_dir,
        image_size=image_size,
        require_mask=require_mask,
        use_translation_aug=use_translation_aug,
        use_rotation_aug=use_rotation_aug,
        max_shift_x=max_shift_x,
        max_shift_y=max_shift_y,
        max_rotate_angle=max_rotate_angle,
        translation_p=translation_p,
        rotation_p=rotation_p,
        aug_repeat_count=aug_repeat_count
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataset, loader


def print_dataset_info(dataset, name="dataset"):
    labels = [s["label"] for s in dataset.samples]
    patients = [s["patient_id"] for s in dataset.samples]
    has_masks = [s["has_mask"] for s in dataset.samples]

    print(f"===== {name} =====")
    print(f"num raw samples : {len(dataset.samples)}")
    print(f"aug repeat count: {dataset.aug_repeat_count}")
    print(f"total samples   : {len(dataset)}")
    print(f"num patients    : {len(set(patients))}")
    print(f"label count     : {dict(Counter(labels))}")
    print(f"has mask        : {sum(has_masks)} / {len(has_masks)}")
    print(f"image_size      : {dataset.image_size}")
    print(f"use translation : {dataset.use_translation_aug}")
    print(f"use rotation    : {dataset.use_rotation_aug}")


def show_samples(dataset, num_samples=3, start_idx=0):
    end_idx = min(start_idx + num_samples, len(dataset))
    actual_n = end_idx - start_idx
    plt.figure(figsize=(12, 4 * actual_n))

    for row, i in enumerate(range(start_idx, end_idx)):
        sample = dataset[i]
        img = sample["image"][0].numpy()
        msk = sample["mask"][0].numpy()
        lbl = sample["label"].item()
        hm = sample["has_mask"].item()

        plt.subplot(actual_n, 3, 3 * row + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Image | label={lbl}")
        plt.axis("off")

        plt.subplot(actual_n, 3, 3 * row + 2)
        plt.imshow(msk, cmap="gray")
        plt.title(f"Mask | has_mask={hm} | sum={msk.sum():.0f}")
        plt.axis("off")

        plt.subplot(actual_n, 3, 3 * row + 3)
        plt.imshow(img, cmap="gray")
        plt.imshow(msk, cmap="Reds", alpha=0.35)
        plt.title("Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def debug_raw_nrrd(mask_path):
    mask, header = nrrd.read(mask_path)
    mask = np.asarray(mask)
    print("mask_path:", mask_path)
    print("raw shape:", mask.shape)
    print("raw dtype:", mask.dtype)
    unique_vals = np.unique(mask)
    if len(unique_vals) > 20:
        print("raw unique (first 20):", unique_vals[:20], "...")
    else:
        print("raw unique:", unique_vals)
    squeezed = np.squeeze(mask)
    print("squeezed shape:", squeezed.shape)
    if squeezed.ndim == 2:
        plt.figure(figsize=(5, 5))
        plt.imshow(squeezed, cmap="gray")
        plt.title("Raw NRRD (squeezed)")
        plt.axis("off")
        plt.show()
    else:
        print("squeezed result is not 2D, skip imshow")
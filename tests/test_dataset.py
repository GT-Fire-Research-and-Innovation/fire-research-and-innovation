"""FireDataset: one item per window, model-ready shapes, TESSERA as a static map."""

from firepred.config.channels import BASE_MODEL_CHANNELS, TESSERA_CHANNELS
from firepred.data.archive import find_fire_sources, fire_name
from firepred.data.dataset import FireDataset
from firepred.data.loader import batch_inputs, make_dataloader, validate_dataset_output
from firepred.data.tessera import compute_tessera_stats


def test_items_have_model_shapes(data_root):
    root, _ = data_root
    dataset = FireDataset(root / "train", ts_length=6, target_is_single_day=True, use_augmentations=True)
    assert len(dataset) == 3  # windows across fires, not fires
    sample = dataset[0]
    assert tuple(sample["data"].shape) == (BASE_MODEL_CHANNELS, 6, 256, 256)
    assert tuple(sample["labels"].shape) == (2, 256, 256)
    assert (sample["labels"].sum(dim=0) == 1).all()  # background and fire are complementary
    validate_dataset_output(dataset, BASE_MODEL_CHANNELS, "train")


def test_tessera_adds_a_static_map_repeated_on_the_batch(data_root):
    root, _ = data_root
    stats = compute_tessera_stats([root / "tessera" / f"{fire_name(p)}.npz" for p in find_fire_sources(root / "train")])
    dataset = FireDataset(root / "val", ts_length=6, target_is_single_day=True,
                          tessera_dir=root / "tessera", tessera_stats=stats)
    assert tuple(dataset[0]["static"].shape) == (TESSERA_CHANNELS, 256, 256)
    batch = next(iter(make_dataloader(dataset, 1, shuffle=False, num_workers=0, pin_memory=False)))
    inputs = batch_inputs(batch, "cpu")
    assert tuple(inputs.shape) == (1, BASE_MODEL_CHANNELS + TESSERA_CHANNELS, 6, 256, 256)

"""Project-wide constants agree with each other and with the data they describe."""

from firepred.config import dataset, training


def test_channel_arithmetic():
    """27 stored channels become 43 model channels, and 172 with TESSERA."""
    assert len(dataset.STORED_CHANNEL_NAMES) == dataset.N_STORED_CHANNELS == 27
    assert dataset.BASE_MODEL_CHANNELS == 27 - 1 + 17 == 43
    assert dataset.TESSERA_CHANNELS == 129
    assert dataset.model_input_channels(False) == 43
    assert dataset.model_input_channels(True) == 172


def test_special_channels_are_what_their_names_say():
    """The angle and land-cover channel indices point at the channels their names say."""
    names = dataset.STORED_CHANNEL_NAMES
    assert isinstance(dataset.DEGREE_CHANNELS, list)  # x[list] selects channels; a tuple would not
    assert [names[i] for i in dataset.DEGREE_CHANNELS] == ["wind direction", "aspect", "forecast wind direction"]
    assert names[dataset.LANDCOVER_CHANNEL] == "land cover"


def test_normalization_covers_every_stored_channel():
    """There is a positive normalization statistic for every stored channel."""
    assert len(dataset.NORMALIZATION_MEAN) == len(dataset.NORMALIZATION_STD) == 27
    assert all(std > 0 for std in dataset.NORMALIZATION_STD)


def test_paths_point_into_the_repo():
    """The repo root and the default test CSV resolve inside the checkout."""
    assert (dataset.REPO_ROOT / "firepred").is_dir()
    assert dataset.DEFAULT_TEST_CSV.is_file()


def test_window_repo_names_match_the_hugging_face_repos():
    """Window repo names follow the Hugging Face repos (...-processed-<ts>-<it>)."""
    assert dataset.window_repo_for(6, 1) == "SamuelWu318/ts-satfire-processed-6-1"
    assert dataset.window_repo_for(10, 3) == "SamuelWu318/ts-satfire-processed-10-3"


def test_training_defaults_are_the_papers():
    """The default class weights and loss names are the paper's."""
    assert training.PAPER_CE_WEIGHT == (1.0, 446.7836)
    assert training.LOSSES == ("dice", "dicece", "dicefocal")


def test_default_loss_is_still_the_papers():
    """Without --loss, models get the paper's loss (Dice for attunet, Dice+CE otherwise)."""
    from firepred.training.config import LossConfig

    assert LossConfig.paper_loss("swinunetr3d") == "dicece" and LossConfig.paper_loss("attunet") == "dice"
    assert training.LR_SCHEDULES[0] == "none"

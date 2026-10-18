"""Project-wide constants agree with each other and with the data they describe."""

from firepred.config import channels, dataset, paths, training


def test_channel_arithmetic():
    assert len(channels.STORED_CHANNEL_NAMES) == channels.N_STORED_CHANNELS == 27
    assert channels.BASE_MODEL_CHANNELS == 27 - 1 + 17 == 43
    assert channels.TESSERA_CHANNELS == 129
    assert channels.model_input_channels(False) == 43
    assert channels.model_input_channels(True) == 172


def test_special_channels_are_what_their_names_say():
    names = channels.STORED_CHANNEL_NAMES
    assert isinstance(channels.DEGREE_CHANNELS, list)  # x[list] selects channels; a tuple would not
    assert [names[i] for i in channels.DEGREE_CHANNELS] == ["wind direction", "aspect", "forecast wind direction"]
    assert names[channels.LANDCOVER_CHANNEL] == "land cover"


def test_normalization_covers_every_stored_channel():
    assert len(channels.NORMALIZATION_MEAN) == len(channels.NORMALIZATION_STD) == 27
    assert all(std > 0 for std in channels.NORMALIZATION_STD)


def test_paths_point_into_the_repo():
    assert (paths.REPO_ROOT / "firepred").is_dir()
    assert paths.DEFAULT_TEST_CSV.is_file()


def test_window_repo_names_match_the_hugging_face_repos():
    assert dataset.window_repo_for(6, 1) == "SamuelWu318/ts-satfire-processed-6-1"
    assert dataset.window_repo_for(10, 3) == "SamuelWu318/ts-satfire-processed-10-3"


def test_training_defaults_are_the_papers():
    assert training.PAPER_CE_WEIGHT == (1.0, 446.7836)
    assert training.LOSSES == ("dice", "dicece", "dicefocal")


def test_default_loss_is_still_the_papers():
    from firepred.training.run import default_loss

    assert default_loss("swinunetr3d") == "dicece" and default_loss("attunet") == "dice"
    assert training.LR_SCHEDULES[0] == "none"

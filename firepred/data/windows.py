"""GeoTIFF frames -> model-ready windows and next-day targets, for one fire.

Level 1 (data, dataset building). Used by data/windows_master.py while building the dataset.
Functions return values and print nothing; build_fire_windows returns its
progress notes for the caller to show.

For a fire with N daily VIIRS_Day GeoTIFFs (plus matching VIIRS_Night and
FirePred files), a window starting on day i stacks days i .. i+length-1 as the
input and uses day i+length as the target:

    input   (27, length, 256, 256): day bands 0-5, night bands 6-7, FirePred 8-26,
            cropped to rows/cols 128:384; NaN/Inf -> 0. Channels 3, 4 (I4, I5)
            and 6, 7 (night I4, I5) hold the running maximum over the window's
            days so far, a memory of where it has been hot.
    target  (256, 256): pixels burned on day i+length that were not burned by
            day i+length-1 ("the difference between the burned area masks of
            the last observed day and the next day" [paper]).

The burned-area mask accumulates active fire (AF band) and, with label_sel 0,
the NIFC perimeters (BA band) too; label_sel 1 uses AF only.
"""

import os
from dataclasses import dataclass

import numpy as np

from firepred.config.dataset import CROP_OFFSET, CROP_SIZE, N_STORED_CHANNELS, TS_INTERVAL, TS_LENGTH


@dataclass(frozen=True)
class FireWindows:
    """Hold one fire's input windows, their next-day targets and what happened while building them.

    A fire with N+T days of imagery gives N windows. A fire with too few days
    (or no imagery) gives zero windows, and its notes say why. The notes replace
    printing, so the caller decides what to show.

    Attributes:
        inputs (np.ndarray): (N, 27, T, 256, 256) float32 input windows.
        targets (np.ndarray): (N, 256, 256) float32 next-day targets (1 = newly burned).
        notes (tuple[str, ...]): progress and warning messages.
    """

    inputs: np.ndarray
    targets: np.ndarray
    notes: tuple = ()

    @property
    def count(self):
        """Count the windows.

        This is N, the first dimension of inputs. It is 0 for a fire that produced
        no windows.

        Returns:
            int: the number of windows.
        """
        return len(self.inputs)


@dataclass(frozen=True)
class DayFrame:
    """Hold one day's 27 stored channels, cropped, and that day's fire masks.

    Each day combines three GeoTIFFs: the VIIRS day bands, the VIIRS night
    bands and the FirePred weather, terrain and fuel bands. They are cropped to
    the fire's fixed 256 x 256 square, and missing values become 0.

    Attributes:
        image (np.ndarray): (27, 256, 256) float32; day bands 0-5, night 6-7, FirePred 8-26.
        af (np.ndarray): (256, 256) bool, active-fire detections of the day.
        ba (np.ndarray): (256, 256) bool, the NIFC burned-area perimeter of the day.
        notes (tuple[str, ...]): messages about missing or unusual night files.
        debug (str): full-image vs crop label counts, filled only when asked.
    """

    image: np.ndarray
    af: np.ndarray
    ba: np.ndarray
    notes: tuple = ()
    debug: str = ""


def read_tiff(file_path):
    """Read one GeoTIFF into an array.

    rasterio is imported here, not at module load, so the dataset masters can
    be imported on a machine without geospatial packages. The profile carries
    the georeference.

    Args:
        file_path (str | Path): the GeoTIFF.

    Returns:
        tuple[np.ndarray, dict]: (array (bands, rows, cols), rasterio profile).
    """
    import rasterio  # imported here so the dataset masters load without rasterio

    with rasterio.open(file_path, "r") as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile


def empty_windows(length, notes):
    """Make a FireWindows that holds no windows.

    The arrays still have the right number of dimensions, with zero windows, so
    callers can treat every result the same. The notes say why there are none.

    Args:
        length (int): input days per window (the T dimension).
        notes (list[str]): messages explaining the empty result.

    Returns:
        FireWindows: inputs (0, 27, length, 256, 256) and targets (0, 256, 256).
    """
    return FireWindows(
        np.zeros((0, N_STORED_CHANNELS, length, CROP_SIZE, CROP_SIZE), dtype=np.float32),
        np.zeros((0, CROP_SIZE, CROP_SIZE), dtype=np.float32),
        tuple(notes),
    )


def binary_mask(array):
    """Turn a raw label band into a true/false fire mask.

    Only finite values above 0 count as fire. NaN, infinity, 0 and negative
    no-data values are background. This must happen before masks are combined,
    because NaN would otherwise count as True.

    Args:
        array (np.ndarray): a raw label band.

    Returns:
        np.ndarray: bool mask of the same shape.
    """
    return np.isfinite(array) & (array > 0)


def read_day_frame(file_name, offset=CROP_OFFSET, size=CROP_SIZE, debug=False):
    """Read one day's three GeoTIFFs and crop them to the fire's square.

    The VIIRS_Day file gives bands 0-5 and the fire masks; the matching
    VIIRS_Night file gives two night bands (zeros when missing or unusable);
    the FirePred file gives 19 weather, terrain and fuel bands. NaN and
    infinity become 0 so the model never sees them.

    Args:
        file_name (str): the local VIIRS_Day GeoTIFF; the night and FirePred
            files are found next to it by name.
        offset (int): first row/column of the crop.
        size (int): crop size in pixels.
        debug (bool): also count fire pixels in the full image and the crop.

    Returns:
        DayFrame: the cropped channels, the day's masks and any notes.

    Raises:
        ValueError: on too few day bands, a day/night grid mismatch or a wrong crop shape.
    """
    notes = []
    array_day, day_profile = read_tiff(file_name)
    if array_day.shape[0] < 7:
        raise ValueError(
            f"{file_name}: expected at least 7 day bands, "
            f"found {array_day.shape[0]}"
        )
    original_shape_x = array_day.shape[1]
    original_shape_y = array_day.shape[2]

    # Night bands (I4, I5 at night) come from the matching VIIRS_Night file.
    # Missing or broken night files become zeros instead of stopping the build.
    night_file = file_name.replace("VIIRS_Day", "VIIRS_Night")
    array_night = None
    if os.path.exists(night_file):
        array_night, night_profile = read_tiff(night_file)
        if array_night.shape[0] == 5:
            notes.append(f"{night_file}: correcting day/night alignment")
            array_night = array_night[3:, :, :]

        if array_night.shape[0] < 2:
            # Unusable night capture: a missing value, so zeros
            # (TS-SatFire paper), like a missing night file.
            notes.append(f"{night_file}: {array_night.shape[0]} band(s); using zeros")
            array_night = None
        elif array_night.shape[1:] != array_day.shape[1:]:
            # The crop is by pixel index, so it only matches the
            # day crop when both rasters share one grid.
            if night_profile["transform"] != day_profile["transform"]:
                raise ValueError(
                    f"Day/night grid mismatch for {file_name}: "
                    f"{array_day.shape[1:]} versus "
                    f"{array_night.shape[1:]}"
                )
            notes.append(
                f"{night_file}: size {array_night.shape[1:]} differs "
                f"from day {array_day.shape[1:]} on the same grid"
            )

    # No usable night capture: two all-zero bands keep the channel layout fixed.
    if array_night is None:
        array_night = np.zeros((2, original_shape_x, original_shape_y), dtype=np.float32)

    # FirePred holds the 19 auxiliary bands (vegetation, weather, forecast, terrain, drought, land cover).
    array_pred, _ = read_tiff(file_name.replace("VIIRS_Day", "FirePred"))

    # Stack the 27 stored channels for this day, cropped to the fire square:
    # 6 day bands + 2 night bands + 19 FirePred bands.
    row_slice = slice(offset, offset + size)
    col_slice = slice(offset, offset + size)
    img = np.concatenate(
        (
            array_day[:6, row_slice, col_slice],
            array_night[:, row_slice, col_slice],
            array_pred[:, row_slice, col_slice],
        ),
        axis=0,
    )
    img = img[:, :size, :size]
    expected_shape = (N_STORED_CHANNELS, size, size)
    if img.shape != expected_shape:
        raise ValueError(f"{file_name}: expected cropped input shape {expected_shape}, found {img.shape}")

    # Model inputs must be finite before maximum accumulation.
    img = np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels for this day: band 6 = active fire detections, band 7 = NIFC
    # burned-area perimeter (missing on some days, then treated as empty).
    # Explicit masks first: np.logical_or on raw arrays would read NaN as True.
    af = binary_mask(array_day[6, row_slice, col_slice])
    if array_day.shape[0] >= 8:
        ba = binary_mask(array_day[7, row_slice, col_slice])
    else:
        ba = binary_mask(np.zeros((size, size), dtype=np.float32))

    debug_line = ""
    if debug:
        af_full = binary_mask(array_day[6])
        if array_day.shape[0] >= 8:
            ba_full = binary_mask(array_day[7])
        else:
            ba_full = np.zeros((original_shape_x, original_shape_y), dtype=bool)
        debug_line = " ".join((
            os.path.basename(file_name),
            f"AF full={np.count_nonzero(af_full)}",
            f"AF crop={np.count_nonzero(af)}",
            f"BA full={np.count_nonzero(ba_full)}",
            f"BA crop={np.count_nonzero(ba)}",
        ))
    return DayFrame(img, af, ba, tuple(notes), debug_line)


def build_fire_windows(
    loc_name,
    local_map,
    length=TS_LENGTH,
    interval=TS_INTERVAL,
    label_sel=1,
    debug=False,
):
    """Build one fire's input windows and next-day targets from its daily GeoTIFFs.

    A window starting on day i stacks days i .. i+length-1 as the input and uses
    day i+length as the target; a new window starts every `interval` days. The
    target is the pixels burned on the target day that were not burned by the
    last input day (the fire's growth). The heat bands keep a running maximum,
    so a pixel hot earlier in the window stays hot. Nothing is printed: messages
    are returned in the result's notes.

    Args:
        loc_name (str): the fire id, used in messages.
        local_map (dict[str, str]): {source repo path: local file path} of the fire's GeoTIFFs.
        length (int): input days per window.
        interval (int): days between window starts.
        label_sel (int): 1 = active fire only, 0 = active fire plus NIFC perimeters.
        debug (bool): add per-day label counts to the notes.

    Returns:
        FireWindows: the windows (possibly none) and notes.

    Raises:
        ValueError: if a processed window contains NaN or infinity.
    """
    notes = []
    # Step 1: the fire's daily frames, in date order. Each day has one VIIRS_Day
    # GeoTIFF; its night and FirePred (weather/terrain/fuel) files sit next to it.
    day_files = sorted(
        path
        for remote_name, path in local_map.items()
        if "VIIRS_Day" in remote_name
    )

    if not day_files:
        notes.append(f"{loc_name}: no VIIRS_Day files")
        return empty_windows(length, notes)

    # Step 2: output geometry. Every frame is cropped to the same 256 x 256 square
    # (rows/cols 128..384), so a fire's windows all cover the same ground.
    n_channels = N_STORED_CHANNELS
    output_shape_x = CROP_SIZE
    output_shape_y = CROP_SIZE

    # Windows go straight into one (windows, C, T, H, W) array. Stacking
    # per-window arrays, transposing and copying again tripled the RAM a
    # long fire needs (about 10 GiB for an 87-day fire).
    n_windows = sum(
        1
        for i in range(0, len(day_files), interval)
        if i + length < len(day_files)
    )
    loc_x = np.zeros(
        (n_windows, n_channels, length, output_shape_x, output_shape_y),
        dtype=np.float32,
    )
    loc_y = np.zeros(
        (n_windows, output_shape_x, output_shape_y),
        dtype=np.float32,
    )

    # Label accumulators must be Boolean. Raw NaNs must never be passed
    # directly to np.logical_or because bool(np.nan) is True.
    # Step 3: running state carried from day to day.
    #   ba_label / af_acc_label: every pixel ever burned so far (NIFC perimeter / active fire).
    #   max_img: per-pixel maximum of each channel so far (used for the heat bands).
    #   new_base_*: a saved copy of that state, restored before the next window starts.
    ba_label = np.zeros(
        (output_shape_x, output_shape_y),
        dtype=bool,
    )
    af_acc_label = np.zeros_like(ba_label)
    new_base_acc_label = af_acc_label.copy()
    new_base_ba_label = ba_label.copy()

    max_img = np.zeros(
        (n_channels, output_shape_x, output_shape_y),
        dtype=np.float32,
    )
    new_base_max_img = max_img.copy()

    # Step 4: slide a window over the days. Window k starts on day k * interval and
    # uses `length` input days plus the next day as its target.
    for window_idx, i in enumerate(range(0, len(day_files), interval)):
        # A window needs `length` input frames plus one target frame.
        if i + length >= len(day_files):
            notes.append(f"{loc_name}: dropping incomplete tail")
            break

        # Read the window's days one at a time: j = 0 .. length-1 are inputs, j = length is the target day.
        for j in range(length + 1):
            frame = read_day_frame(day_files[i + j], debug=debug)
            notes.extend(frame.notes)
            if frame.debug:
                notes.append(frame.debug)
            af, ba = frame.af, frame.ba

            # Heat memory: the thermal bands (day I4/I5 = channels 3-4, night I4/I5 =
            # channels 6-7) are replaced by their running maximum, so a pixel that was hot
            # earlier in the window still looks hot today. All other channels keep today's value.
            max_img = np.maximum(frame.image, max_img)
            img = np.concatenate(
                (
                    frame.image[:3, ...],
                    max_img[3:5, ...],
                    frame.image[[5], ...],
                    max_img[6:8, ...],
                    frame.image[8:, ...],
                ),
                axis=0,
            )

            # Accumulate: once a pixel has burned it stays burned. label_sel picks the
            # burned-area definition: 1 = active fire only, otherwise active fire + NIFC perimeters.
            ba_label = np.logical_or(ba_label, ba)
            af_acc_label = np.logical_or(af_acc_label, af)

            if label_sel == 1:
                final_label = af_acc_label
            else:
                final_label = np.logical_or(
                    af_acc_label,
                    ba_label,
                )

            # Save state at the next stride boundary. Restoring this
            # state after the window prevents overlapping windows from
            # leaking future information into their early frames.
            if j == interval - 1:
                new_base_acc_label = af_acc_label.copy()
                new_base_ba_label = ba_label.copy()
                new_base_max_img = max_img.copy()

            # Input day: store the frame and remember the burned area so far.
            # Target day: the label is what is burned now but was not burned on the last
            # input day, i.e. only the newly burned pixels (the fire's growth).
            if j < length:
                prev_final_label = final_label.copy()
                loc_x[window_idx, :, j] = img
            else:
                new_mask = np.logical_and(
                    np.logical_not(prev_final_label),
                    final_label,
                )
                loc_y[window_idx] = new_mask

                if debug:
                    notes.append(
                        f"previous accumulated: {np.count_nonzero(prev_final_label)} "
                        f"current accumulated: {np.count_nonzero(final_label)} "
                        f"new target: {np.count_nonzero(new_mask)}"
                    )

        # Rewind the running state to the next window's start, so overlapping windows
        # never see days that lie after their own start.
        af_acc_label = new_base_acc_label.copy()
        ba_label = new_base_ba_label.copy()
        max_img = new_base_max_img.copy()

    # Step 5: sanity checks and a short report of how many windows contain new fire.
    if n_windows == 0:
        notes.append(f"{loc_name}: not enough time-series frames")
        return empty_windows(length, notes)

    # One window at a time, so the check needs no full-size mask.
    for window in loc_x:
        if not np.isfinite(window).all():
            raise ValueError(f"{loc_name}: processed inputs contain NaN/Inf")
    if not np.isfinite(loc_y).all():
        raise ValueError(f"{loc_name}: processed labels contain NaN/Inf")

    positive_per_window = np.count_nonzero(
        loc_y > 0,
        axis=(1, 2),
    )
    notes.append(
        f"{loc_name}: {np.count_nonzero(positive_per_window)}/"
        f"{len(positive_per_window)} windows contain positives; "
        f"{positive_per_window.sum()} total positive pixels"
    )
    if positive_per_window.sum() == 0:
        notes.append(f"WARNING: {loc_name} has no positive prediction targets")

    return FireWindows(loc_x, loc_y, tuple(notes))

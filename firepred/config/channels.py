"""The input channels: what each stored channel is and how it becomes model input.

Level 0 (config). Used by data/ (normalization and preprocessing), models/
(input width) and training/ (channel checks).

A stored window holds 27 channels, in the order of the TS-SatFire paper's
Table 1 (figures/Bands.png), 0-based here. Before the model sees it:
  * every channel except the angles and land cover is standardized with
    NORMALIZATION_MEAN/STD;
  * the three angle channels (degrees) become sin(angle);
  * land cover (integer class 1..17, 0 = missing) becomes 17 one-hot planes.
So the model receives 27 - 1 + 17 = 43 channels. TESSERA mode appends 128
embedding bands plus 1 validity mask: 43 + 129 = 172.
"""

# Stored channel order. Channels 0-5 are VIIRS day bands, 6-7 VIIRS night
# bands, 8-26 the FirePred auxiliary bands (vegetation, weather, terrain,
# fuel). Channels 3, 4, 6, 7 hold the running maximum over the window's days
# (see firepred/prep/windows.py).
STORED_CHANNEL_NAMES = (
    "VIIRS I1",                      # 0
    "VIIRS I2",                      # 1
    "VIIRS I3",                      # 2
    "VIIRS I4",                      # 3  running max
    "VIIRS I5",                      # 4  running max
    "VIIRS M11",                     # 5
    "VIIRS I4 night",                # 6  running max
    "VIIRS I5 night",                # 7  running max
    "NDVI",                          # 8
    "EVI",                           # 9
    "total precipitation",           # 10
    "wind speed",                    # 11
    "wind direction",                # 12 degrees
    "min temperature",               # 13
    "max temperature",               # 14
    "energy release component",      # 15
    "specific humidity",             # 16
    "slope",                         # 17
    "aspect",                        # 18 degrees
    "elevation",                     # 19
    "PDSI",                          # 20 Palmer drought severity index
    "land cover",                    # 21 class 1..17
    "total precipitation surface",   # 22
    "forecast wind speed",           # 23
    "forecast wind direction",       # 24 degrees
    "forecast temperature",          # 25
    "forecast specific humidity",    # 26
)
N_STORED_CHANNELS = 27

# Angles in degrees. Must stay a list: x[[12, 18, 24]] selects three
# channels, while x[(12, 18, 24)] would index a single element. [released]
DEGREE_CHANNELS = [12, 18, 24]

# Land-cover class per pixel, replaced by one-hot planes. [released]
LANDCOVER_CHANNEL = 21
N_LANDCOVER_CLASSES = 17

# Per-channel training statistics of the 27 stored channels. [released]
NORMALIZATION_MEAN = [
    18.224253,
    26.95519,
    20.09066,
    318.25967,
    308.78717,
    14.165086,
    291.29214,
    288.97382,
    5110.5547,
    2556.2627,
    0.3907487,
    3.4994626,
    216.23518,
    276.5463,
    291.8275,
    70.32086,
    0.0054306216,
    10.120554,
    175.33012,
    1290.8367,
    -1.5219007,
    7.3989105,
    7.584937,
    1.4395763,
    3.306973,
    19.259102,
    0.0057929577,
]
NORMALIZATION_STD = [
    15.438321,
    14.408274,
    10.552524,
    13.1312475,
    12.155249,
    9.652911,
    12.435288,
    8.750125,
    2400.766,
    1206.8983,
    2.37979,
    1.6343528,
    85.730644,
    47.332256,
    50.045837,
    22.48386,
    0.0021515382,
    8.429097,
    104.73222,
    823.01483,
    1.9954495,
    4.1257873,
    26.547232,
    1.2017097,
    48.207355,
    5.4114914,
    0.0017134654,
]

# Model input width without TESSERA: land cover's 1 channel becomes 17.
BASE_MODEL_CHANNELS = N_STORED_CHANNELS - 1 + N_LANDCOVER_CLASSES  # 43

# TESSERA: 128 embedding bands per pixel plus one validity mask. [ours]
TESSERA_BANDS = 128
TESSERA_CHANNELS = TESSERA_BANDS + 1


def model_input_channels(use_tessera):
    """43, or 172 when TESSERA channels are appended."""
    return BASE_MODEL_CHANNELS + (TESSERA_CHANNELS if use_tessera else 0)

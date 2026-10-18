# 1. The question

> "If you can't explain it simply, you don't understand it well enough."
> Before any architecture, pin down exactly what goes in, what comes out, and
> how the answer is graded.

## The question in one sentence

*Given the last six days of a wildfire, which pixels will burn **for the first
time** tomorrow?*

Each word matters:

- **pixels**: the answer is a map, a yes/no decision for every pixel, not one
  number per fire.
- **for the first time**: already-burned ground doesn't count. The target is
  the *growth* of the fire.
- **tomorrow**: one day ahead, from one window of the past.

## What one example looks like

Picture a 96 km x 96 km square of ground centred on a fire, cut into a
256 x 256 grid. Each cell is one VIIRS pixel, 375 m on a side (the satellite's
resolution). For each of the 6 days before "tomorrow", every cell carries a
column of measurements.

| Group | Channels | What it tells the model |
|---|---|---|
| Satellite, daytime (VIIRS I1-I5, M11) | 6 | Reflectance and **heat**. I4 and I5 are thermal bands, and fire is hot. |
| Satellite, night (I4, I5) | 2 | Heat again, about 12 hours later |
| Vegetation (NDVI, EVI) | 2 | How much fuel there is and how green it is |
| Weather (precipitation, wind speed/direction, temperatures, humidity, energy release component) | 8 | Whether conditions favour spread and which way the wind pushes |
| Forecast (wind speed/direction, temperature, humidity) | 4 | The same for *tomorrow*, as forecast |
| Terrain (slope, aspect, elevation) | 3 | Fire runs uphill; aspect sets which slopes are sun-dried |
| Drought (PDSI), land cover | 2 | How dry the region is, and what is growing there |

That is 27 stored channels (`firepred/config/channels.py`, which names every
one). Three things happen before the model sees them
(`firepred/data/transforms.py`):

1. **Standardize.** Every channel except the angles and land cover becomes
   (value − mean) / std, with fixed statistics from the training set, so
   elevation in metres and humidity in kg/kg land on comparable scales.
2. **Angles become sines.** Wind direction, aspect and forecast wind direction
   are in degrees. To a network, 359 and 1 look far apart even though they
   point almost the same way. The code feeds `sin(angle)` instead. (Only the
   sine: 0° and 180°, north and south, both become 0. Chapter 9, B4.)
3. **Land cover becomes 17 yes/no channels.** "Class 7" is a name, not a
   number: class 7 is not "more" than class 3. One-hot encoding gives each class
   its own channel. So 27 - 1 + 17 = **43 channels**.

One training example, a *window*, is therefore a block of numbers:

```
input  x: 43 channels x 6 days x 256 x 256 pixels  = 16,908,288 numbers
target y: 256 x 256 pixels, each "newly burned tomorrow" (1) or not (0)
```

The model's job is to turn 16.9 million numbers into 65,536 yes/no answers.

With TESSERA (`--tessera`) each pixel also gets 128 numbers summarising
what the ground looked like over the **previous year**, from a satellite
foundation model, plus one mask channel saying whether that summary exists.
Those 129 channels don't change from day to day, so they are stored once per
fire and copied across the 6 days on the GPU (`firepred/data/loader.py:
batch_inputs`), giving 172 channels.

## Where the target comes from

`firepred/prep/windows.py` builds the target from two daily masks:
*active fire* (pixels the satellite flagged as burning that day) and
*burned area* (official NIFC fire perimeters). It **accumulates** them: once a
pixel has burned, it stays burned. The target is then

```
target = burned_by(tomorrow)  AND NOT  burned_by(today)
```

which is "the difference between the burned area masks of the last observed day
and the next day" (TS-SatFire paper). Why not simply predict tomorrow's whole
burned area? Because almost all of it already burned today. A model that copies
today's map would score brilliantly and know nothing about spread. Subtracting
today forces the model to predict *change*.

There is a second, quieter piece of memory: thermal channels 3, 4, 6 and 7 store
the **running maximum** heat seen so far in the window, not that day's heat. A
pixel that burned three days ago still "remembers" being hot.

## The needle in the haystack

New fire is rare. The paper weights fire pixels **446.78 times** more than
background in the loss, a weight "set from the class distribution of the
training set". Read it as: for every newly burned pixel there are roughly 447
that don't burn. In a 256 x 256 window that is about 150 burning pixels among
65,536. Many windows have none at all: a fire that stalls overnight.

**A reality check on real windows.** Seven training fires from the 6-day,
1-day-stride window repo (84 windows, built with the paper's label rule) give:

| | |
|---|---|
| New-fire share | 0.28% of pixels, about **1 in 353** |
| Windows with no new fire | 13 of 84 (**15%**) |
| New-fire pixels per window | mean 185; per-fire medians from 0 to 233; maximum 1,854 |

That's a small sample (7 of about 125 training fires), but it shows the shape:
the imbalance is in the paper's range, and it is **heavy-tailed**. Most windows
hold a few dozen new pixels, a few hold over a thousand, and one in seven holds
none.

This one fact shapes everything that follows:

- A model that always answers "no fire" is right **99.8%** of the time and
  useless. Accuracy is the wrong score.
- The loss has to be built so the rare class still pulls hard (chapter 7).
- The metrics only look at the fire class.

## How the answer is graded

For each window, compare the predicted fire pixels P with the true fire pixels T:

```
TP = pixels in both    FP = predicted but did not burn    FN = burned but missed

F1  = 2 TP / (2 TP + FP + FN)        (also called Dice)
IoU = TP / (TP + FP + FN)            (the paper's Equation 2)
```

Both ignore true negatives, the ocean of correctly predicted "no fire", so a
model gets no credit for the easy part. `firepred/training/evaluation.py` scores
each 2021 test fire as the mean over its windows, then averages over fires,
so a 40-day fire counts as much as a 5-day one. One convention matters: when
a window has no fire **and** the model predicts none, it scores a perfect 1
(`firepred/training/metrics.py`). Chapter 9 returns to what that does to the
numbers.

## The setup in numbers

| | |
|---|---|
| Fires | 179 wildfires (mostly in the US), 2017-2021; train/validation from 2017-2020, test = 24 fires from 2021 |
| Window | 6 input days -> the next day (a new window starts every day: `-ts 6 -it 1`) |
| Grid | 256 x 256 pixels x 375 m = 96 km x 96 km |
| Input | 43 channels (172 with TESSERA) |
| Output | 2 numbers per pixel: "background" and "fire" logits |
| Imbalance | about 1 new-fire pixel per 447 (1 in 353 in a 7-fire sample; 15% of windows empty) |
| Score | F1 and IoU of the fire class, per fire, then averaged |

Next: [2. From pixels to pictures](02-from-pixels-to-pictures.md). Why
looking at one pixel at a time can't work, and what convolutional networks fix.

# SwinUNETR-3D explainer: verified research notes

Compiled 2026-10-17. Each claim cites a numbered reference ([n], list at bottom), the released code, or this repo. **UNVERIFIED** = not confirmed against a primary source. **Inference** = my reasoning from cited facts, not a claim the source makes. Numbers are copied from source tables; arithmetic is marked "derived".

## 1. TS-SatFire (Zhao, Gerard, Ban; Sci. Data 12, 1817, 2025) [1]

**Dataset**
- 179 fire events, contiguous US, Jan 2017 – Oct 2021; 3552 VIIRS images; 71 GB; tasks: active fire (AF), daily burned area (BA), next-day progression prediction (Pred). Published online 2025-11-19.
- Pred split: train 2017–2020 (34+34+8+49 events), validation 13 events from 2017–2020, test 24 fires from 2021.
- Inputs: VIIRS I1–I5, M11, and night I4/I5 at 375 m, plus auxiliary data that follow the WildfireSpreadTS setup (NDVI/EVI, weather, forecasts, topography, PDSI, land cover). 27 stored channels (paper Table 1). In this repo, those 27 channels become 43 after one-hot encoding the 17 land-cover classes (`run_spatial_temp_model_pred.py:40`).
- Pred label = burned-area mask on day t+1 minus burned-area mask on the last observed day, so only newly burned pixels count. The paper's reason: using the raw t+1 burned-area mask would reward copying the current burned area.
- Preprocessing: I4/I5 take the pixel-wise max over the current and all earlier timestamps; missing values are set to 0; auxiliary data are resampled to 375 m bilinearly.

**Pred training setup, from the paper's Methods**
- The loss switched from DiceLoss to DiceCELoss because models stayed stuck in a local minimum under Dice alone. The authors attribute this to class imbalance.
- CE weight `[1.0, 446.7836]`, set from the class distribution of the training set.
- Grid search: 1:1 Dice+CE versus CE alone, crossed with lr ∈ {0.01, 0.001}.
- Final settings: U-Net-3D, UNETR-3D and SwinUNETR-3D used 1:1 Dice+CE with lr 0.001, at batch sizes **64 / 4 / 8**. Attention-U-Net-3D kept Dice loss with lr 0.0012 and batch 8.
- SwinUNETR-3D: patch (1,2,2), so patches split only space; window (T,4,4), so attention covers the full time series. Feature size 36 was picked from {24, 36}, and larger sizes were not tested because of hardware limits. That choice appears in the AF/BA hyperparameter paragraph; that it also applies to Pred is implied but not stated. U-Net, UNETR and SwinUNETR are each held near 30M parameters.
- Input length: 6 days (Table 3).

**What the released code differs on** (github.com/zhaoyutim/TS-SatFire, `run_spatial_temp_model_pred.py`, last commit 2025-12-11)
- Config: image (T,256,256), window (T,4,4), depths (2,2,2,2), BatchNorm, `attn_version='v1'`, Adam, `MAX_EPOCHS=100`. A `weight_decay = lr/10` value is computed and logged, but never passed to the optimizer.
- The criterion in the released code is `DiceLoss(sigmoid=True)`. Neither DiceCE nor the 446.78 weight appears in the public repo.
- Test metric: logits are averaged over the time axis (`.mean(2)`), then `sigmoid > 0.5` is applied to channel 1. Each frame is scored with sklearn `f1_score` / `jaccard_score` using `zero_division=1.0`. Scores are averaged over a fire's frames, then over fires. This repo's `run_spatial_temp_model.py:345` uses the same convention.

**Pred benchmark (paper Table 3; 6 input days; test year 2021)**

| Model | Params | F1 | IoU |
|---|---|---|---|
| U-Net-3D | 31.7M | **0.375** | **0.338** |
| SwinUNETR-3D | 33.2M | 0.374 | 0.331 |
| UNETR-3D | 34.8M | 0.371 | 0.336 |
| Attention-U-Net-3D | 94.5M | 0.354 | 0.312 |

- **The winner is U-Net-3D, not SwinUNETR-3D.** Margins (derived): +0.001 F1 and +0.007 IoU over SwinUNETR-3D. The spread across all four models is 0.021 F1.
- Seed variation (Table 4, 3 seeds, Pred): F1 0.371 ± 0.003, IoU 0.332 ± 0.003. The top three models therefore differ by about one seed standard deviation or less, which is effectively a tie (inference).
- Internal inconsistency: the Table 4 footnote names SwinUNETR-3D as the best Pred model, but Table 3 ranks U-Net-3D marginally higher. Table 3's caption mentions only AF and BA, yet the table includes Pred.
- SwinUNETR-3D does win BA (F1 0.855, IoU 0.768), and UNETR-3D wins AF among spatio-temporal models (0.811 / 0.706).
- The paper's explanation: all four baselines score similarly, and segmentation nets built to extract spatial features are poorly suited to inferring future states.
- Feature importance (zero one feature at a time): removing I2 cuts F1 by 11%, M11 by 6.5%; most others barely matter. Paper's three hypotheses: auxiliary data too coarse, segmentation models fail to extract it, or naive concatenation is poor fusion. SwinUNETR predictions show many FPs and FNs (Fig. 11).
- Time-length ablation (Table 5, SwinUNETR-3D, Pred):

| T | 6 | 4 | 2 |
|---|---|---|---|
| F1 | 0.374 | 0.354 | 0.366 |
| IoU | 0.331 | 0.317 | 0.321 |

  The paper says Pred scores rise with T, but T=2 beats T=4. The T=6 versus T=2 gain is +0.008 F1, about 3 seed standard deviations (derived). For BA and AF, shorter series score better.

## 2. Model lineage (one line each)

- **U-Net** (Ronneberger et al., MICCAI 2015) [2]: encoder–decoder with skip connections, so fine spatial detail survives downsampling.
- **3D U-Net** (Çiçek et al., MICCAI 2016) [3]: the same design with 3D convolutions. TS-SatFire's U-Net-3D swaps 2D convolutions for 3D and never downsamples time [1].
- **Attention U-Net** (Oktay et al., MIDL 2018) [4]: attention gates on the skips suppress irrelevant regions.
- **Swin Transformer** (Liu et al., ICCV 2021) [5]: hierarchical ViT. Self-attention runs inside local windows that shift between layers, giving cost linear in image size.
- **Video Swin** (Liu et al., CVPR 2022) [6]: shifted windows extended to 3D space-time windows; locality as the inductive bias for video.
- **Swin V2** (Liu et al., CVPR 2022) [7]: residual post-norm, scaled cosine attention and log-spaced continuous position bias, for stable scaling and transfer across resolutions. The TS-SatFire code ships V1 and V2 attention and uses V1.
- **UNETR** (Hatamizadeh et al., WACV 2022) [8]: plain ViT encoder on 3D patches; transformer features at several depths feed a CNN decoder through skips.
- **Swin UNETR** (Hatamizadeh et al., BrainLes/MICCAI 2021, LNCS 2022, pp. 272–284) [9]: Swin encoder plus CNN decoder via multi-resolution skips; built for 3D MRI brain tumours. Self-supervised pretraining of the Swin UNETR encoder on CT: Tang et al., CVPR 2022 [10].

## 3. Wildfire spread prediction literature

- **Next Day Wildfire Spread (NDWS)**, Huot et al., IEEE TGRS 60, 2022 [11]
  - 18,545 samples of 64 × 64 km tiles at 1 km; one input day predicts day t+1; MODIS fire masks (per [1]).
  - Convolutional autoencoder: AUC-PR 28.4%, precision 33.6%, recall 43.1% (derived F1 ≈ 0.378). Persistence baseline: AUC-PR 11.5%. Logistic regression and random forest score lower.
- **WildfireSpreadTS (WSTS)**, Gerard, Zhao, Sullivan, NeurIPS 2023 Datasets & Benchmarks [12]
  - 607 fires, 13,607 images, 2018–2021; 23 channels; 375 m VIIRS active-fire maps; 24 h horizon.
  - Metric: average precision (AP), chosen because the data are imbalanced and AP covers all thresholds. Evaluation is 12-fold cross-validation over years.
  - Baselines, AP for Veg / Multi / All feature sets:
    - Persistence 0.193
    - Logistic regression: max 0.288
    - Res18 U-Net, 1 day: 0.328 / 0.341 / 0.341
    - Res18 U-Net, 5 days: 0.333 / 0.344 / 0.325
    - ConvLSTM, 5 days: 0.306 / 0.310 / 0.292
    - **UTAE, 5 days: 0.372 / 0.350 / 0.321**
  - Multi-day finding: the best model is UTAE on 5 days of vegetation features, which the authors read as support for multi-temporal modelling. However, more days or more features **sometimes reduced** performance. The authors call this an optimization difficulty in high-dimensional input, and note that explicit spatio-temporal models were hard to optimize while the U-Net was easy.
  - Fold-to-fold standard deviations are about 0.08–0.09 AP, much larger than most differences between settings.
- **WSTS+**, Lahrichi, Bova, Johnson, Malof, WACV 2026 [13]
  - Changes tested: pretrained encoders, focal loss, and selecting models by validation AP instead of F1.
  - New state of the art: **UTAE(Res18), 5 days: 0.478 / 0.477 / 0.475 AP**. Res18-Unet, 5 days, Veg: 0.472.
  - **SwinUnet trails** (5 days: 0.447 / 0.453 / 0.435), and so does SegFormer. With controlled tuning, attention models did not beat ResNet-18.
  - Ablation (Res18-Unet, Veg): dropping focal loss costs 24.18% AP. Also dropping AP-based selection costs 29.45% in total. Pretraining made almost no difference in that ablation.
  - Random 4-fold cross-validation inflates AP (Res18 Veg: 0.455 under year-wise folds versus 0.527 under random folds).
  - The WSTS+ extension covers 2016–2023, 1,005 fires and 24,462 images. Scores on it are lower (about 0.35–0.36).
- **WildfireSpreadBench**, Gopakumar & Pannozzo, NeurIPS 2026 Climate Change AI workshop [14]
  - Six models on the WSTS 2021 test year. Model rankings flip between AP and F1/IoU at threshold 0.5.
  - BCE U-Net has the best AP (0.562, Veg) but F1 0.308 and predicts 4.8× the observed fire area.
  - ConvLSTM has the best F1 (0.583) and IoU (0.411) on Veg; ResNet18 U-Net F1 is 0.568 / 0.585 (Veg / All).
  - UTAE predicts 15.9–23× the observed fire area; a flow-matching generative model reaches F1 0.402 / 0.425.
  - Going from 7 to 23 channels changes AP by 0.03 on average.
- **NDWS follow-ups (preprints)**: FireSenseNet [15] reaches F1 0.4176 / AUC-PR 0.3435 (SegFormer F1 0.3502); putting the previous day's fire in the target and relabelling unknown pixels as 0 inflates F1 by 44–50%. Esparza et al. [16]: SwinUNETR plus wind/slope attention bias, retrieval and gating reaches F1 0.4216 / AUC-PR 0.3673 (mixed ensemble 0.4292 / 0.3790).
- **Lightweight WSTS models (preprints)**: TD-FusionUNet [17] F1 0.591 at 370k params; ShearFuse-UNet [18] F1 0.596 at 267k, versus 0.589 for Res18 U-Net.
- **Other**: VIIRS VNP14 is a better next-day target than MODIS MOD14 [19]. Predictive uncertainty concentrates near fire perimeters [20]; a UTAE AP of 0.378 attributed to [20] is **UNVERIFIED**.
- **Comparability warning (inference):** F1 scores on WSTS (about 0.57–0.60) and TS-SatFire Pred (about 0.37) cannot be compared. WSTS targets the next-day active-fire map, while TS-SatFire targets only newly burned area [1]. Averaging and metrics also differ: per-frame macro F1 versus pooled scores or AP.

## 4. Candidate architecture pathways

**4a. Factorized space-time attention** (TimeSformer, ICML 2021 [21]; ViViT, ICCV 2021 [22])
- Evidence: TimeSformer's "divided" attention (temporal, then spatial, in each block) was its most accurate variant on video classification; ViViT offers several factorized variants.
- Why: SwinUNETR-3D spends attention on all T steps inside 4×4 spatial windows, while a wind-driven run can cross many 375 m pixels in a day. Factorizing buys larger or global spatial attention at similar cost (inference).
- Risk: data-hungry with ~125 training fires; attention backbones trailed ResNet-18 in WSTS+ [13].

**4b. U-TAE** (Garnot & Landrieu, ICCV 2021) [23]
- Design: shared per-date CNN encoder; temporal attention at the lowest resolution yields masks applied to every skip connection.
- Evidence: best on WSTS and WSTS+ [12,13] at 1.1M params (14.6M with ResNet-18 encoder).
- Why: most signal sits in the last day's fire state, and temporal attention learns how much history to use.
- Risk: predicts 15.9–23× the observed area at threshold 0.5 [14], so it needs threshold tuning; collapses time early.

**4c. Recurrent and video-prediction models** (ConvLSTM, NeurIPS 2015 [24]; PredRNN, TPAMI 2023 [25]; SimVP, CVPR 2022 [26]; SimVPv2 [27])
- Why: built to forecast future frames, the bias TS-SatFire says segmentation nets lack [1]; SimVP shows an all-CNN encoder–translator–decoder competing with recurrent models.
- Evidence is mixed: ConvLSTM AP 0.310 trails U-Net [12] but has the best F1 in [14].
- Risk: recurrent training instability, small receptive fields, little to gain at T ≤ 6.

**4d. Earthformer** (Gao et al., NeurIPS 2022) [28]
- Design: cuboid attention (local attention within space-time cuboids) plus global vectors, for Earth-system forecasting (SEVIR, ENSO).
- Why: cheap global context for smooth weather fields; cuboid shapes are tunable.
- Risk: designed for dense regression, not sparse binary masks on small data.

**4e. Mamba family**: VMamba (NeurIPS 2024) [29], Vision Mamba (ICML 2024) [30], VideoMamba (ECCV 2024) [31], remote-sensing RS-Mamba [32] and RSMamba [33].
- Why: global receptive field at linear cost; patch-2 tokens for 6×256×256 number 98,304 (derived).
- Risk: **no Mamba next-day fire-spread paper found**, so any benefit is UNVERIFIED; token scan order matters; small data.

**4f. Geospatial foundation models**
- Prithvi-EO-2.0 [34]: burn-scar IoU 76.8 (v1, 100M) up to 83.2 (v2, 600M).
- SatMAE (NeurIPS 2022) [35]: temporal and multispectral masked-autoencoder pretraining.
- Clay [36]: open masked-autoencoder model, no paper; details **UNVERIFIED**.
- TESSERA (CVPR 2026) [37]: annual 128-d embedding per 10 m pixel from Sentinel-1/2 time series.
- Galileo [38]: multimodal including weather; venue **UNVERIFIED**. AnySat (CVPR 2025) [39]: any resolution or modality.
- Why: WSTS proposes self-supervised pretraining against optimization and imbalance problems [12]; WSTS+ found pretraining nearly always helped [13]. Embeddings may describe fuel better than 17 one-hot classes (inference).
- Risk: scale mismatch (annual 10 m vs daily 375 m); static embeddings cannot show fire dynamics; 43 → 172 channels in this repo risks WSTS's "extra features hurt" result [12].

**4g. Physics-informed models and neural operators**
- Background: Rothermel's rate-of-spread model [40]; semi-empirical simulators such as FARSITE [1]; PINN parameter learning for spread models [41]; FNO resolution-invariant PDE operators [42].
- Related work: wind/slope attention bias on SwinUNETR [16]; advection-constrained net on simulations [43].
- Why: TS-SatFire's weather and terrain features go largely unused [1]; physics priors could force their use.
- Risk: no FNO next-day satellite-spread paper found (UNVERIFIED); in [16] the wind bias tracked input wind but only weakly matched observed displacement; 375 m daily data is coarse for front physics.

**4h. Probabilistic and generative forecasting**
- Work: diffusion surrogate on cellular-automata simulations (GMD 2026) [44]; conditional flow matching on WRF-SFIRE [45]; flow matching on WSTS under-predicts, at 0.82–0.84× observed area [14]. WSTS suggests generative models for stochastic spread [12].
- Why: spread at 375 m / 24 h is plausibly stochastic; ensembles give probability maps and perimeter uncertainty [20].
- Risk: sampling cost; mostly simulator-trained; single-sample F1 misrepresents an ensemble.

**4i. Losses for class imbalance**: focal (ICCV 2017) [46], Tversky (MLMI 2017) [47], focal Tversky (ISBI 2019) [48], generalized Dice [49], Lovász-softmax (CVPR 2018) [50], boundary loss (MIDL 2019 / MedIA 2021) [51].
- Evidence: dropping focal loss cost 24% AP in WSTS+ [13].
- Why: positives are ~1/447 of pixels (CE weight [1]). Tversky trades FP against FN toward the F1 operating point; Lovász optimizes an IoU surrogate; boundary loss targets the perimeter, where new burn occurs (inference).
- Risk: boundary loss needs distance maps plus a region loss; focal/Tversky settings shift calibration, so re-tune the threshold.

## 5. Caveats: DiceCE and empty-mask F1

**DiceCE on 2-channel logits (MONAI source [52])**
- `sigmoid=True` reaches only the Dice term; with more than one channel, CE is `nn.CrossEntropyLoss` (softmax). Dice scores σ(z₀), σ(z₁) independently; CE depends only on z₁ − z₀.
- Inference thresholds σ(z₁) > 0.5, i.e. z₁ > 0, while CE's boundary is z₁ > z₀: training and inference disagree (inference from code; no paper found on this combination).
- **`weight` semantics changed in MONAI 1.3.0.** In 1.1–1.2, `ce_weight` touched CE only; from 1.3.0, `weight` also scales the per-class Dice term, so 446.78 multiplies the fire-class Dice.
- With one-hot float targets, PyTorch treats targets as class probabilities and divides the weighted sum by N, not by the summed weights [53], so CE also scales with the weight.
- So "1:1 Dice+CE" means different effective balances by MONAI version and label format; the paper gives no MONAI version (**UNVERIFIED**). This repo's `build_criterion` docstring already notes the both-terms behaviour.
- Empty labels: Dice behaviour depends on reduction axes Φ and smoothing ε [54]. Under MONAI defaults (`batch=False`, ε = 1e-5), a no-new-burn frame gives a fire-channel loss near 1 with almost no gradient, treated like a missing label (inference consistent with [54]); `batch=True` or a larger ε changes this.
- When evaluation uses Dice or IoU, losses built on those metrics beat CE and weighted CE [55].

**F1 with the empty-mask = 1 convention**
- sklearn defines F1 as undefined when TP+FP+FN = 0; `zero_division=1.0` returns 1.0 in that case [56].
- Reinke et al. [57] list empty references and predictions as a pitfall: per-image metrics divide by zero and give invalid scores, so the convention must be stated explicitly. See also Metrics Reloaded [58].
- Consequences (inference): an empty frame with no prediction scores 1.0, but one false-positive pixel drops it to 0; per-fire means depend on the share of no-growth days; the scheme favours conservative models and is not comparable with pooled F1 or AP.
- Related: protocol choices inflated NDWS F1 by 44–50% [15]; AP and thresholded F1 rank models differently [14]. The share of empty frames in the TS-SatFire test set is **UNVERIFIED**; worth computing.

## References

1. Zhao, Gerard, Ban. TS-SatFire. *Scientific Data* 12:1817 (2025). doi:10.1038/s41597-025-06271-3. Code: https://github.com/zhaoyutim/TS-SatFire
2. Ronneberger, Fischer, Brox. U-Net. MICCAI 2015, LNCS pp. 234–241. doi:10.1007/978-3-319-24574-4_28; arXiv:1505.04597
3. Çiçek et al. 3D U-Net. MICCAI 2016, LNCS pp. 424–432. doi:10.1007/978-3-319-46723-8_49
4. Oktay et al. Attention U-Net. MIDL 2018. arXiv:1804.03999
5. Liu et al. Swin Transformer. ICCV 2021. arXiv:2103.14030; https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html
6. Liu et al. Video Swin Transformer. CVPR 2022. arXiv:2106.13230
7. Liu et al. Swin Transformer V2. CVPR 2022. arXiv:2111.09883
8. Hatamizadeh et al. UNETR. WACV 2022. arXiv:2103.10504
9. Hatamizadeh et al. Swin UNETR. BrainLes 2021, LNCS (2022) pp. 272–284. doi:10.1007/978-3-031-08999-2_22; arXiv:2201.01266
10. Tang et al. Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis. CVPR 2022. arXiv:2111.14791
11. Huot et al. Next Day Wildfire Spread. *IEEE TGRS* 60 (2022). doi:10.1109/TGRS.2022.3192974; arXiv:2112.02447
12. Gerard, Zhao, Sullivan. WildfireSpreadTS. NeurIPS 36 (2023) Datasets & Benchmarks, pp. 74515–74529. https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf
13. Lahrichi, Bova, Johnson, Malof. Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark. WACV 2026. arXiv:2502.12003
14. Gopakumar, Pannozzo. WildfireSpreadBench. NeurIPS 2026 CCAI workshop. arXiv:2609.22191
15. Han et al. FireSenseNet. arXiv:2604.07675 (2026, preprint)
16. Esparza et al. Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction. arXiv:2609.17763 (2026, preprint)
17. Luo et al. TD-FusionUNet. arXiv:2602.11672 (2026, preprint)
18. Meco et al. ShearFuse-UNet. arXiv:2606.14071 (2026, preprint)
19. Karlsson et al. Comparing Next-Day Wildfire Predictability of MODIS and VIIRS. arXiv:2503.08580 (2025)
20. Chakravarty. Spatial Uncertainty Quantification in Wildfire Forecasting. NeurIPS 2025 CCAI workshop. arXiv:2510.09666
21. Bertasius, Wang, Torresani. Is Space-Time Attention All You Need for Video Understanding? ICML 2021 (PMLR 139). arXiv:2102.05095
22. Arnab et al. ViViT. ICCV 2021. arXiv:2103.15691
23. Sainte Fare Garnot, Landrieu. Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks (U-TAE). ICCV 2021. arXiv:2107.07933
24. Shi et al. Convolutional LSTM Network. NeurIPS 2015. arXiv:1506.04214
25. Wang et al. PredRNN. *IEEE TPAMI* 45:2208–2225 (2023). doi:10.1109/TPAMI.2022.3165153 (conference version NeurIPS 2017, not checked)
26. Gao et al. SimVP. CVPR 2022. arXiv:2206.05099
27. Tan et al. SimVPv2. IEEE TMM (accepted, per arXiv). arXiv:2211.12509
28. Gao et al. Earthformer. NeurIPS 2022. arXiv:2207.05833
29. Liu et al. VMamba. NeurIPS 2024. arXiv:2401.10166
30. Zhu et al. Vision Mamba. ICML 2024. arXiv:2401.09417
31. Li et al. VideoMamba. ECCV 2024, LNCS pp. 237–255. doi:10.1007/978-3-031-73347-5_14; arXiv:2403.06977
32. Zhao et al. RS-Mamba for Large Remote Sensing Image Dense Prediction. arXiv:2404.02668 (2024)
33. Chen et al. RSMamba. arXiv:2403.19654 (2024)
34. Szwarcman et al. Prithvi-EO-2.0. arXiv:2412.02732 (2024)
35. Cong et al. SatMAE. NeurIPS 2022. arXiv:2207.08051
36. Clay Foundation Model. https://madewithclay.org/ (no paper)
37. Feng et al. TESSERA. CVPR 2026 (https://cvpr.thecvf.com/virtual/2026/poster/37322). arXiv:2506.20380
38. Tseng et al. Galileo. arXiv:2502.09356 (2025)
39. Astruc et al. AnySat. CVPR 2025. arXiv:2412.14123
40. Rothermel. A mathematical model for predicting fire spread in wildland fuels. USDA FS Res. Pap. INT-115 (1972). https://research.fs.usda.gov/treesearch/32533
41. Vogiatzoglou et al. PINNs for parameter learning of wildfire spreading. arXiv:2406.14591 (2024)
42. Li et al. Fourier Neural Operator. arXiv:2010.08895 (ICLR 2021 venue UNVERIFIED here; OpenReview blocked)
43. Lawenda et al. Interpretable Patch-Based Deep Learning for Wildfire Spread Prediction from Ensemble Simulations. arXiv:2609.18555 (2026)
44. Yu et al. Denoising Diffusion Surrogate for Wildfire Spread. *Geosci. Model Dev.* 19:1027–1054 (2026). doi:10.5194/gmd-19-1027-2026; arXiv:2507.00761
45. Shaddy et al. Probabilistic Forecasting of Localized Wildfire Spread Based on Conditional Flow Matching. arXiv:2603.26975 (2026)
46. Lin et al. Focal Loss. ICCV 2017. arXiv:1708.02002
47. Salehi et al. Tversky Loss. MLMI 2017, LNCS pp. 379–387. doi:10.1007/978-3-319-67389-9_44
48. Abraham, Khan. Focal Tversky Loss. ISBI 2019, pp. 683–687. doi:10.1109/ISBI.2019.8759329
49. Sudre et al. Generalised Dice Overlap. arXiv:1707.03237 (2017; DLMIA venue UNVERIFIED)
50. Berman et al. Lovász-Softmax. CVPR 2018. arXiv:1705.08790
51. Kervadec et al. Boundary Loss. MIDL 2019 (PMLR 102); *MedIA* 67:101851 (2021). doi:10.1016/j.media.2020.101851
52. MONAI `monai/losses/dice.py` (`DiceLoss`, `DiceCELoss`), tags 1.1.0–1.5.0 and dev. https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py
53. PyTorch `CrossEntropyLoss` docs (v2.14). https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
54. Tilborghs et al. The Dice Loss in the Context of Missing or Empty Labels. MICCAI 2022, LNCS pp. 527–537. doi:10.1007/978-3-031-16443-9_51
55. Eelbode et al. Optimization for Medical Image Segmentation. *IEEE TMI* 39:3679–3690 (2020). doi:10.1109/TMI.2020.3002417
56. scikit-learn `f1_score` docs. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
57. Reinke et al. Understanding metric-related pitfalls in image analysis validation. *Nature Methods* (2024). doi:10.1038/s41592-023-02150-0
58. Maier-Hein et al. Metrics Reloaded. *Nature Methods* (2024). doi:10.1038/s41592-023-02151-z

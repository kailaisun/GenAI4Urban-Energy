# GenAI4Urban-Energy

## Introduction

The repository contains the code implementation of the paper: SENSE: Satellite-based ENergy Synthesis for Sustainable Environment via Satellite Imagery.


## Installation

Download or clone the repository.

```shell
git clone https://github.com/kailaisun/GenAI4Urban-Energy.git
cd GenAI4Urban-Energy
```

We recommend using Conda ([Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)) for installation. 

### Environment Installation 

Please refer to the [UrbanControlnet](https://github.com/kailaisun/UrbanControlNet).

And install [segmentation_models](https://github.com/qubvel-org/segmentation_models.pytorch).

## Dataset Preparation

The dataset is built from publicly available global sources:
- **Urban boundaries** — [GHS Urban Centre Database (2023)](https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php), covering 500 metropolitan areas with 400 m × 400 m grids.
- **Satellite imagery** — [Mapbox Static Tiles API](https://docs.mapbox.com/api/maps/static-tiles/).
- **Population and building data** — GHSL P2023A (2020): [GHS-BUILT-S](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S), [GHS-BUILT-V](https://human-settlement.emergency.copernicus.eu/ghs_buV2023.php), [GHS-POP](https://human-settlement.emergency.copernicus.eu/ghs_pop2019.php).
- **Environmental constraints** — [OpenStreetMap](https://www.openstreetmap.org), including major roads, water bodies, and railways.


#### Dataset Download

Download [MUSE](https://huggingface.co/datasets/skl24/MUSE).

We provide an example for create energy map:
```shell
create_building_energy_use_image.py
```


#### Dataset preparation
```shell
python 5cities-merge_data_preprocess-all.py
```

## GenAI Model Training

Please refer to the [UrbanControlnet](https://github.com/kailaisun/UrbanControlNet).

#### Finetuning decoder

```shell
python E_decodere_train.py
python H_decoder_train.py
```

## Downstream Energy Prediction 

Training:
```shell
python segformer-image2energy_same_train-real.py
```

Testing:
```shell
python segformer-energy-performance.py
```


## Model evaluation

For computing metric (e.g., FID, SSIM, FSIM, PSNR, etc.), please see our another repo: [Evaluation-Metrics](https://github.com/T5-AI/Evaluation-Metrics)

## Acknowledgements

This research is supported by the National Research Foundation (NRF), Prime Minister’s Office under its Campus for Research Excellence and Technological Enterprise (CREATE) programme.
The Mens, Manus, and Machina (M3S) is an interdisciplinary research group (IRG) of the Massachusetts Institute of Technology and the Singapore MIT Alliance for Research and Technology (SMART) centre.

## Citation



## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

## Contact Us

If you have other questions❓, please contact us in time 👬


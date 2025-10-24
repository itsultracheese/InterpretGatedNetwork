# A hybrid of shapelet logic and neural networks with confidence-gating for accuracy and interpretability

Shapelets are short sequences with certain pattern which were originally defined as discriminative time-series subsequences to distinguish different categories

<img width="711" height="318" alt="image" src="https://github.com/user-attachments/assets/0d223e08-b8bc-4cb3-aa20-9aa24aa6bdfd" />

**Interpretability Gated Networks (InterpGN)** is a hybrid framework for time-series classification that combines:
- an **interpretable shapelet-based expert** (Shapelet Bottleneck Model, SBM),
- a **deep neural network backbone** (e.g., FCN),
- and a **novel gating mechanism** that dynamically decides whether human-understandable shapelet features are sufficient or if deeper representation learning is needed.

<img width="698" height="345" alt="image" src="https://github.com/user-attachments/assets/5eb88d84-d6c8-4547-99cd-58b18bcb328d" />

The goal of this project is to try to enhance the work of InterpGN for biomedical (e.g. EEG) data

---

## Project Structure

```
├── exp/ # Auxiliary experiment scripts (legacy or task-specific) 
├── layers/ # Core layers like shapelet extraction, distance computation, gating 
├── models/ # Model definitions like SBM, InterpGN, and base DNNs
├── notebook/ # Jupyter notebooks for result analysis and interpretability
├── reproduce/ # Reproduction scripts for UEA, EEG, and other benchmarks 
├── result/ # Experiment outputs: CSV logs
│ └── UEA/ # Precomputed results from the paper (for reference) 
├── utils/ # Utilities: data loaders, metrics, logging, reproducibility helpers 
├── run.py # Main training and evaluation script 
├── requirements.txt # Python dependencies 
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 2. Prepare Datasets

Download datasets and place them in `./data/`:

We used following EEG datasets from the multivariate UEA (but you can use any from UEA & UCR Time Series Classification Archive):

1) **`FingerMovements`** has 316 samples in train and 100 samples in test split. The data was recorded using 28 channels, it has 50 timepoints and was downsampled to 100Hz

2) **`SelfRegulationSCP1`** has 268 samples in train and 293 samples in test split. The data was recorded using 6 channels, it has 896 timepoints and sampling rate of 256Hz
   
3) **`SelfRegulationSCP2`** has 200 samples in train and 180 samples in test split. The data was recorded using 7 channels, it has 1152 timepoints and sampling rate of 256Hz.

### 3. Run Experiments

To reproduce our EEG experiments:
```bash
bash ./reproduce/run_eeg.sh
```
- Results (CSV files with test accuracy, etc.) are saved to ./result/.
- Model checkpoints (if enabled) go to ./checkpoints/.

---

## Custom Experiments

The main entry point is run.py. Most configurations can be controlled via command-line arguments or by editing the bash scripts (e.g., reproduce/run_eeg.sh).

| Parameter               | Description                                      | Example Values                                                                 |
|-------------------------|--------------------------------------------------|--------------------------------------------------------------------------------|
| `--model`               | Model architecture to use                        | `SBM`, `InterpGN`                                                              |
| `--dnn_type`            | Backbone deep neural network                     | `FCN`, `ResNet`                                                                |
| `--num_shapelet`        | Number of shapelets in the interpretable expert  | `10`, `20`, `50`                                                               |
| `--lambda_div`          | Diversity regularization strength                | `0.01`, `0.1`, `1.0`                                                           |
| `--lambda_reg`          | Shapelet regularization strength                 | `0.01`, `0.1`, `1.0`                                                           |
| `--pool`                | Pooling function over shapelet distances         | `max`, `lse`                                                                   |
| `--distance_func`       | Distance metric between time series and shapelets| `default`, `cosine`, `manhattan`, `pearson`, `mse`, `eucledian`, `chebyshev`   |
| `--pool_tau`            | Temperature parameter for `lse` pooling          | `0.5`, `1`, `5`, `10`, `15`, `20`, `25`                                        |
| `--learnable_tau`       | Whether `tau` is trainable (boolean flag)        | *add flag to enable*                                                           |
| `--seed`                | Random seed for reproducibility                  | `0`, `42`, `1234`, `2023`, `8237`                                              |
| `--dataset`             | Name of dataset (must exist in `./data/`)        | `SelfRegulationSCP2`, `FingerMovements`, `FaceDetection`                       |

---

### Next Steps
Our experiments highlight the need for the more rigorous analysis on the approaches to determine the optimal number and lengths of the shapelets.
The experiments demonstrated that the test accuracy on the three datasets benefitted from utilizing sampling frequency to induce the lengths of the shapelets, thus, it might be useful to test this hypothesis on the other datasets as well.
Since our experiments demonstrated that the learned shapelets are more useful for the classification task, it might make sense to try to utilize the similar approach for learning the shapelets for the time series representation in the transformer models such as ShapeFormer, or even the foundation models for EEG data.

### Logging

Here is the link to our [Neptune.ai](https://app.neptune.ai/o/gribanovds/org/interp-gn/runs/compare?viewId=standard-view&detailsTab=charts&dash=charts&compare=EwFiA)

## Amina's weird experiments

All the notebooks are in the `experiments` folder.

To use the pre-discovered shapelets, download them from the https://disk.yandex.ru/d/iOaiDiW2zWa9MQ and put into `store` folder in the project directory.

To reproduce the shapelet discovery process:

1. Clone https://github.com/xuanmay2701/shapeformer/tree/main
2. Install their requirements `pip install -r requirements.txt`
3. Change the 82 line in `cpu_main.py` to `shapelet_discovery.set_window_size(int(args.window_size))`
4. To the end of the `cpu_main.py` append:
```
            sc_path = "store/" + problem + "_sd2.pkl"
            file = open(sc_path, 'wb')
            pickle.dump(shapelet_discovery, file)
            file.close()
```
5. Run `python cpu_main.py --data_path [data path] --window_size [window size]`, the window size is 100 for the SelfRegulation datasets and 20 for the FingerMovements.

The scripts for ablation study on K is in `reproduce/run_k_grid.sh`, the scripts for learning the shapelets of lengths based on the sampling frequency are in `reproduce/run_new_lengths_fm.sh`, `reproduce/run_new_lengths_scp1.sh`, and `reproduce/run_new_lengths_scp2.sh`

---

### References
- Yunshi Wen, Tengfei Ma, Ronny Luss, Debarun Bhattacharjya, Achille Fokoue, Anak Agung Julius. Shedding Light on Time Series Classification using Interpretability Gated Networks
- Xuan-May Le, Ling Luo, Uwe Aickelin, and Minh-Tuan Tran. 2024. ShapeFormer: Shapelet Transformer for Multivariate Time Series Classification. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24). Association for Computing Machinery, New York, NY, USA, 1484–1494. https://doi.org/10.1145/3637528.3671862
- Carl H. Lubba, Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher, and Nick S. Jones. 2019. Catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis. Data Min. Knowl. Discov. 33, 6 (Nov 2019), 1821–1852. https://doi.org/10.1007/s10618-019-00647-x

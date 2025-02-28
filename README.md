# Interpretability Gated networks: A mixture-of-experts approach for interpretable time-series classification using shapelets.

<div align="center">
    <h3><a href="https://openreview.net/forum?id=n34taxF0TC">Shedding Light on Time Series Classification using Interpretability Gated Networks</a></h3>
    <h4>ICLR 2025</h4>
    <h4>Yunshi Wen, Tengfei Ma, Ronny Luss, Debarun Bhattacharjya, Achille Fokoue, Anak Agung Julius</h4>
</div>

Interpretability Gated Networks (InterpGN) is a hybrid framework for time-series classification that combines the strengths of a shapelet-based interpretable model (Shapelet Bottleneck Model) and deep neural networks. It uses a novel gating function - based on the confidence of an interpretable expert employing shapelets - to decide when human-understandable features suffice or when deeper analysis is needed. 


## Usage

### Environment Setup:

Package versions: Python 3.11. PyTorch 2.4.0 (should also be compatible with later versions).

### Data Preparation
Download the datasets and put them in the `./data` folder. The datasets are available at:
- [UEA Multivariate Time-Series Classification](https://timeseriesclassification.com/)
- [Monash Time-Series Extrinsic Regression](http://tseregression.org/)

### Experiments
For classification experiments, run 
```bash
bash ./reproduce/run_uea.sh
```

For regression experiments, run 
```bash
bash ./reproduce/run_regression.sh
```

The results for each run are saved in the `./result` folder, and the trained model weights are saved in the `./checkpoints` folder.

### Result Analysis, Visualization, and Interpretability

We provide notebooks as examples for analyzing the results and intepretable predictions learned by the Shapelet Bottleneck Model (SBM) and the Interpretability Gated Network (InterpGN).

After running the experiments, use `./notebook/benchmarks.ipynb` to collect the results for benchmarking. Examples of visualizing the shapelets, explanations, and faithfulness are provided in `./notebook/visualization.ipynb`.

We also provide full experiments results from **our experiments and reproductions** in `./result/UEA`.


## Citation
```
@inproceedings{
    wen2025shedding,
    title={Shedding Light on Time Series Classification using Interpretability Gated Networks},
    author={Yunshi Wen and Tengfei Ma and Ronny Luss and Debarun Bhattacharjya and Achille Fokoue and Anak Agung Julius},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```

## Acknowledgement

We thank the research community for the great work on time-series analysis, the open-source codebase, and the datasets, including but not limited to:
- The codebase is developed based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
- The UEA and UCR teams for collecting and sharing the [time-series classification datasets](https://timeseriesclassification.com/).
- The Monash team for collecting and sharing the [time-series extrinsic regression datasets](http://tseregression.org/).
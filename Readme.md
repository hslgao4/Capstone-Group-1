# Advancing Temporal Forecasting: A Comparative Analysis of Conventional Paradigms and Deep Learning Architectures on Publicly Accessible Datasets
## Table of Content
- [Project Overview](#project-overview)
- [Repo structure](#repo-structure)
- [Results and finding](#results-and-finding)

  
## Project Overview
The primary objective of this study is to evaluate and compare the performance of:
- **Classical time series models**: Autoregressive(**AR**), Moving Average (**MA**), Autoregressive Moving Average (**ARMA**), and Autoregressive Integrated Moving Average (**ARIMA**)
- **Advanced modern techniques**: Long Short-Term Memory (**LSTM**), Bidirectional LSTM (**Bi-LSTM)**, Sequence-to-Sequence architectures (**Seq2Seq**)
- **State-of-the-art**: **Transformers**

On three public datasets:

- [Weather Station Beutenberg Dataset](https://www.kaggle.com/datasets/mnassrib/jena-weather-dataset)
- [Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
- [Air Pollution Forecasting Dataset](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate)


## Repo Structure
* **code** 
  * **component**: all the defined function under [utils.py](./code/component/utils.py) & defined classes.
  * **data**: raw data & processed data used in modeling
  * **main code**
    * **main:** main script to run the whole pipeline
    * **test:** experiment test files
* **full_report:** [full report](./full_report/full_report.pdf) & latex folder
* **presentation:** the presentation slides
* **research_paper:** [academic research paper](./research_paper/research_paper.pdf) & latex folder


## Results
* **Classical models:**
![classical_model_results](research_paper/classical.pdf)

* **Modern techniques:**
![lstm_results](research_paper/modern.pdf)

* **State-of-the -art:**
![transformer_results](research_paper/trans.pdf)






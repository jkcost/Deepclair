# DeepClair: Utilizing Market Forecasts for Effective Portfolio Selection

Utilizing market forecasts is pivotal in optimizing portfolio selection strategies. We introduce DEEP CLAIR, a novel framework for portfolio selection.
DEEPCLAIR leverages a transformer-based time
series forecasting model to predict market trends,
facilitating more informed and adaptable portfo-
lio decisions. To integrate the forecasting model
lio selection framework, we introduced a two-step
strategy: first, pre-training the time-series model
on market data, followed by fine-tuning the port
folio selection architecture using this model. Ad
ditionally, we investigated the optimization tech
nique, Low-Rank Adaptation (LoRA), to enhance
the pre-trained forecasting model for fine-tuning in
investment scenarios. This work bridges market
forecasting and portfolio selection, facilitating the
advancement of investment strategies.

## Content

1. [Requirements](#Requirements)
2. [Data Preparing]()
3. [Training](Training)
5. [Acknowledgement](Acknowledgement)



## Requirements

- Python 3.8 or higher.
- Pytorch == 2.0.1
- Pandas >= 1.5.3
- Numpy >= 1.25.0
- ...

## Data Preparing

The data is not attached separately, but the overall usage is as follows.

The following files are needed:

|                    File_name                     |                  shape                      |                  description                   |
| :----------------------------------------------: | :--------------------------------------:    | :--------------------------------------------: |                                         
|                 stocks_data.npy                  |       [num_stocks, num_days, Close_price]   |       the inputs for asset scoring unit        |
|                 market_data.npy                  |       [num_days, close_price of num_stocks] |     the inputs for marketing scoring unit (Pretrained Time forecast model)      |
|                     ror.npy                      |       [num_stocks, num_days]                | rate of return file for calculating the return |
| relation_file (e.g. industry_classification.npy) |       [num_stocks, num_stocks]              |     the relation matrix used in GCN layer      |






These files should be placed in the ./data/INDEX_NAME folder, e.g. ./data/DJIA/stocks_data.npy

## Training

Initially, the training of the forecast model is required. This process is initiated by executing the run.py script, which is located within the forecast folder, using Python. Once the model has been successfully trained, it must be stored within the ./models directory of Deepclair. Subsequent to this placement, further training of the model is achievable by executing the run.py script within Deepclair, supplemented with the --c hyper.json argument. This step utilizes the specified hyperparameters defined in hyper.json for the training process.

Some of the available arguments are:

| Argument          | Description                                                | Default                     | Type  |
| ----------------- | ---------------------------------------------------------- | --------------------------- | ----- |
| `--config`        | Deafult configuration file                                 | hyper.json                  | str   |
| `--window_len`    | Input window size                                          | 5 (dayss)                  | int   |
| `--market`        | Stock market                                               | IXIC                       | str   |
| `--train_type`             | Either including or excluding the training of LoRa and pretrain models |      Lora           | str   |
| `--batch_size`    | Batch size number                                          | 128                        | Int   |
| `--lr`            | learning rate                                              | 2e-5                       | float |
| `-model`         | Which model should be selected for time series forecast    |  Fedformer                 |  str   |


## Acknowledgement

This project would not have been finished without using the codes or files from the following open source projects:


- this project is inspired by [DeepTrader](https://github.com/CMACH508/DeepTrader)


## Reference


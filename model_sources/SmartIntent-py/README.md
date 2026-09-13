# SmartIntentNN V2

Leverage TensorFlow with Python for model training, replacing TensorFlow.js.

CuDNN support is insufficient in `tfjs-node-gpu`.

## JS Version

The following JS version is now used for web tool API and ever used for SmartIntentNN V1:

<https://github.com/web3se-lab/web3-sekit>

## Installation

```sh
pip install -r requirements.txt
```

## Dataset

- **Training:** 16,000 (from ID: 1) smart contracts
- **Evaluation:** 10,000 (from ID: 20000) smart contracts

## Execution

**Run in python for Model Training and Evaluation:**

```bash
# Summarize model
python3 ./smartbert_bilstm.py summary

# Train model
python3 ./smartbert_bilstm.py train

# For 2nd-round balance training
python3 ./smartbert_bilstm.py train2

# Evaluate model
python3 ./smartbert_bilstm.py evaluate
```


**Convert to TFJS model:**

First change env variables in `convert_tfjs.sh`: `MODEL`, `DIR`, `JS`.

Then run convert:

```bash
./convert_tfjs.sh
```

**Run in ipynb for GPT Experiment:**

- **Important**: You first need to fill in the following variables in Step 1.
```bash
# You need to define your chat API and request headers here
# chat_api_url = 
# chat_headers = {}
```
- **[GPT_Experiment.ipynb](./GPT_Experiment.ipynb)**: This file contains the code for GPT experiment.
- **[prompt.md](./prompt.md)**: The prompt given to the GPT.


## Experiments

### Setup

**Pre-trained model** is **[SmartBERT](https://github.com/web3se-lab/SmartBERT)**, used for code embedding, trained on **16000** smart contracts.

- **Loss:** BinaryFocalCrossentropy
- **LSTM/BiLSM Units:** 128/256
- **Dropout:** 0.5
- **Mask Pad:** 0.0
- **Seq Length:** 256
- **Embedding Dim:** 768

**Sequential complete data training**

- **Epoch:** 100
- **Batch:** 80
- **Batch Size:** 200
- **Total:** 16000
- **Start ID:** 1
- **End ID:** 17197
- **Learning Rate:** 0.001

**random balanced-sample completion training**

- **Epoch:** 100
- **Batch:** 20
- **Batch Size:** 100 Random data
- **Total:** 2000
- **Start ID:** 1
- **End ID:** 17197
- **Learning Rate:** 0.0001

### SmartIntentNN2 Complete Training Turn

The results are:

| Category       | Accuracy | Precision  | Recall     | F1 Score   |
| -------------- | -------- | ---------- | ---------- | ---------- |
| fee            | 0.9394   | 0.9288248  | 0.9279471  | 0.9283857  |
| disableTrading | 0.9643   | 0.8526682  | 0.761658   | 0.8045977  |
| blacklist      | 0.9689   | 0.84528834 | 0.74937654 | 0.7944481  |
| reflect        | 0.9887   | 0.98382837 | 0.9789819  | 0.9813991  |
| maxTX          | 0.9623   | 0.9480992  | 0.9287565  | 0.9383281  |
| mint           | 0.9513   | 0.82222223 | 0.79753655 | 0.8096912  |
| honeypot       | 0.9933   | 0.84615386 | 0.49107143 | 0.62146884 |
| reward         | 0.9937   | 0.96196514 | 0.9396285  | 0.95066553 |
| rebase         | 0.9915   | 0.5294118  | 0.10465116 | 0.17475724 |
| maxSell        | 0.9971   | 1.0        | 0.06451613 | 0.12121211 |
|                |          |            |            |            |
| Average        | 0.975050 | 0.87184619 | 0.67441238 | 0.71249536 |
| Total          | 0.97505  | 0.9269809  | 0.89620465 | 0.911333   |

### SmartIntentNN2 Balanced Sample Training Turn

The results are:

| Category       | Accuracy | Precision  | Recall     | F1 Score   |
| -------------- | -------- | ---------- | ---------- | ---------- |
| fee            | 0.9452   | 0.91173184 | 0.96385545 | 0.9370693  |
| disableTrading | 0.9753   | 0.8953745  | 0.84248704 | 0.8681259  |
| blacklist      | 0.9813   | 0.8579744  | 0.91895264 | 0.8874172  |
| reflect        | 0.993    | 0.9806139  | 0.9967159  | 0.9885993  |
| maxTX          | 0.975    | 0.9610137  | 0.95790154 | 0.959455   |
| mint           | 0.9393   | 0.72322583 | 0.86297154 | 0.7869428  |
| honeypot       | 0.991    | 0.5833333  | 0.6875     | 0.6311475  |
| reward         | 0.9947   | 0.94452775 | 0.9752322  | 0.95963436 |
| rebase         | 0.9958   | 0.7        | 0.89534885 | 0.78571427 |
| maxSell        | 0.9987   | 0.71428573 | 0.9677419  | 0.8219178  |
|                |          |            |            |            |
| Average        | 0.97893  | 0.82720809 | 0.9068707  | 0.86260234 |
| Total          | 0.97893  | 0.9089568  | 0.947648   | 0.92789924 |


## Ablation Tests

### RoBERTa Complete Training Turn

The following shows the results of an ablation test using **RoBERTa** as the pre-trained model on a multi-classification task for 10 types of risky intents in smart contracts (Round 1, Complete Training Turn):

| Category           | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| **fee**            | 0.9182   | 0.8754    | 0.9407 | 0.9069   |
| **disableTrading** | 0.9533   | 0.7862    | 0.7088 | 0.7455   |
| **blacklist**      | 0.9503   | 0.6771    | 0.7269 | 0.7011   |
| **reflect**        | 0.9884   | 0.9832    | 0.9787 | 0.9809   |
| **maxTX**          | 0.9575   | 0.9202    | 0.9443 | 0.9321   |
| **mint**           | 0.9376   | 0.7487    | 0.7821 | 0.7651   |
| **honeypot**       | 0.9928   | 0.8226    | 0.4554 | 0.5862   |
| **reward**         | 0.9925   | 0.9539    | 0.9288 | 0.9412   |
| **rebase**         | 0.9914   | 0.5000    | 0.0814 | 0.1400   |
| **maxSell**        | 0.9966   | 0.0000    | 0.0000 | 0.0000   |
|                    |          |            |            |            |
| **Average**     | 0.9679   | 0.7267    | 0.6547  | 0.6699   |
| **Total**       | 0.9679   | 0.8813    | 0.8960  | 0.8886   |


### RoBERTa Balanced Sample Training Turn

The following shows the results of an **ablation test** using **RoBERTa** as the pre-trained model on a multi-classification task for 10 types of risky intents in smart contracts (Round 2, Balance Sample Training Turn):

| Category           | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| **fee**            | 0.9430   | 0.9182    | 0.9499 | 0.9338   |
| **disableTrading** | 0.9555   | 0.7802    | 0.7503 | 0.7649   |
| **blacklist**      | 0.9558   | 0.7055    | 0.7706 | 0.7366   |
| **reflect**        | 0.9922   | 0.9790    | 0.9957 | 0.9873   |
| **maxTX**          | 0.9684   | 0.9483    | 0.9495 | 0.9489   |
| **mint**           | 0.9114   | 0.6125    | 0.8653 | 0.7173   |
| **honeypot**       | 0.9887   | 0.4968    | 0.6875 | 0.5768   |
| **reward**         | 0.9853   | 0.8219    | 0.9861 | 0.8966   |
| **rebase**         | 0.9944   | 0.6293    | 0.8488 | 0.7228   |
| **maxSell**        | 0.9979   | 0.6000    | 0.9677 | 0.7407   |
|                    |          |            |            |            |
| **Average**     | 0.9693   | 0.7492    | 0.8771  | 0.8026   |
| **Total**       | 0.9693   | 0.8670    | 0.9274  | 0.8962   |


### CodeBERT Complete Training Turn

The following shows the results of an ablation test using **CodeBERT** as the pre-trained model on a multi-classification task for 10 types of risky intents in smart contracts (Round 1, Complete Training Turn):

| Category           | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| **fee**            | 0.9164   | 0.8787    | 0.9310 | 0.9041   |
| **disableTrading** | 0.9580   | 0.7985    | 0.7554 | 0.7764   |
| **blacklist**      | 0.9629   | 0.8137    | 0.6970 | 0.7508   |
| **reflect**        | 0.9887   | 0.9822    | 0.9806 | 0.9814   |
| **maxTX**          | 0.9568   | 0.9444    | 0.9139 | 0.9289   |
| **mint**           | 0.9427   | 0.8066    | 0.7352 | 0.7692   |
| **honeypot**       | 0.9931   | 0.8413    | 0.4732 | 0.6057   |
| **reward**         | 0.9901   | 0.9176    | 0.9303 | 0.9239   |
| **rebase**         | 0.9918   | 0.6111    | 0.1279 | 0.2115   |
| **maxSell**        | 0.9965   | 0.0000    | 0.0000 | 0.0000   |
|                    |          |            |            |            |
| **Average**     | 0.9697   | 0.7594    | 0.6545  | 0.6852   |
| **Total**       | 0.9697   | 0.9017    | 0.8847  | 0.8931   |


### CodeBERT Balanced Sample Training Turn

The following shows the results of an **ablation test** using **CodeBERT** as the pre-trained model on a multi-classification task for 10 types of risky intents in smart contracts (Balance Sample Training Turn):

| Category           | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| **fee**            | 0.9266   | 0.8992    | 0.9310 | 0.9148   |
| **disableTrading** | 0.9657   | 0.8180    | 0.8290 | 0.8235   |
| **blacklist**      | 0.9722   | 0.8119    | 0.8504 | 0.8307   |
| **reflect**        | 0.9865   | 0.9780    | 0.9777 | 0.9778   |
| **maxTX**          | 0.9605   | 0.9101    | 0.9676 | 0.9380   |
| **mint**           | 0.8993   | 0.5726    | 0.8861 | 0.6957   |
| **honeypot**       | 0.9883   | 0.4847    | 0.7054 | 0.5745   |
| **reward**         | 0.9906   | 0.8954    | 0.9675 | 0.9301   |
| **rebase**         | 0.9923   | 0.5306    | 0.9070 | 0.6695   |
| **maxSell**        | 0.9899   | 0.2348    | 1.0000 | 0.3804   |
|                    |          |            |            |            |
| **Average**     | 0.9672   | 0.7135    | 0.9022  | 0.7735   |
| **Total**       | 0.9672   | 0.8516    | 0.9332  | 0.8906   |

## Citation

```tex
@article{huang2025smart,
  title={Smart Contract Intent Detection with Pre-trained Programming Language Model},
  author={Huang, Youwei and Li, Jianwen and Fang, Sen and Li, Yao and Yang, Peng and Hu, Bin},
  journal={arXiv preprint arXiv:2508.20086},
  year={2025}
}
```

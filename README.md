# Deep & Graph Neural Networks 

## Notebooks

### [`TabularData.ipynb`](TabularData.ipynb) - customer churn

Kaggle Playground Series S6E3. Builds a sklearn `ColumnTransformer` pipeline (ordinal + one-hot + scaling), then trains and compares Random Forest, XGBoost, PyTorch TabNet, and Keras MLPs. Neural models tuned with Optuna; XGBoost explained with SHAP. Generates test predictions for submission.

### [`SentimentAnalysis.ipynb`](SentimentAnalysis.ipynb) - tweet sentiment

TweetEval 3-class labels (negative / neutral / positive). EDA, tokenization, and PyTorch RNN, Bi-RNN, and 1D CNN models with Optuna hyperparameter search, compared against fine-tuned DistilBERT. Confusion matrices and Captum token attributions to compare how each architecture handles difficult phrases.

### [`OxfordPetsTransferLearning.ipynb`](OxfordPetsTransferLearning.ipynb) - pet breed classification

Oxford-IIIT Pet dataset (37 breeds). **TensorFlow / Keras** transfer learning with MobileNetV2, ResNet50, and EfficientNetB0: frozen backbone, augmentation (lite vs strong), partial fine-tuning, confusion matrix, Grad-CAM on errors.

---

## Data

| Notebook | Source |
|----------|--------|
| Tabular | [Playground S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
| Sentiment | Hugging Face `tweet_eval`  |
| Pets | `tensorflow_datasets` Oxford-IIIT Pet  |



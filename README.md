# Identifying Pregnancy in Clinical Notes

Identifying medical conditions and events in clinical notes can add business value to hospitals and other organizations with clinical text data.

This project used Python, the HuggingFace transformers library, and Gradio to fine-tune a large language model to identify pregnancy in clinical notes, and publish an app interface for testing use cases.

Key steps in this project:

- Used the Kaggle API to download a publicly available dataset of clincial notes
- Augmented the dataset with synthetic data, and used upsampling to address class imbalance
- Used the transformers module to fine-tune an existing large language model to the classification task
- Evaluated model performance on a test set

# Identifying Pregnancy in Clinical Notes

Identifying medical conditions in clinical notes can add business value to health care organizations. Pregnancy is one example of a condition that can be identified.

This project used Python, the HuggingFace transformers library, and Gradio to fine-tune a large language model to identify pregnancy in clinical notes and publish an app interface.

Key steps in this project:

- Used the `kaggle` API to download a publicly available dataset of clincial notes
- Augmented the dataset with synthetic data, and used upsampling to address class imbalance
- Used the HuggingFace `transformers` module to fine-tune an existing large language model to the classification task
- Evaluated model performance on a test set
- Used the `gradio` module to develop an app to enable users to test use cases
- Published the [fine-tuned model](https://huggingface.co/elliealbertson/identifying_pregnancy_clinical_notes) and [Gradio app](https://huggingface.co/spaces/elliealbertson/identifying_pregnancy_clinical_notes) on HuggingFace

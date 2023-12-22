
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import gradio as gr

model_path = 'elliealbertson/identifying_pregnancy_clinical_notes'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict(text):

    inputs = tokenizer(text, return_tensors="pt")
    num_tokens = inputs['input_ids'].size(1)

    if num_tokens <= 512:

        outputs = model(**inputs)

        predicted_class_id = torch.argmax(outputs.logits).item()

        probability_of_predicted_class = round(torch.nn.functional.softmax(outputs.logits, dim=1)[0, predicted_class_id].item(),2)

        if (predicted_class_id == 0) & (probability_of_predicted_class >= 0.5):
            predicted_class_label = "No, the note does not discuss the patient's pregnancy based on the model's assessment."
        elif (predicted_class_id == 1) & (probability_of_predicted_class >= 0.5):
            predicted_class_label = "Yes, the note discusses the patient's pregnancy based on the model's assessment."
        else:
            predicted_class_label = "The model was unable to determine with high certainty whether or not the note discusses the patient's pregnancy. Please provide additional text or a different note."
        
        return predicted_class_label
    
    else:
        
        error_message = 'Unfortunately the model is limited in how much text it can process at once. Please enter a shorter note.'

        return error_message

with gr.Blocks() as interface:
    gr.HTML("<div style='text-align:center;'><div><h1>Identifying Pregnancy in Clinical Notes</h1></div>")
    gr.HTML("<p align='center'>Use this app to classify a clinical note as discussing or not discussing the patient's pregnancy.</p>")
    with gr.Row():
        with gr.Column():
            inputs = gr.Textbox(label='Input a clinical note here:', lines=4)
            button = gr.Button('Assess Note')
            gr.Examples(['The patient is pregnant.', 'She has high cholesterol and hypertension.', 'Normal vaginal delivery.', 'Fetus development normal.', 'Presented with nausea.', 'Broken arm and leg.'], inputs)
        with gr.Column():
            outputs=gr.Textbox(label="Does the note discuss the patient's pregnancy?", lines=4)
            button.click(fn=predict, inputs=inputs, outputs=outputs, queue=False)
    gr.HTML("<p align='center'>Model fine-tuned from <a href='https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT' target='_blank'> Bio+ClinicalBERT </a>.</p>")

interface.launch()
from flask import Flask, request, jsonify
import os
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the path for saving uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path for models
MODEL_FOLDER = 'deep-text-recognition-benchmark'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Define the models
models = {
    'None-ResNet-None-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
    'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
    'None-VGG-None-CTC.pth': 'https://drive.google.com/uc?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
    'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/uc?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
    'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/uc?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
    'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
}

# Download models if they don't already exist
for k, v in models.items():
    model_path = os.path.join(MODEL_FOLDER, k)
    if not os.path.exists(model_path):
        os.system(f'gdown -O {model_path} "{v}"')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Assuming the language code is part of the form data
        language_code = request.form.get('language', 'en')  # Default to English if not provided

        # Here you can add conditions to select the model based on language_code
        model_path = os.path.join(MODEL_FOLDER, "TPS-ResNet-BiLSTM-Attn.pth")

        # Run the model and extract results
        result = os.popen(f'CUDA_VISIBLE_DEVICES=0 python3 {MODEL_FOLDER}/demo.py \
                            --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
                            --image_folder {UPLOAD_FOLDER}/ --saved_model {model_path}').read()

        # Delete the uploaded file after processing
        os.remove(file_path)

        # Return only the recognized text without the headers
        return jsonify({'recognized_text': extract_predicted_labels(result)})

# Function to extract predicted labels from the recognized text, excluding header line
def extract_predicted_labels(recognized_text):
    # Split the text by lines after the header
    lines = recognized_text.split('\n')[7:]  # Adjust the index to skip header lines
    # Initialize an empty list to hold the predicted labels
    predicted_labels = []
    # Process each line for predicted labels
    for line in lines:
        if line.strip():  # Check if the line is not empty
            # Extracting only the part after 'uploads/' and before the first tab for filename
            # and the part between the first and second tab for predicted label
            parts = line.split('\t')
            if len(parts) >= 2:  # Check if line contains enough parts
                filename = parts[0].strip().split('/')[-1]  # Get only the file name
                predicted_label = parts[1].strip()
                predicted_labels.append((filename, predicted_label))
    return predicted_labels

if __name__ == '__main__':
    app.run(debug=True)

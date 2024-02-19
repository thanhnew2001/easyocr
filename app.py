from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the path for saving uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the models
models = {
    'None-ResNet-None-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
    'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
    'None-VGG-None-CTC.pth': 'https://drive.google.com/uc?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
    'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/uc?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
    'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/uc?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
    'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
}

# Download models
for k, v in models.items():
    os.system(f'gdown -O {os.path.join("deep-text-recognition-benchmark", k)} "{v}"')

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
        language_code = request.form.get('language', 'eng')  # Default to English if not provided

        # Here you can add conditions to select the model based on language_code
        # For simplicity, we use one model for demonstration
        model_path = os.path.join("deep-text-recognition-benchmark", "TPS-ResNet-BiLSTM-Attn.pth")

        # Run the model
        result = os.popen(f'CUDA_VISIBLE_DEVICES=0 python3 deep-text-recognition-benchmark/demo.py \
                            --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
                            --image_folder {UPLOAD_FOLDER}/ --saved_model {model_path}').read()

        # Here, you should parse the output of the model to extract the recognized text
        # For simplicity, we just return the raw result
        return jsonify({'recognized_text': result})

if __name__ == '__main__':
    app.run(debug=True)

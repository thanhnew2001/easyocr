from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)

# Define the path for saving uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract image file names and their corresponding predicted labels
def extract_image_labels(recognized_text):
    # Initialize an empty list to hold the image file names and predicted labels
    image_labels = []
    # Regex pattern to extract file name and predicted label
    pattern = re.compile(r'(uploads\/[^\t]+)\t([^\t]+)')
    # Process each line
    for line in recognized_text.split('\n'):
        # Find matches using the regular expression
        matches = pattern.search(line)
        if matches:
            # Extract the image file name and predicted label
            image_file, predicted_label = matches.groups()
            # Add the pair to the list
            image_labels.append((image_file, predicted_label))
    return image_labels

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

        # Run your model here and get the recognized text
        # For example purposes, let's assume the recognized_text is returned from your model
        # recognized_text = your_model_function(file_path)  # Replace with actual model function call
        recognized_text = "Your recognized text here"  # Placeholder for actual recognized text

        # Extracting image file names and predicted labels
        image_labels = extract_image_labels(recognized_text)
        
        # After processing, delete the uploaded file to clean up
        os.remove(file_path)
        
        return jsonify({'image_labels': image_labels})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from resnet_model import get_resnet18  # Adjust the import path as necessary
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from fashion_mnist_model import FashionMNISTModel, preprocess_image_with_thresholding

app = Flask(__name__)

# Load model 1
fabric = get_resnet18(num_classes=27)  # Adjust num_classes as needed
checkpoint1 = torch.load('model/fabric.pth', map_location='cpu')
# Adjust keys for model 1
fabric_state_dict = {key.replace('module.', ''): value for key, value in checkpoint1['model'].items()}
fabric.load_state_dict(fabric_state_dict)
fabric.eval()

# Load model 2
fibre = get_resnet18(num_classes=33)  # Adjust num_classes as needed
checkpoint2 = torch.load('model/fiber.pth', map_location='cpu')
# Adjust keys for model 2
fibre_state_dict = {key.replace('module.', ''): value for key, value in checkpoint2['model'].items()}
fibre.load_state_dict(fibre_state_dict)
fibre.eval()

model_path = 'model/fashion_mnist_cnn_model.pth'  
model = FashionMNISTModel(1, 10, 10)  
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.route('/check-liveness', methods=['GET'])
def check_liveness():
    print('liveness check request')
    return jsonify({'status': 'alive'}), 200

@app.route('/fabric', methods=['POST'])
def predict_model1():
    print('fabric request')
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    print("got file")
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = fabric(image)
        probabilities, classes = torch.topk(torch.nn.functional.softmax(outputs, dim=1), 5)
        predictions = []
        for i in range(5):
            predictions.append({
                'class': classes[0][i].item(),
                'probability': probabilities[0][i].item()
            })

        return jsonify(predictions)


@app.route('/fiber', methods=['POST'])
def predict_model2():
    print('fiber request')
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    print("got file")
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = fibre(image)
        probabilities, classes = torch.topk(torch.nn.functional.softmax(outputs, dim=1), 5)
        predictions = []
        for i in range(5):
            predictions.append({
                'class': classes[0][i].item(),
                'probability': probabilities[0][i].item()
            })

        return jsonify(predictions)

@app.route('/fashion', methods=['POST'])
def predict_fashion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Assuming the preprocess_image_with_thresholding function returns the preprocessed image tensor
    _, image_tensor = preprocess_image_with_thresholding(file)
    
    # Predict the class of the fashion item
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = class_names[class_id]
        return jsonify({'class_name': class_name})



# if __name__ == '__main__':
#     app.run(debug=True)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')

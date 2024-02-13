from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from resnet_model import get_resnet18  # Adjust the import path as necessary

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/check-liveness', methods=['GET'])
def check_liveness():
    return jsonify({'status': 'alive'}), 200

@app.route('/fabric', methods=['POST'])
def predict_model1():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
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


# if __name__ == '__main__':
#     app.run(debug=True)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
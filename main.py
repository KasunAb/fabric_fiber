from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from resnet_model import get_resnet18  # Adjust the import path as necessary

app = Flask(__name__)

# Load model 1
fabric = get_resnet18(num_classes=27)  # Adjust num_classes as needed
checkpoint1 = torch.load('model/fabric.pth', map_location='cpu')
fabric.load_state_dict(checkpoint1['model'])
fabric.eval()

# Load model 2
fibre = get_resnet18(num_classes=27)  # Adjust num_classes as needed
checkpoint1 = torch.load('model/fiber.pth', map_location='cpu')
fibre.load_state_dict(checkpoint1['model'])
fibre.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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


if __name__ == '__main__':
    app.run(debug=True)


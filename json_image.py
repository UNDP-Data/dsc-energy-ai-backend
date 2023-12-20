import base64
import json
from PIL import Image
from io import BytesIO

def image_to_json(image_path):
    with open(image_path, 'rb') as file:
        image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        json_data = json.dumps({'image': encoded_image})
        return json_data
    
def json_to_image(json_data):
    decoded_data = json.loads(json_data)
    encoded_image = decoded_data['image']
    image_data = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(image_data))
    return image

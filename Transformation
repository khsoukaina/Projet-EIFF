import os
from PIL import Image, ImageEnhance
import random

input_folder = "C:/Users/skhalil/Pictures/train/fh/ok"
output_folder = "C:/Users/skhalil/Pictures/donn"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rotation_angles = [90, 180, 360]

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

       
        if image.mode != 'RGB':
            image = image.convert('RGB')

     
        angle = random.choice(rotation_angles)
        rotated_image = image.rotate(angle, expand=True)

       
        if rotated_image.mode != 'RGB':
            rotated_image = rotated_image.convert('RGB')

        enhancer = ImageEnhance.Color(rotated_image)
        color_factor = random.uniform(1.5, 1.8) 
        enhanced_image = enhancer.enhance(color_factor)

       
        if enhanced_image.mode != 'RGB':
            enhanced_image = enhanced_image.convert('RGB')
        
        
        output_image_path = os.path.join(output_folder, filename)
        enhanced_image.save(output_image_path)

print("Transformation terminée.")

from smartcrop import SmartCrop
from PIL import Image

def apply_smart_crop(image_path, output_size=(224, 224), save_path="output.jpg"):
    sc = SmartCrop()
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Compute best crop
    crop = sc.crop(image, output_size[0], output_size[1])
    top_crop = crop['top_crop']

    # Crop image using top_crop dimensions
    cropped = image.crop((
        top_crop['x'],
        top_crop['y'],
        top_crop['x'] + top_crop['width'],
        top_crop['y'] + top_crop['height']
    ))

    # Resize to output dimensions
    resized = cropped.resize(output_size, Image.LANCZOS)
    resized.save(save_path)
    return resized

# Example
if __name__ == "__main__":
    imgs = [f"output-{i}.png" for i in range(1,8)]
    for img in imgs:
        apply_smart_crop("./confidentiels/"+img, output_size=(224, 224), save_path="./confidentiels/smart_" + img)

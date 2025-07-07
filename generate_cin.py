from PIL import Image, ImageDraw, ImageFont

width, height = 200, 150


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


for letter in alphabet :
    image = Image.new('RGB', (width, height), color='white')

    draw = ImageDraw.Draw(image)
    text = f"A{letter}123456789"
    text_position = (25, (height/2) - 15)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    draw.text(text_position, text, fill='black', font=font)
    image.save("./cins/corrected_cin_" + letter + ".png")
    # image.show()

from PIL import Image


img = Image.open("12.png")

# 1. Grayscale
grayscale = img.convert("L")
grayscale.save("output_grayscale.png")

# 2. Чёрно-белое с дизерингом
bw_dithered = img.convert("1")
bw_dithered.save("output_bw_dithered.png")

# 3. Чёрно-белое без дизеринга
bw_no_dither = img.convert("L").point(lambda x: 255 if x > 128 else 0, mode="1")
bw_no_dither.save("output_bw_no_dither.png")
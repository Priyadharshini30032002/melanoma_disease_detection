# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk

# loaded_image = None
# image_path = None

# def load_image(label):
#     global loaded_image, image_path
#     file_path = filedialog.askopenfilename(
#         filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
#     )
#     if file_path:
#         image_path = file_path
#         loaded_image = Image.open(file_path)   # store PIL image
#         return loaded_image
#     return None
from tkinter import filedialog
from PIL import Image

loaded_image = None
image_path = None

def load_image(label):
    global loaded_image, image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        image_path = file_path
        loaded_image = Image.open(file_path)   # <-- store PIL image globally
        return loaded_image
    return None

import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

def load_image():
    global img, img_display
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_display = img.copy()
    display_image(img_display)

def load_template():
    global templ
    file_path = filedialog.askopenfilename()
    templ = cv2.imread(file_path, cv2.IMREAD_COLOR)

def matching_method():
    global img, templ, img_display
    method = method_var.get()
    result = cv2.matchTemplate(img, templ, method)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_loc = min_loc
    else:
        match_loc = max_loc

    cv2.rectangle(img_display, match_loc, (match_loc[0] + templ.shape[1], match_loc[1] + templ.shape[0]), (255,0,0), 2)
    display_image(img_display)

def display_image(img_to_show):
    b,g,r = cv2.split(img_to_show)
    img_adjusted = cv2.merge((r,g,b))
    im_pil = Image.fromarray(img_adjusted)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    label_img.config(image=imgtk)
    label_img.photo = imgtk

root = tk.Tk()
root.title("Template Matching")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

btn_load_image = tk.Button(frame, text="Load Image", command=load_image)
btn_load_image.pack(side=tk.LEFT, padx=10)

btn_load_template = tk.Button(frame, text="Load Template", command=load_template)
btn_load_template.pack(side=tk.LEFT, padx=10)

method_var = tk.IntVar(value=cv2.TM_CCOEFF)
methods = [
    ("SQDIFF", cv2.TM_SQDIFF),
    ("SQDIFF NORMED", cv2.TM_SQDIFF_NORMED),
    ("CCORR", cv2.TM_CCORR),
    ("CCORR NORMED", cv2.TM_CCORR_NORMED),
    ("COEFF", cv2.TM_CCOEFF),
    ("COEFF NORMED", cv2.TM_CCOEFF_NORMED)
]
for text, method in methods:
    tk.Radiobutton(frame, text=text, variable=method_var, value=method).pack(side=tk.LEFT)

btn_match = tk.Button(frame, text="Match", command=matching_method)
btn_match.pack(side=tk.LEFT, padx=10)

label_img = tk.Label(root)
label_img.pack(padx=20, pady=20)

root.mainloop()
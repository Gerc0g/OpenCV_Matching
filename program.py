import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

def load_image():
    global img, img_display, imgtk_main
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_display = img.copy()
        display_image(img_display, label_img, "main")

def load_template():
    global templ, templ_display, imgtk_template
    file_path = filedialog.askopenfilename()
    if file_path:
        templ = cv2.imread(file_path, cv2.IMREAD_COLOR)
        templ_display = templ.copy()
        display_image(templ_display, label_templ, "template")

def display_image(cv_image, label, key):
    b, g, r = cv2.split(cv_image)
    img_adjusted = cv2.merge((r, g, b))
    img_pil = Image.fromarray(img_adjusted)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label.config(image=imgtk)
    label.image = imgtk  # Keep a reference!
    if key == "main":
        imgtk_main = imgtk  # Save a reference!
    else:
        imgtk_template = imgtk  # Save a reference!

def matching_method():
    if 'img' not in globals() or 'templ' not in globals():
        messagebox.showerror("Error", "Please load both image and template first.")
        return
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(img, templ, method)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    match_loc = max_loc
    cv2.rectangle(img_display, match_loc,
                  (match_loc[0] + templ.shape[1], match_loc[1] + templ.shape[0]),
                  (255, 0, 0), 2)
    display_image(img_display, label_img, "main")

def update_threshold(value):
    threshold = float(value) / 100
    matching_method()

def clear_rectangles():
    global img_display
    if 'img' in globals():
        img_display = img.copy()
        display_image(img_display, label_img, "main")
    else:
        messagebox.showerror("Error", "Please load an image first.")

root = tk.Tk()
root.title("Сопоставление шаблонов")
root.geometry("1200x600")

frame_controls = tk.Frame(root)
frame_controls.pack(padx=10, pady=10)

btn_load_image = tk.Button(frame_controls, text="Загрузить изображение", command=load_image)
btn_load_image.pack(side=tk.LEFT, padx=10)

btn_load_template = tk.Button(frame_controls, text="Загрузить шаблон", command=load_template)
btn_load_template.pack(side=tk.LEFT, padx=10)

btn_match = tk.Button(frame_controls, text="Сопоставить", command=matching_method)
btn_match.pack(side=tk.LEFT, padx=10)

btn_clear = tk.Button(frame_controls, text="Очистить рамки", command=clear_rectangles)
btn_clear.pack(side=tk.LEFT, padx=10)

label_threshold = tk.Label(frame_controls, text="Порог:")
label_threshold.pack(side=tk.LEFT, padx=10)

slider_threshold = tk.Scale(frame_controls, from_=0, to=100, orient=tk.HORIZONTAL, command=update_threshold)
slider_threshold.set(80)
slider_threshold.pack(side=tk.LEFT, padx=10)

label_img = tk.Label(root)
label_img.pack(side=tk.LEFT, padx=20)

label_templ = tk.Label(root)
label_templ.pack(side=tk.LEFT, padx=20)

root.mainloop()

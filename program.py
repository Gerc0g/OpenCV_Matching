import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame
from tkinter.ttk import Combobox
import cv2
from PIL import Image, ImageTk
import numpy as np

class TemplateMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сопоставление шаблонов")
        self.root.geometry("1200x700")

        self.img = None
        self.img_display = None
        self.gray_image_1 = None
        self.templ = None
        self.templ_display = None
        self.gray_image_2 = None
        self.max_display_width = 400
        self.max_display_height = 300
        self.img_path = None

        self.matching_methods = {
            "SQDIFF": cv2.TM_SQDIFF,
            "SQDIFF NORMED": cv2.TM_SQDIFF_NORMED,
            "CCORR": cv2.TM_CCORR,
            "CCORR NORMED": cv2.TM_CCORR_NORMED,
            "CCOEFF": cv2.TM_CCOEFF,
            "CCOEFF NORMED": cv2.TM_CCOEFF_NORMED,
        }

        self.selected_method = tk.StringVar()
        self.selected_method.set("CCOEFF NORMED")  # Метод по умолчанию

        self.create_widgets()

    def create_widgets(self):
        # Фрейм для управления
        frame_controls = tk.Frame(self.root)
        frame_controls.pack(side=tk.TOP, pady=10)

        tk.Button(frame_controls, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_controls, text="Загрузить шаблон", command=self.load_template).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_controls, text="Сопоставить", command=self.matching_method).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_controls, text="Очистить рамки", command=self.clear_rectangles).pack(side=tk.LEFT, padx=10)

        tk.Label(frame_controls, text="Метод:").pack(side=tk.LEFT, padx=10)
        self.method_combobox = Combobox(frame_controls, textvariable=self.selected_method, values=list(self.matching_methods.keys()))
        self.method_combobox.pack(side=tk.LEFT, padx=10)

        # Метка для отображения результата
        self.result_label = tk.Label(frame_controls, text="", font=("Arial", 12))
        self.result_label.pack(side=tk.LEFT, padx=10)

        # Фрейм для изображений
        self.frame_images = tk.Frame(self.root)
        self.frame_images.pack(pady=10)

        # LabelFrame для исходного изображения
        self.frame_img = LabelFrame(self.frame_images, text="Исходное изображение", font=("Arial", 12))
        self.frame_img.pack(side=tk.LEFT, padx=20)

        # LabelFrame для шаблона
        self.frame_templ = LabelFrame(self.frame_images, text="Шаблон", font=("Arial", 12))
        self.frame_templ.pack(side=tk.LEFT, padx=20)

        # LabelFrame для метрик
        self.frame_metrics = LabelFrame(self.frame_images, text="Метрики", font=("Arial", 12))
        self.frame_metrics.pack(side=tk.LEFT, padx=20)

        # Метрики
        self.metrics_text = tk.Text(self.frame_metrics, height=10, width=40)
        self.metrics_text.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img_path = file_path
            self.img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.img_display = self.resize_image_with_aspect_ratio(self.img, self.max_display_width, self.max_display_height)
            self.gray_image_1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            #self.gray_image_1 = cv2.GaussianBlur(self.gray_image_1, (5, 5), 0)  # Фильтрация шума
            self.update_canvas(self.img_display, self.frame_img, "img")
            # Сброс переменных шаблона
            self.templ = None
            self.templ_display = None
            self.gray_image_2 = None
            if hasattr(self, 'canvas_templ'):
                self.canvas_templ.delete("all")

    def load_template(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.templ = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.templ_display = self.resize_image_with_aspect_ratio(self.templ, self.max_display_width, self.max_display_height)
            self.gray_image_2 = cv2.cvtColor(self.templ, cv2.COLOR_BGR2GRAY)
            self.gray_image_2 = cv2.bitwise_not(self.gray_image_2)
            #self.gray_image_2 = cv2.GaussianBlur(self.gray_image_2, (5, 5), 0)  # Фильтрация шума
            self.update_canvas(self.templ_display, self.frame_templ, "templ")

    def resize_image_with_aspect_ratio(self, image, max_width, max_height):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        if w > h:
            new_width = min(w, max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(h, max_height)
            new_width = int(new_height * aspect_ratio)
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def update_canvas(self, cv_image, frame, key):
        b, g, r = cv2.split(cv_image)
        img_adjusted = cv2.merge((r, g, b))
        img_pil = Image.fromarray(img_adjusted)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Определение размеров изображения
        img_width, img_height = img_pil.size

        # Обновление canvas и размеров фрейма
        if key == "img":
            if hasattr(self, 'canvas_img'):
                self.canvas_img.destroy()
            self.canvas_img = tk.Canvas(frame, width=img_width, height=img_height)
            self.canvas_img.pack()
            self.canvas_img.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas_img.image = imgtk
        elif key == "templ":
            if hasattr(self, 'canvas_templ'):
                self.canvas_templ.destroy()
            self.canvas_templ = tk.Canvas(frame, width=img_width, height=img_height)
            self.canvas_templ.pack()
            self.canvas_templ.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas_templ.image = imgtk
    '''
    def matching_method(self):
        if self.img is None or self.templ is None:
            messagebox.showerror("Error", "Пожалуйста, загрузите изображение и шаблон.")
            return

        method_name = self.selected_method.get()
        method = self.matching_methods[method_name]

        result = cv2.matchTemplate(self.gray_image_1, self.gray_image_2, method)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

        threshold = 0.01 if method_name in ["SQDIFF", "SQDIFF NORMED"] else 0.99

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        mean_val = np.mean(result)
        std_dev = np.std(result)

        # Порог стандартного отклонения
        std_dev_threshold = 0.05

        metrics = {
            'Метод': method_name,
            'Минимальное значение': min_val,
            'Максимальное значение': max_val,
            'Координаты минимального значения': min_loc,
            'Координаты максимального значения': max_loc,
            'Среднее значение': mean_val,
            'Стандартное отклонение': std_dev
        }

        self.display_metrics(metrics)

        match_val = min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val
        match_loc = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc

        # Проверка порога стандартного отклонения
        if std_dev > std_dev_threshold:
            self.result_label.config(text="Шаблон не найден (стандартное отклонение слишком высоко)", fg="red")
            return

        if (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val <= threshold) or \
           (method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val >= threshold):
            h, w = self.templ.shape[:2]
            # Рисуем более отчетливый прямоугольник (толще и другого цвета)
            cv2.rectangle(self.img, match_loc, (match_loc[0] + w, match_loc[1] + h), (0, 255, 0), 3)
            self.img_display = self.resize_image_with_aspect_ratio(self.img, self.max_display_width, self.max_display_height)
            self.update_canvas(self.img_display, self.frame_img, "img")
            self.result_label.config(text="Шаблон найден", fg="green")
        else:
            self.result_label.config(text="Шаблон не найден", fg="red")
    '''
    ''' БОЛЕЕ НОВАЯ
    def matching_method(self):
        if self.img is None or self.templ is None:
            messagebox.showerror("Error", "Пожалуйста, загрузите изображение и шаблон.")
            return

        method_name = self.selected_method.get()
        method = self.matching_methods[method_name]

        result = cv2.matchTemplate(self.gray_image_1, self.gray_image_2, method)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

        threshold = 0.01 if method_name in ["SQDIFF", "SQDIFF NORMED"] else 0.99

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        mean_val = np.mean(result)
        std_dev = np.std(result)

        # Установка порогов для метрик
        mean_val_threshold = 0.5
        std_dev_threshold = 0.5

        metrics = {
            'Метод': method_name,
            'Координаты минимального значения': min_loc,
            'Координаты максимального значения': max_loc,
            'Среднее значение': f"{mean_val:.5f}",
            'Стандартное отклонение': f"{std_dev:.5f}"
        }

        self.display_metrics(metrics)

        match_val = min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val
        match_loc = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc

        # Фильтрация на основе метрик
        if mean_val > mean_val_threshold:
            self.result_label.config(text="Шаблон не найден (Среднее значение выше порога)", fg="red")
            return
        if std_dev > std_dev_threshold:
            self.result_label.config(text="Шаблон не найден (Стандартное отклонение слишком высоко)", fg="red")
            return

        if (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val <= threshold) or \
        (method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val >= threshold):
            h, w = self.templ.shape[:2]
            # Рисуем более отчетливый прямоугольник (толще и другого цвета)
            cv2.rectangle(self.img, match_loc, (match_loc[0] + w, match_loc[1] + h), (0, 255, 0), 3)
            self.img_display = self.resize_image_with_aspect_ratio(self.img, self.max_display_width, self.max_display_height)
            self.update_canvas(self.img_display, self.frame_img, "img")
            self.result_label.config(text="Шаблон найден", fg="green")
        else:
            self.result_label.config(text="Шаблон не найден", fg="red")
    '''
    def matching_method(self):
        if self.img is None or self.templ is None:
            messagebox.showerror("Error", "Пожалуйста, загрузите изображение и шаблон.")
            return

        method_name = self.selected_method.get()
        method = self.matching_methods[method_name]

        result = cv2.matchTemplate(self.gray_image_1, self.gray_image_2, method)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

        threshold = 0.01 if method_name in ["SQDIFF", "SQDIFF NORMED"] else 0.99

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        mean_val = np.mean(result)
        std_dev = np.std(result)

        # Установка порогов для метрик
        mean_val_threshold = 0.2
        std_dev_threshold = 0.2

        metrics = {
            'Метод': method_name,
            'Координаты минимального значения': min_loc,
            'Координаты максимального значения': max_loc,
            'Среднее значение': f"{mean_val:.5f}",
            'Стандартное отклонение': f"{std_dev:.5f}",
            'Минимальное значение': f"{min_val:.5f}",
            'Максимальное значение': f"{max_val:.5f}"
        }

        self.display_metrics(metrics)

        match_val = min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val
        match_loc = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc

        # Фильтрация на основе метрик
        if mean_val > mean_val_threshold:
            self.result_label.config(text="Шаблон не найден (Среднее значение выше порога)", fg="red")
            return
        if std_dev > std_dev_threshold:
            self.result_label.config(text="Шаблон не найден (Стандартное отклонение слишком высоко)", fg="red")
            return
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val > threshold:
            self.result_label.config(text="Шаблон не найден (Минимальное значение выше порога)", fg="red")
            return
        if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and match_val < threshold:
            self.result_label.config(text="Шаблон не найден (Максимальное значение ниже порога)", fg="red")
            return

        h, w = self.templ.shape[:2]
        # Рисуем более отчетливый прямоугольник (толще и другого цвета)
        cv2.rectangle(self.img, match_loc, (match_loc[0] + w, match_loc[1] + h), (0, 255, 0), 3)
        self.img_display = self.resize_image_with_aspect_ratio(self.img, self.max_display_width, self.max_display_height)
        self.update_canvas(self.img_display, self.frame_img, "img")
        self.result_label.config(text="Шаблон найден", fg="green")

    def display_metrics(self, metrics):
        self.metrics_text.delete(1.0, tk.END)
        for key, value in metrics.items():
            self.metrics_text.insert(tk.END, f"{key}: {value}\n")

    def clear_rectangles(self):
        if self.img_path:
            self.img = cv2.imread(self.img_path)  # Перезагрузить оригинальное изображение
            self.img_display = self.resize_image_with_aspect_ratio(self.img, self.max_display_width, self.max_display_height)
            self.gray_image_1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.gray_image_1 = cv2.GaussianBlur(self.gray_image_1, (5, 5), 0)  # Фильтрация шума
            self.update_canvas(self.img_display, self.frame_img, "img")
            self.result_label.config(text="")  # Очистить метку результата
        else:
            messagebox.showerror("Error", "Пожалуйста, загрузите изображение.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateMatcherApp(root)
    root.mainloop()

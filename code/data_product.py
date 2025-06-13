
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import threading
from PIL import Image, ImageTk
import sys, os
from model_creation import TreeDecision_SVR,TreeDecision_XGBOOST,TreeDecision_XGBOOST_KPCA,TreeDecision_XGBOOST_PCA
import traceback

def resource_path(relative_path):
    try:
        return os.path.join(sys._MEIPASS, relative_path)
    except AttributeError:
        return os.path.abspath(relative_path)


features = ['price_ori', 'item_rating', 'price_actual', 'total_rating', 'favorite', 'discount']


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Prediction App")
        self.root.geometry("800x600")  # Set a fixed size to match the image

        self.canvas = None
        self.bg_photo = None  # To avoid garbage collection
        self.model_cache = {}

        # === Background Image ===
        bg_image = Image.open(resource_path("ssj_goku.jpg")).resize((800, 600))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        background_label = tk.Label(root, image=self.bg_photo)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # === UI Widgets on Top ===
        self.model_options = ['svr', 'gxboost', 'gxboost+pca', 'kpca']
        self.selected_model = tk.StringVar(value=self.model_options[0])

        frame = tk.Frame(root, bg="#ffffff", bd=2)
        frame.place(x=20, y=20)

        tk.Label(frame, text="Select Model:", bg="#ffffff").pack()
        tk.OptionMenu(frame, self.selected_model, *self.model_options, command=self.model_changed).pack(pady=5)

        tk.Button(frame, text="Upload CSV", command=self.upload_csv).pack(pady=10)

    def upload_csv(self):
        threading.Thread(target=self.process_csv).start()

    def process_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return

        try:
            df = pd.read_csv(path)

            if ('total_sold' not in df.columns or 
                'price_actual' not in df.columns or 
                'price_ori' not in df.columns):
                self.show_error("CSV must contain 'total_sold', 'price_actual', and 'price_ori' columns.")
                return

            df['discount'] = 100 * (1. - df['price_actual'] / df['price_ori'])

            self.y_true = df['total_sold']
            self.X_test = df[features]

            # After loading data, trigger plotting with selected model
            self.root.after(0, self.load_and_plot_model)

        except Exception as e:
            self.show_error(traceback.format_exc())


    def load_and_plot_model(self):
        try:
            model_file = {
                'svr': 'tree_model_SVR.pkl',
                'gxboost': 'TreeDecision_XGBOOST_model.pkl',
                'gxboost+pca': 'TreeDecision_XGBOOST_PCA_model.pkl',
                'kpca': 'TreeDecision_XGBOOST_KPCA_model.pkl'
            }.get(self.selected_model.get())

            if not model_file:
                self.show_error("Selected model is not supported.")
                return

            model = joblib.load(resource_path(model_file))
            y_pred = model.prediction(self.X_test)

            if isinstance(y_pred, pd.DataFrame):
                y_pred = y_pred.iloc[:, 0]

            y_true_aligned, y_pred_aligned = self.y_true.align(y_pred)
            error = abs(y_true_aligned - y_pred_aligned)

            self.plot_results(y_true_aligned, y_pred_aligned, error)

        except Exception as e:
            self.show_error(str(e))

    def show_error(self, message):
        self.root.after(0, lambda: messagebox.showerror("Error", message))

    def plot_results(self, y_true, y_pred, error):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig = plt.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(y_true, y_pred, color='green', label='Predicted vs Actual')
        ax.plot(y_true, y_true, color='red', label='Ideal Line')
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title('Actual vs Predicted Sales')
        ax.legend()
        ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def model_changed(self, _):
        if hasattr(self, 'X_test') and hasattr(self, 'y_true'):
            self.load_and_plot_model()


if __name__ == "__main__":
    print(resource_path("ssj_goku.jpg"))
    print(resource_path("tree_model_SVR.pkl"))
    print(resource_path("TreeDecision_XGBOOST_model.pkl"))
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()

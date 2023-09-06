import tkinter as tk
from PIL import ImageTk, Image,ImageDraw,ImageGrab
import time
import win32gui
from model import predict_image,preprocess_image,model

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=300, height=300, bg='black')
        self.canvas.pack()
        self.button_predict = tk.Button(self.root, text='Predict', command=self.predict)
        self.button_predict.pack()
        self.button_erase_all = tk.Button(self.root, text='Erase All', command=self.erase_all)
        self.button_erase_all.pack()
        self.canvas.bind('<B1-Motion>', self.draw)

    def draw(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='white',outline='white')

    def predict(self):
        # Call your predict function here with the image path as parameter
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        file=time.time()*1000
        im = ImageGrab.grab(rect).save(str(file)+".png")  # get image of the current location
        img_arr=preprocess_image(str(file)+".png")
        predict=predict_image(model,img_arr)
        print(f"Predicted Canvas : {predict} Accuracy : {acc}")

    def erase_all(self):
        self.canvas.delete('all')


app = App()
app.root.mainloop()

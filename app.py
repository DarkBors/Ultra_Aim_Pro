
from tkinter import *
import tkinter as tk
import tkinter.messagebox
import customtkinter as ck


## Setting up Initial Things:
ck.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ck.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        ## Setting up Initial Things:
        self.title("ULTRA-AIM-PRO")
        self.geometry("720x550")
        self.resizable(True, True)


        # # Make the Frame:
        # frame_1 = ck.CTkFrame(master=app)
        # frame_1.pack(pady=10, padx=10, fill="both", expand=True)


        ## Creating a Container:
        container = ck.CTkFrame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)        

        





if __name__ == "__main__":
    app = App()
    app.mainloop()
    


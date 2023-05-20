###################################################################################
##                             ULTRA-AIM-PRO                                     ##
##                                                                               ##
## Ultra96-based PYNQ AI-Managed Performance and Reliability Optimization system ##
##                                                                               ##
##                  Created by: Dark Bors v0.0.2-beta                          ##
##                                                                               ##
##                                                                 Final Project ##
###################################################################################

import tkinter as tk
import cv2
import time
import tkinter.messagebox as messagebox
import numpy as np
import logging

from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    
    
class GUI(tk.Tk):   
    def __init__(self):
        super().__init__()

        # Create a logger instance
        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)

        # Create a file handler to save logs to a file
        file_handler = logging.FileHandler('debugLog.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create a console handler to print logs to the terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define the log message format
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Set the formatter for both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Log the message without line breaks
        logger.info("\n"
            "###################################################################################\n"
            "##                             ULTRA-AIM-PRO                                     ##\n"
            "##                                                                               ##\n"
            "## Ultra96-based PYNQ AI-Managed Performance and Reliability Optimization system ##\n"
            "##                                                                               ##\n"
            "##                  Created by: Dark Bors v0.0.2-beta                 ##\n"
            "##                                                                               ##\n"
            "##                                                                 Final Project ##\n"
            "###################################################################################"
        )

        
        # Set up the window properties
        self.title("ULTRA-AIM-PRO", )
        self.geometry("1400x750")
        # self.resizable(False, False)

        # Set minimum size and make the first column and row of the grid expandable
        self.minsize(800, 450)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Set up style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background='#282C34', foreground='white')
        self.style.configure('TNotebook.Tab', background='#2C3849', foreground='white', borderwidth=0)
        self.style.map('TNotebook.Tab', background=[('selected', '#0078D7')], foreground=[('selected', 'white')])
        
        
        
        # Set up the tab control and tabs
        self.tabControl = ttk.Notebook(self)
        self.dashboardTab = ttk.Frame(self.tabControl)
        self.analysisTab = ttk.Frame(self.tabControl)
        self.rtvTab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.dashboardTab, text="Dashboard")
        self.tabControl.add(self.analysisTab, text="Analysis")
        self.tabControl.add(self.rtvTab, text="RTV")
        self.tabControl.pack(expand=1, fill="both")

        # Set up the widgets in the Dashboard tab
        self.create_dashboard_tab()
        self.slider1 = None
        self.slider2 = None
        self.slider3 = None
        # self.save_user_details = None
        
        self.current_row = 0
        self.current_row_data = tk.StringVar(value="0,0,0,0,0")

        # Initialize live_data_frame as a class attribute
        self.live_data_frame = tk.LabelFrame(self.dashboardTab, text="Live Data", font=("Arial Baltic", 14), fg="white", bg="#1a1a1a")
        self.live_data_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")

        # Live data section
        live_data_frame = tk.LabelFrame(self.dashboardTab, text="Live Data", font=("Arial Baltic", 14), fg="white", bg="#1a1a1a")
        live_data_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")
        
        self.N_label = tk.Label(live_data_frame, text="N: ", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.N_label.pack()
        self.f_label = tk.Label(live_data_frame, text="f: ", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.f_label.pack()
        self.V_label = tk.Label(live_data_frame, text="V: ", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.V_label.pack()
        self.T_label = tk.Label(live_data_frame, text="T: ", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.T_label.pack()
        self.ttf_label = tk.Label(live_data_frame, text="ttf: ", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.ttf_label.pack()

        # Set up the widgets in the Analysis tab
        self.create_analysis_tab()
        self.screenshot_button_analysis = None

        # Set up the widgets in the RTV tab
        self.create_rtv_tab()
        self.screenshot_button_rtv = None

        # # Initialize the video capture object
        # self.cap = cv2.VideoCapture(0)

        # Start the main loop
        self.mainloop()


    def create_rtv_tab(self):
        # RTV tab - camera and buttons
        self.camera_label = tk.Label(self.rtvTab)
        self.camera_label.pack()

        # Take screenshot button
        self.screenshot_button_rtv = tk.Button(self.rtvTab, text="Take screenshot", font=("Arial Baltic", 14),
                                               command=self.take_screenshot_rtv)
        self.screenshot_button_rtv.pack(pady=10)

        # Set up camera
        self.cap = cv2.VideoCapture(0)
        self.show_camera_feed()
            
    def show_camera_feed(self):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (750, 650))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.camera_label.config(image=img)
            self.camera_label.image = img
            

        else:
            # Handle the case where the read fails by displaying a placeholder image or logging an error message
            placeholder_img = ImageTk.PhotoImage(Image.new('RGB', (750, 650)))
            self.camera_label.config(image=placeholder_img)
            self.camera_label.image = placeholder_img
            #LOG
            logging.error("⛔ Failed to read from camera")

        self.after(10, self.show_camera_feed)
    
    
    
    def save_user_details(self):
        return

    # To add later !
    def take_screenshot_analysis(self):
        return

    # To add later !
    def refresh_graphs_analysis(self):
        return

    def take_screenshot_rtv(self):
        ret, frame = self.cap.read()

        if ret:
            filepath = filedialog.asksaveasfilename(defaultextension=".png")
            if filepath:
                cv2.imwrite(filepath, frame)
                #LOG
                logging.info("ℹ️ screenshot saved : %s", filepath)

    def create_dashboard_tab(self):
        # Top right - logo
        logo_img = ImageTk.PhotoImage(Image.open("rsz_11logo.png"))
        logo_label = tk.Label(self.dashboardTab, image=logo_img, bg="#1a1a1a")
        logo_label.image = logo_img
        logo_label.place(relx=1.0, anchor="ne")

        # Device details
        device_details = tk.LabelFrame(self.dashboardTab, text="Device details", font=("Arial Baltic", 14), fg="white", bg="#1a1a1a")
        device_details.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        device_details_lbl = tk.Label(device_details, text="Device name: ULTRA96\nFirmware version: 1.0", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        device_details_lbl.pack(padx=10, pady=10)

        # Open data file button
        open_file_button = tk.Button(device_details, text="Open Data File", font=("Arial Baltic", 12), bg="#0072c6", fg="white", command=self.open_data_file)
        open_file_button.pack(side="top", padx=10, pady=10)
        
        self.filepath_label = tk.Label(device_details, text="", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.filepath_label.pack(side="bottom", padx=10, pady=10)
        
        self.data_file_label = tk.Label(device_details, text="", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.data_file_label.pack(side="bottom", padx=10, pady=10)

        # Timer label
        self.timer_label = tk.Label(device_details, text="WTC: 00:00:00:000", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        self.timer_label.pack(side="bottom", padx=10, pady=10)

       
        # Lower left - fields to fill in name, date and time and a button that saves this information.
        user_details = tk.LabelFrame(self.dashboardTab, text="User details", font=("Arial Baltic", 14), fg="white", bg="#1a1a1a")
        user_details.grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        name_lbl = tk.Label(user_details, text="Name", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        name_lbl.grid(row=1, column=0, padx=10, pady=10)
        self.name_entry = tk.Entry(user_details, font=("Arial Baltic", 12))
        self.name_entry.grid(row=1, column=1, padx=10, pady=10)
        date_lbl = tk.Label(user_details, text="Date", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        date_lbl.grid(row=2, column=0, padx=10, pady=10)

        date_entry = tk.Entry(user_details, font=("Arial Baltic", 12))
        date_entry.grid(row=2, column=1, padx=10, pady=10)

        time_lbl = tk.Label(user_details, text="Time", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        time_lbl.grid(row=3, column=0, padx=10, pady=10)

        time_entry = tk.Entry(user_details, font=("Arial Baltic", 12))
        time_entry.grid(row=3, column=1, padx=10, pady=10)

        save_btn = tk.Button(user_details, text="Save", font=("Arial Baltic", 12), bg="#0072c6", fg="white", command=self.save_user_details)
        save_btn.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
    
        self.style.configure('TLabelFrame', background='#1a1a1a', foreground='white')
        # Upper right - three-slide bars
        data_controls = tk.LabelFrame(self.dashboardTab, text="Data controls", font=("Arial Baltic", 14), fg="white", bg="#1a1a1a")
        data_controls.grid(row=0, column=3, rowspan=2, padx=5, pady=5, sticky="nw")
        
        # slider 1 - Voltage
        slider1_lbl = tk.Label(data_controls, text="Voltage", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        slider1_lbl.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.slider1 = ttk.Scale(data_controls, from_=0, to=100, orient="horizontal", style="Custom.Horizontal.TScale")
        self.slider1.set(50)
        self.slider1.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # slider 2 - Frequency
        slider2_lbl = tk.Label(data_controls, text="Frequency", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        slider2_lbl.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        self.slider2 = ttk.Scale(data_controls, from_=0, to=100, orient="horizontal", style="Custom.Horizontal.TScale")
        self.slider2.set(50)
        self.slider2.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # slider 3 - Dynamic Logic
        slider3_lbl = tk.Label(data_controls, text="Dynamic Logic", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
        slider3_lbl.grid(row=4, column=3, padx=5, pady=5, sticky="w")

        self.slider3 = ttk.Scale(data_controls, from_=0, to=100, orient="horizontal", style="Custom.Horizontal.TScale")
        self.slider3.set(50)
        self.slider3.grid(row=5, column=3, padx=5, pady=5, sticky="w")
        
            
                    
    def open_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return 

        try:
            with open(file_path, "r") as file:
                header = file.readline().split()
                data = []
                for line in file:
                    row = line.split()
                    row_data = {}
                    for i, value in enumerate(row):
                        if value.strip():
                            try:
                                row_data[header[i]] = float(value)
                            except ValueError:
                                row_data[header[i]] = 0.0
                    data.append(row_data)

            self.data = data
            self.current_index = 0

            # Display the data in the GUI
            self.display_data_in_gui(data)

        except Exception as e:
            # LOG
            logging.error(f"⛔ An error occurred while opening the file: {e}")
            return logging.error("⛔ An error occurred while opening the file.")


                    
            
    def display_data_in_gui(self, data):
        self.data_rows = data
        
        # Initialize the timer
        self.timer_start = time.time()
        self.timer_running = False
        self.update_timer()
       


        def update_data_display():
            # Check if labels are created
            if hasattr(self, "f_label"):
                # Update the labels' text with the current row data
                row_data = self.data[self.current_index]
                self.f_label.config(text=f"f: {row_data['f']}", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
                #LOG
                logging.info("ℹ️ Display data: f=%s", row_data['f'])
                self.V_label.config(text=f"V: {row_data['V']}", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
                #LOG
                logging.info("ℹ️ Display data: V=%s", row_data['V'])
                self.T_label.config(text=f"T: {row_data['T']}", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
                #LOG
                logging.info("ℹ️ Display data: T=%s", row_data['T'])
                self.ttf_label.config(text=f"ttf: {row_data['ttf']}", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
                #LOG
                logging.info("ℹ️ Display data: ttf=%s", row_data['ttf'])
                self.N_label.config(text=f"N: {row_data['N']}", font=("Arial Baltic", 12), fg="white", bg="#1a1a1a")
                #LOG
                logging.info("ℹ️ Display data: N=%s", row_data['N'])
                
                
                # Move to the next row of data
                self.current_index += 1
                if self.current_index >= len(self.data_rows):
                    self.current_index = 0
                
                # Schedule the next update
                self.after(10000, update_data_display)
                #LOG
                logging.info("ℹ️ -->⏱ Update Timer ⏱<--") 

                # Update the analysis graphs
                self.update_analysis_graphs()
                              

            else:
                # If labels are not created, try again after a short delay
                self.after_idle(update_data_display)
                logging.error("⛔ labels are not created, try again after a short delay")

        # Start the initial update
        update_data_display()

        # Create and place the "PAUSE" and "NEXT" buttons if they don't exist
        if not hasattr(self, 'pause_button') or not hasattr(self, 'next_button'):
            self.create_pause_next_buttons(self.live_data_frame)

    def create_pause_next_buttons(self, parent):
        # code for creating pause and next buttons
        pass


    def pause_data_display(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="RESUME")
            self.after_cancel(self.after_id)
        else:
            self.pause_button.config(text="PAUSE")
            self.after_id = self.after(15000, self.display_data_in_gui)


    def update_timer(self):
        elapsed_time = time.time() - self.timer_start
        milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
        seconds = int(elapsed_time % 60)
        minutes = int((elapsed_time / 60) % 60)
        hours = int((elapsed_time / 3600) % 24)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
        self.timer_label.config(text=f"WTC: {time_str}")
        self.after(10, self.update_timer)

    def start_timer(self):
        self.timer_start = time.time()
        self.timer_running = True


    def stop_timer(self):
        self.timer_running = False

    
    def create_analysis_tab(self):
        # Analysis tab - graphs and buttons
        fig1 = Figure(figsize=(5, 4), dpi=100)
        fig1.subplots_adjust(bottom=0.15)
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Training and validation loss", color="#0E3264")
        ax1.set_xlabel("Epochs", color="#0E3264")
        ax1.set_ylabel("Loss", color="#0E3264")
        ax1.grid(True, color="#0E3264")
        line1, = ax1.plot([1, 2, 3], [1, 2, 3], color="cyan")

        fig2 = Figure(figsize=(5, 4), dpi=100)
        fig2.subplots_adjust(bottom=0.15)
        ax2 = fig2.add_subplot(111)
        ax2.set_title("Predictions and true values", color="#0E3264")
        ax2.set_xlabel("True Values", color="#0E3264")
        ax2.set_ylabel("Predictions", color="#0E3264")
        ax2.grid(True, color="#0E3264")
        line2, = ax2.plot([1, 2, 3], [1, 2, 3], color="cyan")

        fig3 = Figure(figsize=(5, 4), dpi=100)
        fig3.subplots_adjust(bottom=0.15)
        self.ax3 = fig3.add_subplot(111)
        self.ax3.set_title("How voltage affects the TTF", color="#0E3264")
        self.ax3.set_xlabel("Voltage", color="#0E3264")
        self.ax3.set_ylabel("TTF", color="#0E3264")
        self.ax3.grid(True, color="#0E3264")
        self.line3, = self.ax3.plot([], [], color="cyan")

        fig4 = Figure(figsize=(5, 4), dpi=100)
        fig4.subplots_adjust(bottom=0.15)
        self.ax4 = fig4.add_subplot(111)
        self.ax4.set_title("How frequency affects the TTF", color="#0E3264")
        self.ax4.set_xlabel("Frequency", color="#0E3264")
        self.ax4.set_ylabel("TTF", color="#0E3264")
        self.ax4.grid(True, color="#0E3264")
        self.line4, = self.ax4.plot([1, 2, 3], [1, 2, 3], color="cyan")

        fig5 = Figure(figsize=(5, 4), dpi=100)
        fig5.subplots_adjust(bottom=0.15)
        self.ax5 = fig5.add_subplot(111)
        self.ax5.set_title("How the dynamic logic affects the TTF", color="#0E3264")
        self.ax5.set_xlabel("Dynamic Logic", color="#0E3264")
        self.ax5.set_ylabel("TTF", color="#0E3264")
        self.ax5.grid(True, color="#0E3264")
        self.line5, = self.ax5.plot([1, 2, 3], [1, 2, 3], color="cyan")
        
        fig6 = Figure(figsize=(5, 4), dpi=100)
        fig6.subplots_adjust(bottom=0.15)
        ax6 = fig6.add_subplot(111)
        ax6.set_title("Ambient temperature VS CPU temperature", color="#0E3264")
        ax6.set_xlabel("Ambient temperature", color="#0E3264")
        ax6.set_ylabel("CPU temperature", color="#0E3264")
        ax6.grid(True, color="#0E3264")
        line6, = ax6.plot([1, 2, 3], [1, 2, 3], color="cyan")
        
        canvas1 = FigureCanvasTkAgg(fig1, self.analysisTab)
        canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        canvas2 = FigureCanvasTkAgg(fig2, self.analysisTab)
        canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        self.canvas3 = FigureCanvasTkAgg(fig3, self.analysisTab)
        self.canvas3.get_tk_widget().grid(row=0, column=2, padx=10, pady=10)
        self.canvas4 = FigureCanvasTkAgg(fig4, self.analysisTab)
        self.canvas4.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)
        self.canvas5 = FigureCanvasTkAgg(fig5, self.analysisTab)
        self.canvas5.get_tk_widget().grid(row=1, column=1, padx=10, pady=10)
        canvas6 = FigureCanvasTkAgg(fig6, self.analysisTab)
        canvas6.get_tk_widget().grid(row=1, column=2, padx=10, pady=10)

        # Buttons frame
        buttons_frame = tk.Frame(self.analysisTab, bg="#1a1a1a")
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=20)

        # Take screenshot button
        self.screenshot_button_analysis = tk.Button(buttons_frame, text="Take Screenshot", font=("Arial Baltic", 14),
                                                    command=self.take_screenshot_analysis, bg="#4f8bc9", fg="white",
                                                    relief="flat", activebackground="#4f8bc9", activeforeground="white")
        self.screenshot_button_analysis.pack(side="left", padx=10)
        
        
        
    def update_analysis_graphs(self):
        # Get the data for the graphs
        data = np.array([(row.get('V', 0), row.get('f', 0), row.get('T', 0), row.get('ttf', 0), row.get('N', 0)) for row in self.data_rows])

        # Update fig3
        self.line3.set_data(data[:, 0], data[:, 3])
        self.ax3.relim()
        self.ax3.autoscale_view()

      
        # Update fig4
        self.line4.set_data(data[:, 1], data[:, 3])
      
        self.ax4.relim()
        self.ax4.autoscale_view()

        # Update fig5
        self.line5.set_data(data[:, 4], data[:, 3])
        self.ax5.relim()
        self.ax5.autoscale_view()

        # Redraw the canvases
        self.canvas3.draw()
        self.canvas4.draw()
        self.canvas5.draw()



if __name__ == '__main__':
    gui = GUI()
    gui.mainloop()
else:
    logging.critical("💩 Critical Error ! ! ! 💩")



    #     # Bind the shutdown function to the window's destroy event
    #     self.protocol("WM_DELETE_WINDOW", self.on_shutdown)

    #     # Display the GUI

    # def on_shutdown(self):
    #     # Log the shutdown event
    #     logging.warning(" ⚠️  ULTRA-AIM-PRO is shutting down")

    #     # Perform any cleanup or additional tasks before closing the GUI if needed ! 
    #     # ...

    #     # Close the GUI
    #     self.destroy()
        

        
        

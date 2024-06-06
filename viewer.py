import sys
import os
import json
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageColor
import subprocess
from tkinter import filedialog
import json_to_xml as PAGE          
import cv2


TARGET_SUFFIX = '_tagged'


class ImageViewer(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.img_ht = 750
        self.image_files = []
        self.page = None
        self.index = 0
        self.image = None
        self.photoimage = None
        self.directory = image_dir
        self.region_check = tk.BooleanVar(value=True)
        self.baseline_check = tk.BooleanVar()
        self.line_check = tk.BooleanVar()
        self.order_check = tk.BooleanVar()

        self.get_files()
        
        # Left panel
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, columnspan=6, rowspan=3) 
        
        # Color legend
        self.legend_frame = tk.Frame(root)
        self.legend_frame.grid(row=3, column=0, columnspan=6, rowspan=4)
        self.legend = ColorLegend(self.legend_frame)
        
        # Right panel
        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=6, columnspan=6, rowspan=7)

        # Add image to it
        self.label = tk.Label(self.right_frame)
        self.label.pack()
        
        # The listbox on the left panel
        self.listbox = tk.Listbox(self.left_frame)
        self.listbox.pack()
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

 

        self.button_frame = tk.Frame(root)    
        self.button_frame.grid(row=8, column=0, columnspan=12, rowspan=3)
        self.add_buttons()
        #self.pack()
        self.populate_listbox()
        self.show_image()
        

    def add_buttons(self):
        tk.Checkbutton(self.button_frame, text='Regions', command=self.show_image, variable=self.region_check).grid(row=0, column=0, columnspan=2, sticky=tk.E)
        tk.Checkbutton(self.button_frame, text='Lines', command=self.show_image, variable=self.line_check).grid(row=0, column=2, columnspan=2, sticky=tk.E)
        tk.Checkbutton(self.button_frame, text='Baselines', command=self.show_image, variable=self.baseline_check).grid(row=0, column=4, columnspan=2, sticky=tk.E)
        tk.Checkbutton(self.button_frame, text='Order', variable=self.order_check, command=self.show_image).grid(row=0, column=6, columnspan=2, sticky=(tk.E, tk.N))
        
        
        tk.Button(self.button_frame, text='Next', command=self.show_next).grid(row=1, column=0, columnspan=2, sticky=tk.E)
        tk.Button(self.button_frame, text='Prev', command=self.show_prev).grid(row=1, column=2, columnspan=2, sticky=tk.E)
        tk.Button(self.button_frame, text='Open XML', command=self.open_xml).grid(row=1, column=4, columnspan=2, sticky=tk.E)
        tk.Button(self.button_frame, text='Gen XML', command=self.generate_xml).grid(row=1, column=4, columnspan=2, sticky=tk.E)
        tk.Button(self.button_frame, text='Select Directory', command=self.select_directory).grid(row=1, column=6, columnspan=2, sticky=(tk.E, tk.N))
         
    def get_files(self):
        files = os.listdir(self.directory)
        # get list of image files
        image_files = [f for f in files if f.lower().endswith(('jpeg', 'png', 'gif', 'jpg'))]
        image_files.sort()
        self.list_box_files = image_files
        self.image_files = [os.path.join(self.directory, f) for f in image_files]

        
    def generate_xml(self, show_image=True):
        print('Generating json')
        full_img_name = self.image_files[self.index]
        # first gen json + save
        json_file = PAGE.get_json_file(full_img_name, TARGET_SUFFIX)
        self.page = PAGE.TranscriptionPage(filename=json_file, 
                              imagefile=full_img_name)
        self.page.save()
        print('generating xml')
        # Next gen xml and save
        output_xml = full_img_name[:-3]+'xml'
        xml_obj = PAGE.PageXML(self.page, output_xml)   
        xml_obj.write_xml()
        if show_image:
            print('showing after generating')
            self.show_image()
        
    def populate_listbox(self):
        # Clear the listbox
        self.listbox.delete(0, tk.END)
        # Populate the listbox with image files
        for image in self.list_box_files:
            self.listbox.insert(tk.END, image)
        
    def show_image(self):
        print('Showing ', self.image_files[self.index])
        image_file = self.image_files[self.index]
        json_file = image_file[:-3]+'json'
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img)
        #self.image = Image.open(image_file)
        if os.path.exists(json_file):
            self.page = PAGE.TranscriptionPage(load_from_file=json_file)
        else:
            self.page = None
            print('JSON FILE NON_EXISTENT')
            self.generate_xml(show_image=False)
        if self.region_check.get():
            self.show_regions()
        if self.line_check.get():
            self.show_lines()
        if self.order_check.get():
            self.show_order()
        if self.baseline_check.get():
            self.show_baselines()

        self.resize_and_display_img()

    def show_next(self):
        self.index += 1
        if self.index >= len(self.image_files):
            self.index = 0
        self.show_image()

    def show_prev(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.image_files) - 1
        self.show_image()
        
    def open_xml(self):
        xml_filename = os.path.splitext(self.image_files[self.index])[0] + '.xml'
        subprocess.run(['java', '-jar', 'JPageViewer/JPageViewer.jar', xml_filename])

    def on_select(self, event):
        # Get the selected index from the listbox
        selected_index = self.listbox.curselection()
        if selected_index:
            self.index = selected_index[0]
            self.show_image()

    def select_directory(self):
        directory = None
        directory = filedialog.askdirectory(initialdir=self.directory)
        print('Directory chosen', directory)
        if directory is not None:
            self.directory = directory
            self.get_files()
            self.populate_listbox()
            self.index = 0
            self.show_image()
    
    def resize_and_display_img(self):
        height = self.img_ht
        width = int(self.image.size[0]/self.image.size[1]*height)
        self.image_resized = self.image.resize((width, height))
        self.photoimage = ImageTk.PhotoImage(self.image_resized)
        self.label.config(image=self.photoimage)
        self.label.image = self.photoimage
    
    def show_regions(self):

        draw = ImageDraw.Draw(self.image, 'RGBA')
        for r in self.page.region_dict:
            region_type = self.page.region_dict[r]['type']['name']
            color = self.legend.colors[region_type]
            rgba = ImageColor.getrgb(color) + (100,) 
            polygon = np.array(self.page.json[r]['line_polygon']).astype(int)   
            polygon = [(x, y) for (x,y) in polygon]
            draw.polygon(polygon, fill=rgba, outline="black")
        #self.resize_and_display_img()

        


    def show_lines(self):
        types = ['Archivist_Tag', 'Crossed_out', 'Printed_Regular', 'Printed_Italics', 'Printed_Bold']
        draw = ImageDraw.Draw(self.image, 'RGBA')
        for l in self.page.line_keys():
            if not 'region' in self.page.line_dict[l]:
                print('line', l, 'assigned', self.page.line_dict[l]['assigned'])
            if self.page.line_dict[l]['region'] is None:
                
                continue
            color = "black"
            for t in types:
                
                if self.page.get_tag(l, t):
                    color = self.legend.colors[t]
                    
            polygon = np.array(self.page.json[l]['line_polygon']).astype(int)   
            polygon = [(x, y) for (x,y) in polygon] 
            polygon += [polygon[0]]
            
            
            draw.line(polygon, fill=color, width=5)
        #self.resize_and_display_img()
               
    def show_baselines(self):
        draw = ImageDraw.Draw(self.image, 'RGB')
        for l in self.page.line_dict:
            if len(self.page.json[l]['text']) == 0:
                continue
            baseline = np.array(self.page.json[l]['baseline']).astype(int)
            for ind, (pt1, pt2) in enumerate(zip(baseline, baseline[1:])):
                pt1 = (pt1[0], pt1[1])
                pt2 = (pt2[0], pt2[1])
                if ind == 0:
                    radius = 30
                    draw.ellipse((pt1[0]-radius, pt1[1]-radius, pt1[0]+radius, pt1[1]+radius), fill="red", width=5)
                draw.line([pt1, pt2], fill="black", width=5)       
    
    def show_order(self):
        draw = ImageDraw.Draw(self.image)
        colors = ["blue", "green", "red"]
        c1 = self.page.get_center_poly(self.page.sorted_regions[0][0])
        c1 = (c1[0], c1[1])
        for ind, gp in enumerate(self.page.sorted_regions):
            for r in gp:               
                color = colors[ind%3]
                c2 = self.page.get_center_poly(r)
                c2 = (c2[0], c2[1])
                draw.line([c1, c2], fill=color, width=5)    
                c1 = c2

class ColorLegend:
    def __init__(self, master):
        self.master = master
        

        self.legend_frame = tk.Frame(self.master)
        self.legend_frame.pack()

        self.colors = {
                   'paragraph': "red", 
                    'floating': "green",
                    'heading': "pink", 
                    'logo': "magenta", 
                    'letterhead': "blue", 
                    'decoration':"royalblue",
                    'pageNo': "yellow", 
                    'signature': "orange",
                    'graphic': "cyan", 
                    'image': "white", 
                    'stamp': "brown", 
                    'margin': "gray", 
                    'noise': "plum",
                    'separator':"lightskyblue", 
                    # For lines start here
                    'Archivist_Tag': "red", 
                    'Crossed_out': "cyan",
                    'Printed_Regular': "darkgreen", 
                    'Printed_Italics': "pink",
                    'Printed_Bold': "brown"
            
        }

        self.create_legend()

    def create_legend(self):
        for label, color in self.colors.items():
            color_frame = tk.Frame(self.legend_frame)
            color_frame.pack(anchor="w")

            color_label = tk.Label(color_frame, text=label)
            color_label.pack(side="left")

            color_rect = tk.Canvas(color_frame, width=20, height=10, bg=color)
            color_rect.pack(side="left", padx=5)
        
image_dir = './'

image_dir = None
if len(sys.argv) > 1:
    image_dir = sys.argv[1]
if len(sys.argv) > 2:
    TARGET_SUFFIX = sys.argv[2]
    print('target_suffix is ', TARGET_SUFFIX)

if image_dir is None or not os.path.exists(image_dir):
    print("Please provide image directory as first argument")
    sys.exit(1)

root = tk.Tk()
app = ImageViewer(master=root)
app.mainloop()

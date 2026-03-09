import os
import json
import pandas as pd
import pydicom
import numpy as np
import cv2
from glob import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sv_ttk  
from mdai.visualize import load_dicom_image

class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD Ai explorer")
        self.root.geometry("1400x820") 
        
        sv_ttk.set_theme("dark")
        
        # State Management
        self.image_cache = {}
        self.cache_limit = 12 
        self.current_full_res_img = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        try:
            self.load_config_and_data(config_path)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Could not start: {e}")
            return
            
        self.setup_ui()

    def hex_to_bgr(self, hex_str):
        """Converts #RRGGBB hex strings to (B, G, R) for OpenCV."""
        if not hex_str or not isinstance(hex_str, str):
            return (0, 255, 255) # Default Cyan
        hex_str = hex_str.lstrip('#')
        lv = len(hex_str)
        try:
            rgb = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            return (rgb[2], rgb[1], rgb[0]) # Return BGR
        except:
            return (0, 255, 255)

    def load_config_and_data(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.project_dir = config.get("Project_dir", "./ProjectDir")
        self.project_id = config.get("mdai_project_id")
        self.user_map = config.get("user_map", {})

        anno_matches = glob(os.path.join(self.project_dir, f"*{self.project_id}*annotations*.json"))
        meta_matches = glob(os.path.join(self.project_dir, f"*{self.project_id}*metadata*.json"))

        if not anno_matches or not meta_matches:
            raise FileNotFoundError("Could not find annotation or metadata JSON files in the project directory.")

        with open(anno_matches[0], 'r') as f:
            anno_data = json.load(f)
            annos = anno_data['datasets'][0].get('annotations', [])
            
            # Map Label Names and Colors from labelGroups
            self.label_map = {}
            self.color_map = {}
            for g in anno_data.get('labelGroups', []):
                for l in g.get('labels', []):
                    lid = l['id']
                    self.label_map[lid] = l['name']
                    self.color_map[lid] = self.hex_to_bgr(l.get('color'))

        with open(meta_matches[0], 'r') as f:
            meta_data = json.load(f)
            meta = meta_data['datasets'][0].get('dicomMetadata', [])

        df_annos = pd.json_normalize(annos)
        df_meta = pd.json_normalize(meta)
        if 'labelId' in df_annos.columns:
            df_annos['labelName'] = df_annos['labelId'].map(self.label_map)

        self.merged_df = pd.merge(df_annos, df_meta, on="SOPInstanceUID", how="inner")
        
        # Map SOPInstanceUID to actual file paths
        self.sop_to_file = {}
        for fp in glob(os.path.join(self.project_dir, "*.dcm")):
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                self.sop_to_file[ds.SOPInstanceUID] = os.path.basename(fp)
            except: continue

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Sidebar ---
        sidebar = ttk.Frame(self.paned, style="Card.TFrame")
        self.paned.add(sidebar, weight=1)
        
        ttk.Label(sidebar, text="FEED", font=('Segoe UI Variable', 12, 'bold')).pack(pady=10, padx=10, anchor="w")
        
        control_frame = ttk.Frame(sidebar)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_list)
        
        search_row = ttk.Frame(control_frame)
        search_row.pack(fill=tk.X)
        ttk.Entry(search_row, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_row, text="✕", width=2, command=lambda: self.search_var.set("")).pack(side=tk.RIGHT, padx=(2,0))

        ttk.Button(control_frame, text="RESET VIEW", style="Accent.TButton", command=self.reset_zoom).pack(fill=tk.X, pady=(8, 0))

        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.listbox = tk.Listbox(list_frame, font=('Segoe UI Variable', 10), bg="#252525", fg="#ffffff", borderwidth=0, highlightthickness=0)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Main Viewer ---
        right_container = ttk.Frame(self.paned)
        self.paned.add(right_container, weight=4)

        self.canvas = tk.Canvas(right_container, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Interaction Bindings
        self.canvas.bind("<MouseWheel>", self.handle_zoom)      
        self.canvas.bind("<Button-4>", self.handle_zoom)        
        self.canvas.bind("<Button-5>", self.handle_zoom)        
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<Button-3>", lambda e: self.reset_zoom()) 

        # --- Annotation Info Panel ---
        self.info_frame = ttk.LabelFrame(right_container, text=" ANNOTATION INFO ", padding=8)
        self.info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.meta_fields = {}
        fields = [
            ("Label", 0, 0),        ("Collaborator", 0, 1), ("Timestamp", 0, 2),
            ("Frame", 1, 0),        ("Task ID", 1, 1),      ("Filename", 1, 2)
        ]
        for label, r, c in fields:
            f = ttk.Frame(self.info_frame)
            f.grid(row=r, column=c, sticky="w", padx=12, pady=2)
            ttk.Label(f, text=label.upper(), font=('Segoe UI', 7, 'bold'), foreground="#888888").pack(anchor="w")
            val = ttk.Label(f, text="---", font=('Consolas', 9))
            val.pack(anchor="w")
            self.meta_fields[label] = val

        self.update_listbox(self.merged_df)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    def filter_list(self, *args):
        query = self.search_var.get().lower()
        filtered = self.merged_df[self.merged_df['labelName'].str.lower().str.contains(query, na=False)]
        self.update_listbox(filtered)

    def update_listbox(self, df):
        self.listbox.delete(0, tk.END)
        self.current_df_view = df 
        for _, row in df.iterrows():
            self.listbox.insert(tk.END, f" {row.get('labelName', 'N/A')} — F{row.get('frameNumber', 1)}")

    def reset_zoom(self):
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
        self.render_image()

    def on_select(self, event=None):
        selection = self.listbox.curselection()
        if not selection: return
        
        row = self.current_df_view.iloc[selection[0]]
        self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
        
        sop_uid = row['SOPInstanceUID']
        fname = self.sop_to_file.get(sop_uid)
        if not fname: return

        if sop_uid not in self.image_cache:
            self.image_cache[sop_uid] = load_dicom_image(os.path.join(self.project_dir, fname))
        
        data = self.image_cache[sop_uid]
        f_num = int(row.get('frameNumber', 1))
        img = data[f_num - 1] if data.ndim > 2 else data
        
        # Normalize and Convert to RGB
        img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_rgb = cv2.cvtColor(img_8, cv2.COLOR_GRAY2RGB) if len(img_8.shape) == 2 else img_8

        # --- GET JSON COLOR ---
        label_id = row.get('labelId')
        draw_color = self.color_map.get(label_id, (0, 255, 255)) 

        # --- DRAW POLYGONS ---
        if 'data.foreground' in row and isinstance(row['data.foreground'], list):
            for poly in row['data.foreground']:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_rgb, [pts], True, draw_color, 2, lineType=cv2.LINE_AA)
                
                # 15% Transparency Fill
                overlay = img_rgb.copy()
                cv2.fillPoly(overlay, [pts], draw_color)
                cv2.addWeighted(overlay, 0.15, img_rgb, 0.85, 0, img_rgb)

        # --- DRAW POINTS ---
        elif 'data.x' in row and not pd.isna(row['data.x']):
            cv2.circle(img_rgb, (int(row['data.x']), int(row['data.y'])), 6, draw_color, -1)

        self.current_full_res_img = Image.fromarray(img_rgb)
        
        # Update UI Information
        hex_color = '#%02x%02x%02x' % (draw_color[2], draw_color[1], draw_color[0])
        self.meta_fields["Label"].config(text=row.get('labelName', 'Unknown'), foreground=hex_color)
        self.meta_fields["Collaborator"].config(text=self.user_map.get(row['createdById'], row['createdById']))
        self.meta_fields["Filename"].config(text=fname)
        self.meta_fields["Timestamp"].config(text=str(row.get('createdAt', 'N/A'))[:19])
        self.meta_fields["Task ID"].config(text=self.project_id)
        self.meta_fields["Frame"].config(text=f"Frame {f_num}")
        
        self.render_image()

    def render_image(self):
        if self.current_full_res_img is None: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10: canvas_w, canvas_h = 800, 500

        width, height = self.current_full_res_img.size
        if self.zoom_level == 1.0:
            scale = min(canvas_w / width, canvas_h / height) * 0.95
            new_size = (int(width * scale), int(height * scale))
        else:
            new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
        
        resized = self.current_full_res_img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2 + self.pan_x, canvas_h//2 + self.pan_y, image=self.tk_img)

    def handle_zoom(self, event):
        if event.num == 4 or event.delta > 0: self.zoom_level *= 1.1
        else: self.zoom_level /= 1.1
        self.zoom_level = max(0.05, min(self.zoom_level, 10.0))
        self.render_image()

    def start_pan(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def do_pan(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.render_image()

if __name__ == "__main__":
    root = tk.Tk()
    # Replace "configlocal.json" with the path to your config file
    app = MDAIExplorer(root, "configlocal.json") 
    root.mainloop()

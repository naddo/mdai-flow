import os
import json
import pandas as pd
import numpy as np
import cv2
import mdai
import sys
from glob import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sv_ttk  
from mdai.visualize import load_dicom_image

class MDAIExplorer:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("MD.ai Cloud Explorer")
        self.root.geometry("1450x850") 
        sv_ttk.set_theme("dark")
        
        # State
        self.current_full_res_img = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.project_instances = {}
        self.images_dirs = {}   # dataset_id -> images_dir path
        self.cache_dir = "./mdai_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.active_filters = {"Label": True, "User": True, "Dataset": True}

        try:
            self.load_config_and_data(config_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
            return
            
        self.setup_ui()

    def hex_to_bgr(self, hex_str):
        if not hex_str or not isinstance(hex_str, str): return (0, 255, 255)
        hex_str = hex_str.lstrip('#')
        rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0]) 

    def silence(self):
        """Redirect both stdout and stderr to devnull."""
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull

    def unsilence(self):
        """Restore stdout and stderr."""
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._devnull.close()

    def load_config_and_data(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.project_id = config.get("mdai_project_id")
        self.token = config.get("mdai_token")
        self.domain = config.get("mdai_domain", "ucsf.md.ai")
        self.user_map = config.get("user_map", {})
        d_ids = config.get("mdai_dataset_id", [])
        if isinstance(d_ids, str): d_ids = [d_ids]

        self.silence()
        try:
            client = mdai.Client(domain=self.domain, access_token=self.token)
        finally:
            self.unsilence()

        print("Successfully authenticated.")

        all_annos, all_metas, label_defs = [], [], pd.DataFrame()

        for d_id in d_ids:
            try:
                # Download annotations (suppressed)
                self.silence()
                try:
                    p = client.project(self.project_id, dataset_id=d_id, path=self.cache_dir)
                    self.project_instances[d_id] = p
                    if p is not None and hasattr(p, 'images_dir') and p.images_dir:
                        self.images_dirs[d_id] = p.images_dir
                    client.project(self.project_id, dataset_id=d_id, annotations_only=True)
                finally:
                    self.unsilence()
                print(f"✔ Downloaded annotations for dataset {d_id}")

                # Download metadata (suppressed)
                self.silence()
                try:
                    client.download_dicom_metadata(self.project_id, d_id, path=".")
                finally:
                    self.unsilence()
                print(f"✔ Downloaded metadata for dataset {d_id}")

                # Process annotations
                anno_files = glob(f"*{self.project_id}*annotations*{d_id}*.json")
                if anno_files:
                    res = mdai.common_utils.json_to_dataframe(max(anno_files, key=os.path.getmtime))
                    if res['annotations'] is not None:
                        res['annotations']['datasetId'] = d_id
                        all_annos.append(res['annotations'])
                    if res['labels'] is not None:
                        label_defs = pd.concat([label_defs, res['labels']]).drop_duplicates(subset=['labelId'])

                # Process metadata
                meta_files = glob(f"*{self.project_id}*metadata*{d_id}*.json")
                if meta_files:
                    with open(max(meta_files, key=os.path.getmtime), 'r') as f:
                        m_json = json.load(f)
                        if 'datasets' in m_json:
                            df_m = pd.json_normalize(m_json['datasets'][0].get('dicomMetadata', []))
                            df_m['datasetId'] = d_id
                            all_metas.append(df_m)

            except Exception as e:
                self.unsilence()  # safety restore
                print(f"✘ Failed for dataset {d_id}: {e}")
                continue

        df_a = pd.concat(all_annos, ignore_index=True) if all_annos else pd.DataFrame()
        df_m = pd.concat(all_metas, ignore_index=True) if all_metas else pd.DataFrame()
        self.label_map = dict(zip(label_defs.labelId, label_defs.labelName)) if not label_defs.empty else {}
        self.color_map = {r['labelId']: self.hex_to_bgr(r.get('color')) for _, r in label_defs.iterrows()} if not label_defs.empty else {}
        if not df_a.empty:
            df_a['labelName'] = df_a['labelId'].map(self.label_map)
            if 'data' in df_a.columns:
                coords = pd.json_normalize(df_a['data']).add_prefix('data.')
                df_a = pd.concat([df_a.drop(columns=['data']), coords], axis=1)
        self.merged_df = pd.merge(df_a, df_m, on="SOPInstanceUID", how="left", suffixes=('', '_m')) if not df_m.empty else df_a
        print("✔ All data loaded. Opening GUI...")

    def setup_ui(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # 1. SIDEBAR
        self.sidebar = ttk.Frame(self.paned, width=450)
        self.paned.add(self.sidebar, weight=1)

        search_frame = ttk.Frame(self.sidebar)
        search_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_table)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(fill=tk.X)

        btn_frame = ttk.Frame(self.sidebar)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.btns = {}
        for f_name in ["Label", "User", "Dataset"]:
            b = tk.Button(btn_frame, text=f"● {f_name}", command=lambda f=f_name: self.toggle_filter(f),
                          bg="#005fb8", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), pady=5)
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            self.btns[f_name] = b

        # Treeview — show="" hides the built-in column header row
        cols = ("Label", "User", "Dataset")
        self.tree = ttk.Treeview(self.sidebar, columns=cols, show="", selectmode="browse")
        for c in cols:
            self.tree.column(c, width=130, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_item_select)

        # 2. VIEWER
        self.v_frame = ttk.Frame(self.paned)
        self.paned.add(self.v_frame, weight=4)
        self.canvas = tk.Canvas(self.v_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(self.v_frame, text="Ready.")
        self.status.pack(side=tk.BOTTOM, pady=5)

        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self, 'lx', e.x) or setattr(self, 'ly', e.y))
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.populate_table()

    def toggle_filter(self, f_name):
        self.active_filters[f_name] = not self.active_filters[f_name]
        b = self.btns[f_name]
        if self.active_filters[f_name]:
            b.config(bg="#005fb8", text=f"● {f_name}")
        else:
            b.config(bg="#333333", text=f"○ {f_name}")
        self.filter_table()

    def filter_table(self, *args):
        q = self.search_var.get().lower()
        for i in self.tree.get_children(): self.tree.delete(i)
        for idx, row in self.merged_df.iterrows():
            lbl, uid = str(row.get('labelName', '')), row.get('createdById')
            user = self.user_map.get(uid, uid if uid else "Unknown")
            ds = str(row.get('datasetId', 'N/A'))
            match = False
            if self.active_filters["Label"] and q in lbl.lower(): match = True
            if self.active_filters["User"] and q in user.lower(): match = True
            if self.active_filters["Dataset"] and q in ds.lower(): match = True
            if q == "" or match:
                self.tree.insert("", tk.END, iid=idx, values=(lbl, user, ds))

    def populate_table(self): self.filter_table()
    def on_item_select(self, e):
        sel = self.tree.selection()
        if not sel: return
        
        row = self.merged_df.iloc[int(sel[0])]
        sop_uid = row.get('SOPInstanceUID')
        ds_id = row.get('datasetId')

        # 1. Search in images_dir (downloaded at startup) or anywhere in cache_dir
        search_roots = []
        if ds_id in self.images_dirs:
            search_roots.append(self.images_dirs[ds_id])
        search_roots.append(self.cache_dir)

        path = None
        for root in search_roots:
            found = glob(os.path.join(root, "**", f"{sop_uid}.dcm"), recursive=True)
            if found:
                path = found[0]
                break

        if not path:
            # 2. Not cached yet — trigger a full dataset image download now
            self.status.config(text="⏳ Downloading DICOM images...")
            self.root.update_idletasks()
            try:
                self.silence()
                try:
                    p = mdai.Client(domain=self.domain, access_token=self.token)
                    proj = p.project(self.project_id, dataset_id=ds_id, path=self.cache_dir)
                    if proj is not None and hasattr(proj, 'images_dir') and proj.images_dir:
                        self.images_dirs[ds_id] = proj.images_dir
                finally:
                    self.unsilence()

                # Search again after download
                for root in [self.images_dirs.get(ds_id, self.cache_dir), self.cache_dir]:
                    found = glob(os.path.join(root, "**", f"{sop_uid}.dcm"), recursive=True)
                    if found:
                        path = found[0]
                        break

                if not path:
                    raise FileNotFoundError(f"Download completed but '{sop_uid}.dcm' not found.")
            except Exception as err:
                messagebox.showerror("Error", f"Could not retrieve image: {err}")
                return

        # 4. Success! Load and draw using the discovered path
        self.status.config(text="✅ Image Loaded")
        self.load_and_draw(path, row)
    
    def load_and_draw(self, path, row):
        try:
            img = load_dicom_image(path)
            img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            if len(img_8.shape) == 2: img_8 = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
            if 'data.x' in row and pd.notna(row['data.x']):
                x, y, w, h = int(row['data.x']), int(row['data.y']), int(row['data.width']), int(row['data.height'])
                cv2.rectangle(img_8, (x, y), (x+w, y+h), self.color_map.get(row['labelId'], (0,255,0)), 3)
            self.current_full_res_img = Image.fromarray(cv2.cvtColor(img_8, cv2.COLOR_BGR2RGB))
            self.render_image()
            self.status.config(text=f"Viewing: {row.get('labelName')}")
        except: pass

    def render_image(self):
        if not self.current_full_res_img: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        iw, ih = self.current_full_res_img.size
        s = min(cw/iw, ch/ih) * self.zoom_level
        resized = self.current_full_res_img.resize((int(iw*s), int(ih*s)), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2 + self.pan_x, ch//2 + self.pan_y, image=self.tk_img)

    def handle_zoom(self, e): self.zoom_level *= 1.1 if e.delta > 0 else 0.9; self.render_image()
    def do_pan(self, e): 
        self.pan_x += (e.x - self.lx); self.pan_y += (e.y - self.ly)
        self.lx, self.ly = e.x, e.y; self.render_image()

if __name__ == "__main__":
    root = tk.Tk(); app = MDAIExplorer(root, "configLocal.json"); root.mainloop()

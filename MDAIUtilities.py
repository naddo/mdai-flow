import os
import json
import csv
from glob import glob
import mdai
import pandas as pd
from html import escape

# -------------------------------
# Helper Functions
# -------------------------------

def find_latest_json_by_project(folder, project_id, json_type_hint=None):
    pattern = f"*{project_id}*"
    if json_type_hint:
        pattern += f"*{json_type_hint}*"
    pattern += ".json"
    files = glob(os.path.join(folder, pattern))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def flatten_entry(entry, prefix=""):
    row = {}
    for k, v in entry.items():
        key_name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            row.update(flatten_entry(v, key_name))
        else:
            row[key_name] = v if v is not None else ""
    return row

def save_csv(rows, path, variables):
    if rows is None:
        rows = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    variables_with_sno = ["S.NO"] + variables
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=variables_with_sno)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row_with_sno = {"S.NO": i, **row}
            writer.writerow(row_with_sno)

def save_html(rows, path, variables, title="Table"):
    if rows is None:
        rows = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    variables_with_sno = ["S.NO"] + variables
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title>")
        f.write("<style>table {border-collapse: collapse;} td, th {border: 1px solid #ccc; padding: 5px; vertical-align: top;} th {background: #f0f0f0;} pre {margin: 0;}</style>")
        f.write("</head><body>")
        f.write(f"<h2>{title}</h2>")
        f.write("<table><tr>")
        for var in variables_with_sno:
            f.write(f"<th>{var}</th>")
        f.write("</tr>")
        for i, row in enumerate(rows, start=1):
            f.write("<tr>")
            row_with_sno = {"S.NO": i, **row}
            for var in variables_with_sno:
                value = str(row_with_sno.get(var, "")).replace("\n", "<br>")
                f.write(f"<td>{value}</td>")
            f.write("</tr>")
        f.write("</table></body></html>")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------
# MD.ai Dataset Download
# -------------------------------

def download_mdai_dataset(config):
    domain = config.get("mdai_domain", "md.ai")
    token = config["mdai_token"]
    project_id = config["mdai_project_id"]
    dataset_id = config["mdai_dataset_id"]
    label_group_id = config.get("mdai_label_group_id", None)
    output_dir = config.get("output_dir", "mdai_output")

    os.makedirs(output_dir, exist_ok=True)

    client = mdai.Client(domain=domain, access_token=token)
    print("🔽 Downloading data from MD.ai...")

    kwargs = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "path": output_dir,
        "annotations_only": True
    }
    if label_group_id:
        kwargs["label_group_id"] = label_group_id

    client.project(**kwargs)
    client.download_dicom_metadata(
        project_id=project_id,
        dataset_id=dataset_id,
        format="json",
        path=output_dir
    )

    print("✅ Download complete!")
    return project_id, output_dir

# -------------------------------
# Export Combined Labels + Annotations + DICOM
# -------------------------------

def export_mdai_json_to_csv_html(config_path, output_dir=None):
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    if output_dir:
        config["output_dir"] = output_dir

    dicom_vars = config.get("dicom_vars", [])
    annotation_vars = config.get("annotation_vars", [])

    project_id, output_dir = download_mdai_dataset(config)

    # -----------------------------
    # Load annotation JSON via MD.ai dataframe utility
    # -----------------------------
    annotation_file = find_latest_json_by_project(output_dir, project_id, "annotations")
    if not annotation_file:
        print("⚠️ No annotation JSON found, skipping annotations.")
        anno_df = pd.DataFrame()
        studies_df = pd.DataFrame()
        labels_df = pd.DataFrame()
    else:
        results = mdai.common_utils.json_to_dataframe(annotation_file)
        anno_df = results['annotations']
        studies_df = results['studies']
        labels_df = results['labels']
        print("Annotation columns:", anno_df.columns.tolist())
        print("Studies columns:", studies_df.columns.tolist())

    # -----------------------------
    # Merge annotations with studies
    # -----------------------------
    if not anno_df.empty and not studies_df.empty:
        studies_df_renamed = studies_df.rename(columns={'studyUid': 'StudyInstanceUID'})
        merged_df = anno_df.merge(studies_df_renamed, on='StudyInstanceUID', how='left')
    else:
        merged_df = pd.DataFrame()

    # -----------------------------
    # Prepare combined rows for CSV/HTML
    # -----------------------------
    combined_rows = []

    # Labels
    for i, row in labels_df.iterrows():
        row_dict = row.to_dict()
        row_dict["Type"] = "Label"
        combined_rows.append(row_dict)

    # Annotations
    for i, row in merged_df.iterrows():
        row_dict = row.to_dict()
        row_dict["Type"] = "Annotation"
        combined_rows.append(row_dict)

    # Determine all variables for CSV/HTML
    all_vars = ["Type"]
    for r in combined_rows:
        for k in r.keys():
            if k not in all_vars:
                all_vars.append(k)

    save_csv(combined_rows, os.path.join(output_dir, f"{project_id}_labels_annotations.csv"), all_vars)
    save_html(combined_rows, os.path.join(output_dir, f"{project_id}_labels_annotations.html"), all_vars, title="Labels + Annotations")

    # -----------------------------
    # Process DICOM metadata with filtering
    # -----------------------------
    dicom_file = find_latest_json_by_project(output_dir, project_id, "dicom_metadata")
    dicom_entries = []

    if dicom_file:
        dicoms_data = load_json(dicom_file)
        for dataset in dicoms_data.get("datasets", []):
            dataset_id_from_config = dataset.get("id", config.get("mdai_dataset_id", ""))
            for entry in dataset.get("dicomMetadata", []):
                entry["datasetId"] = dataset_id_from_config
                dicom_entries.append(entry)

    print(f"🔹 Total DICOM entries loaded: {len(dicom_entries)}")
 #   if dicom_entries:
 #       print("🔹 First 5 DICOM entries (flattened) for debug:")
#        for d in dicom_entries[:5]:
 #           print(flatten_entry(d))

    flattened_dicoms = [flatten_entry(d) for d in dicom_entries]

    # Filter dicom fields based on config
    if dicom_vars:
        filtered_dicoms = []
        for d in flattened_dicoms:
            filtered_dicoms.append({k: v for k, v in d.items() if k in dicom_vars})
        dicom_vars_to_use = dicom_vars
    else:
        filtered_dicoms = flattened_dicoms
        dicom_vars_to_use = sorted({k for d in flattened_dicoms for k in d.keys()})

    save_csv(filtered_dicoms, os.path.join(output_dir, f"{project_id}_dicom.csv"), dicom_vars_to_use)
    save_html(filtered_dicoms, os.path.join(output_dir, f"{project_id}_dicom.html"), dicom_vars_to_use, title="DICOM Metadata")

    print("\n✅ Export complete! Labels + Annotations and DICOM files generated.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    config_path = "config.json"
    output_dir = "mdai_output"
    export_mdai_json_to_csv_html(config_path, output_dir)

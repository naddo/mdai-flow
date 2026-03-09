import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import pydicom  # Required to read the actual DICOM pixel data
import mdai
from mdai.visualize import load_dicom_image, display_annotations
import numpy as np
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

    rows = rows or []

    os.makedirs(os.path.dirname(path), exist_ok=True)

    variables_with_sno = ["S.NO"] + variables

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=variables_with_sno)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):

            row_with_sno = {"S.NO": i, **row}
            writer.writerow(row_with_sno)


def save_html(rows, path, variables, title="Table"):

    rows = rows or []

    os.makedirs(os.path.dirname(path), exist_ok=True)

    variables_with_sno = ["S.NO"] + variables

    with open(path, "w", encoding="utf-8") as f:

        f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title>")
        f.write("<style>table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:5px;} th{background:#f0f0f0;}</style>")
        f.write("</head><body>")
        f.write(f"<h2>{title}</h2>")
        f.write("<table><tr>")

        for var in variables_with_sno:
            f.write(f"<th>{escape(var)}</th>")

        f.write("</tr>")

        for i, row in enumerate(rows, start=1):

            f.write("<tr>")

            row_with_sno = {"S.NO": i, **row}

            for var in variables_with_sno:

                value = escape(str(row_with_sno.get(var, ""))).replace("\n", "<br>")

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
    label_group_id = config.get("mdai_label_group_id")

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
# Export Function
# -------------------------------

def export_mdai_json_to_csv_html(config_path, output_dir=None):

    with open(config_path) as f:
        config = json.load(f)

    debug = config.get("debug", False)

    annotation_filtering = config.get("annotation_filtering", True)
    dicom_filtering = config.get("dicom_filtering", True)

    if output_dir:
        config["output_dir"] = output_dir

    dicom_vars = config.get("dicom_vars", [])
    annotation_vars = config.get("annotation_vars", [])

    mandatory_annotation_vars = config.get(
        "mandatory_annotation_vars",
        ["labelGroupId", "createdByName"]
    )

    project_id, output_dir = download_mdai_dataset(config)

    client = mdai.Client(
        domain=config.get("mdai_domain", "md.ai"),
        access_token=config["mdai_token"]
    )

    # -----------------------------
    # Load project users
    # -----------------------------

    try:

        users = client.project_users(project_id)

        user_map = {u["id"]: u.get("name", "Unknown User") for u in users}

    except Exception as e:

        print("⚠️ Could not fetch project users:", e)

        user_map = {}

    # -----------------------------
    # Load annotation JSON
    # -----------------------------

    annotation_file = find_latest_json_by_project(output_dir, project_id, "annotations")

    if not annotation_file:

        print("⚠️ No annotation JSON found")

        anno_df = pd.DataFrame()
        studies_df = pd.DataFrame()
        labels_df = pd.DataFrame()

    else:

        results = mdai.common_utils.json_to_dataframe(annotation_file)

        anno_df = results["annotations"]
        studies_df = results["studies"]
        labels_df = results["labels"]

        # -----------------------------
        # Ensure label metadata columns exist
        # -----------------------------

        expected_label_columns = [
            "labelId",
            "labelName",
            "labelGroupId",
            "labelGroupName",
            "color",
            "annotationMode",
            "scope"
        ]

        for col in expected_label_columns:

            if col not in labels_df.columns:
                labels_df[col] = ""

        labels_df = labels_df.fillna("")

    # -----------------------------
    # Merge annotations + studies
    # -----------------------------

    if not anno_df.empty and not studies_df.empty:

        studies_df = studies_df.rename(columns={"studyUid": "StudyInstanceUID"})

        merged_df = pd.merge(
            anno_df,
            studies_df,
            on="StudyInstanceUID",
            how="left"
        )

        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_y")]
        merged_df.columns = merged_df.columns.str.replace("_x$", "", regex=True)

    else:

        merged_df = pd.DataFrame()

# -----------------------------
    # Join label metadata and flatten coordinates
    # -----------------------------
    if not merged_df.empty:
        # 1. Map labels from labels_df
        if not labels_df.empty:
            label_metadata = labels_df[["labelId", "labelName", "color"]]
            merged_df = merged_df.merge(label_metadata, on="labelId", how="left")

        # 2. Map User Names using the user_map from your config
        # This fixes the "Unknown User" issue in your HTML
        config_user_map = config.get("user_map", {})
        merged_df["createdByName"] = merged_df["createdById"].map(config_user_map).fillna("Unknown User")

        # 3. Flatten the 'data' column into data.x, data.y, etc.
        # This populates the coordinate columns seen in your latest HTML
        if 'data' in merged_df.columns:
            coords_df = pd.json_normalize(merged_df['data']).add_prefix('data.')
            merged_df = pd.concat([merged_df.drop(columns=['data']), coords_df], axis=1)

    merged_df = merged_df.fillna("")
    # -----------------------------
    # Map createdBy → username
    # -----------------------------
    if not merged_df.empty and "createdById" in merged_df.columns:
        # Get the map from the config file (U_OjzRqO -> Shree, etc.)
        config_user_map = config.get("user_map", {})
        
        # Apply the map to the createdById column
        merged_df["createdByName"] = merged_df["createdById"].map(config_user_map).fillna("Unknown User")

    # -----------------------------
    # Annotation filtering
    # -----------------------------

    if not merged_df.empty:

        vars_to_keep = list(set(annotation_vars + mandatory_annotation_vars))

        for col in vars_to_keep:
            if col not in merged_df.columns:
                merged_df[col] = ""

        if annotation_filtering:
            merged_df_filtered = merged_df[vars_to_keep]
        else:
            merged_df_filtered = merged_df

    else:

        merged_df_filtered = merged_df

    # -----------------------------
    # Combine rows
    # -----------------------------

    combined_rows = []

    if not labels_df.empty:

        for _, row in labels_df.iterrows():

            r = row.to_dict()
            r["Type"] = "Label"

            combined_rows.append(r)

    if not merged_df_filtered.empty:

        for _, row in merged_df_filtered.iterrows():

            r = row.to_dict()
            r["Type"] = "Annotation"

            combined_rows.append(r)

    # -----------------------------
    # Determine all variables
    # -----------------------------

    all_vars = set()

    for r in combined_rows:
        all_vars.update(r.keys())

    all_vars = ["Type"] + sorted([v for v in all_vars if v != "Type"])

    # -----------------------------
    # Save labels + annotations
    # -----------------------------

    save_csv(
        combined_rows,
        os.path.join(output_dir, f"{project_id}_labels_annotations.csv"),
        all_vars
    )

    save_html(
        combined_rows,
        os.path.join(output_dir, f"{project_id}_labels_annotations.html"),
        all_vars,
        title="Labels + Annotations"
    )

    # -----------------------------
    # Process DICOM metadata
    # -----------------------------

    dicom_file = find_latest_json_by_project(output_dir, project_id, "dicom_metadata")

    dicom_entries = []

    if dicom_file:

        dicoms_data = load_json(dicom_file)

        for dataset in dicoms_data.get("datasets", []):

            dataset_id_val = dataset.get("id", config.get("mdai_dataset_id", ""))

            for entry in dataset.get("dicomMetadata", []):

                entry["datasetId"] = dataset_id_val
                dicom_entries.append(entry)

    print(f"🔹 Total DICOM entries loaded: {len(dicom_entries)}")

    flattened_dicoms = [flatten_entry(d) for d in dicom_entries]

    if dicom_filtering and dicom_vars:

        filtered_dicoms = [
            {k: v for k, v in d.items() if k in dicom_vars}
            for d in flattened_dicoms
        ]

        dicom_vars_to_use = dicom_vars

    else:

        filtered_dicoms = flattened_dicoms
        dicom_vars_to_use = sorted({k for d in flattened_dicoms for k in d})

    save_csv(
        filtered_dicoms,
        os.path.join(output_dir, f"{project_id}_dicom.csv"),
        dicom_vars_to_use
    )

    save_html(
        filtered_dicoms,
        os.path.join(output_dir, f"{project_id}_dicom.html"),
        dicom_vars_to_use,
        title="DICOM Metadata"
    )

    print("\n✅ Export complete!")


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":

    config_path = "mdai_config.json"
    output_dir = "mdai_output"

    export_mdai_json_to_csv_html(config_path, output_dir)

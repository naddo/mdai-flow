# main_download.py
from MDAIUtilities import export_mdai_json_to_csv_html  # Import your final function

def main():
    # Path to your JSON config
    config_path = "config.json"  # replace with your actual JSON file

    # Output folder for downloaded data
    output_dir = "./mdai_output"

    # Call the download function
    export_mdai_json_to_csv_html(config_path, output_dir)

if __name__ == "__main__":
    main()

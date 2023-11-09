import os

def remove_html_files(folder_path):
    try:

        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(".html"):
                    file_path = os.path.join(foldername, filename)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
        
        print("HTML files removed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

folder_path = r"C:\Users\Toby\Documents\GitHub\DSP\team_stats"
remove_html_files(folder_path)

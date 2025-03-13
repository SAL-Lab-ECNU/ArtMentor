import os
import json
import pandas as pd

folder_path = "userActions/Entities"
def get_ass(folder_path, output_file):
    results = []  # Store file names and whether the deletion was detected (1 or 0)
    N = 0  # N: Total number of files
    D = 0  # D: Number of deletion operations

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_labels.json'):
            file_path = os.path.join(folder_path, file_name)

            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check the 'removed' field in the 'style'
            style_removed = data.get("style", {}).get("removed", [])
            if style_removed:
                results.append([file_name, 0])  # 0 means incorrect recognition (deletion detected)
                D += 1  # If the 'removed' list has content, count as one deletion
            else:
                results.append([file_name, 1])  # 1 means correct recognition (no deletion detected)

            N += 1  # Count the number of files

    if N == 0:
        print("No matching files found.")
        return

    # Calculate ASS (Artistic Style Sensitivity)
    ass = 1 - (D / N)
    print(f"Artistic Style Sensitivity (ASS) index is: {ass:.2f}")

    # Save the results into a DataFrame
    df = pd.DataFrame(results, columns=["File Name", "Correct Recognition (1=Correct, 0=Incorrect)"])

    # Export the results to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Results have been saved to: {output_file}")


# Main function call
if __name__ == "__main__":
    output_file = "ASS_Result.xlsx"
    get_ass(folder_path, output_file)

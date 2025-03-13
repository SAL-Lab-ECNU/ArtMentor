import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Define dimensions
dimensions = [
    "Realistic", "Deformation", "Imagination", "Color Richness",
    "Color Contrast", "Line Combination", "Line Texture",
    "Picture Organization", "Transformation"
]

# Load JSON file
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Extract all `original` and `current` score sequences for each dimension
def extract_scores(score_comment_dir, output_file_scores):
    score_results = []

    for image_num in range(1, 21):  # Adjust based on actual range
        print(f"Processing image number: {image_num}")

        for dimension in dimensions:
            # Process score_Review file
            score_comment_file = os.path.join(score_comment_dir, f"{image_num}.jpg_{dimension}_score_Review.json")
            score_comment_data = load_json_data(score_comment_file)

            if score_comment_data:
                print(f"Processing file: {score_comment_file}")

                # Extract `original` and `current` scores for all rounds
                for round_data in score_comment_data:
                    if round_data["round"] == 1:
                        continue  # Skip initial round 1 data
                    gpt_score = round_data['data']['scores']['original']
                    user_score = round_data['data']['scores']['current']

                    if gpt_score is not None and user_score is not None:
                        score_results.append({
                            'image': f"{image_num}.jpg",
                            'dimension': dimension,
                            'original': float(gpt_score),
                            'current': float(user_score)
                        })

    # Save the scores to a DataFrame and output to an Excel file
    scores_df = pd.DataFrame(score_results)
    scores_df.to_excel(output_file_scores, index=False)
    print(f"Original and Current scores have been saved to: {output_file_scores}")

# Calculate SC (Spearman) for score consistency
def get_sc(original_scores, current_scores):
    spearman_corr, _ = spearmanr(original_scores, current_scores)
    return spearman_corr

# Calculate SD (Score Difference)
def get_sd(original_scores, current_scores):
    score_diff = np.abs(original_scores - current_scores)
    avg_sd = score_diff.mean()
    return avg_sd

# Process correlations and score differences for each dimension
def calculate_sc_sd(output_file_scores):
    # Load the saved scores
    scores_df = pd.read_excel(output_file_scores)

    # Group by dimension and calculate the SC and SD for each
    sc_sd_results = []
    for dimension in dimensions:
        dim_scores = scores_df[scores_df['dimension'] == dimension]
        original_scores = dim_scores['original']
        current_scores = dim_scores['current']

        # Calculate SC (Spearman correlation) and SD (Average Difference)
        sc_value = get_sc(original_scores, current_scores)
        sd_value = get_sd(original_scores, current_scores)

        sc_sd_results.append({
            'dimension': dimension,
            'SC': sc_value,
            'SD': sd_value
        })

    # Save the SC and SD results
    sc_sd_df = pd.DataFrame(sc_sd_results)
    sc_sd_df.to_excel("SC_SD_Result.xlsx", index=False)
    print(f"SC and SD results have been saved to: SC_SD_Results.xlsx")

# Main function
if __name__ == "__main__":
    score_comment_directory = "Dataset/score_Review"  # Path to JSON files
    output_file_scores = "SC_SD_Results.xlsx"

    # Calculate SC (Spearman) and SD (Average Difference)
    calculate_sc_sd(output_file_scores)

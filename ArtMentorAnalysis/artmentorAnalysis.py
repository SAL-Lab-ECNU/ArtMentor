import os
import json
import numpy as np
import pandas as pd

# Import methods from different analysis modules
from EntityAnalysis import process_json_files, get_Entity_Accuracy, get_Entity_Precision, get_Entity_Recall, get_Entity_F1
from ScoreAnalysis import get_sc, get_sd, extract_scores
from StyleAnalysis import get_ass
from TextAnalysis import get_tar, get_ts, process_directory as process_text_analysis

# Function to load JSON files
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Process entity analysis and save results to Excel
def process_entity_analysis(directory_path, output_file):
    entity_results = []
    for i in range(1, 21):
        file_path = os.path.join(directory_path, f'{i}.jpg_labels.json')
        print(file_path)
        TP, FP, FN, MR = process_json_files(file_path)
        print("TP", TP, "FP", FP, "FN", FN, "MR", MR)
        Accuracy = get_Entity_Accuracy(TP, FP, FN, MR)
        Precision = get_Entity_Precision(TP, FP, MR)
        Recall = get_Entity_Recall(TP, FN, MR)
        F1 = get_Entity_F1(Precision, Recall)
        print("Accuracy", Accuracy, "Precision", Precision, "Recall", Recall, "F1", F1)
        entity_results.append({
            'file': f'{i}.jpg_labels.json',
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1
        })
    entity_results_df = pd.DataFrame(entity_results)
    entity_results_df.to_excel(output_file, index=False)
    print(f"Entity Analysis Results have been saved to {output_file}")

# Process score analysis and save results to Excel
def process_score_analysis(score_review_dir, output_file):
    output_file_scores = "SC_SD_Sequences.xlsx"
    extract_scores(score_review_dir, output_file_scores)

    dimensions = [
        "Realistic", "Deformation", "Imagination", "Color Richness",
        "Color Contrast", "Line Combination", "Line Texture",
        "Picture Organization", "Transformation"
    ]
    score_results = []

    scores_df = pd.read_excel(output_file_scores)
    for dimension in dimensions:
        dim_scores = scores_df[scores_df['dimension'] == dimension]
        original_scores = dim_scores['original']
        current_scores = dim_scores['current']

        # 计算SC和SD
        sc_value = get_sc(original_scores, current_scores)
        sd_value = get_sd(original_scores, current_scores)

        score_results.append({
            'dimension': dimension,
            'SC': sc_value,
            'SD': sd_value
        })

    score_results_df = pd.DataFrame(score_results)
    score_results_df.to_excel(output_file, index=False)
    print(f"Score Analysis Results have been saved to {output_file}")

# Process art style sensitivity (ASS) and save results to Excel
def process_style_analysis(folder_path, output_file):
    get_ass(folder_path, output_file)

# Process text analysis (TMR & TS) and save results to Excel
def process_text_analysis_main(score_comment_dir, suggestion_dir, output_file_tar_rev, output_file_tar_sug, output_file_ts_rev,output_file_ts_sug):
    process_text_analysis(score_comment_dir, suggestion_dir, output_file_tar_rev, output_file_tar_sug, output_file_ts_rev,output_file_ts_sug)

# Main function: call each analysis and save results
if __name__ == "__main__":
    # Entity analysis
    entity_directory = 'Dataset/Entities'
    entity_output_file = "Entity_Results.xlsx"
    process_entity_analysis(entity_directory, entity_output_file)

    # Score analysis
    score_review_directory = 'Dataset/score_Review'
    score_output_file = "SC_SD_Results.xlsx"
    process_score_analysis(score_review_directory, score_output_file)

    # Art style analysis
    style_directory = 'Dataset/Entities'
    style_output_file = "ASS_Results.xlsx"
    process_style_analysis(style_directory, style_output_file)

    # Text analysis (TAR & TS)
    score_comment_directory = 'Dataset/score_Review'
    suggestion_directory = 'Dataset/suggestion'
    output_file_tar_rev = "TMR_Results_rev.xlsx"
    output_file_tar_sug = "TMR_Results_sug.xlsx"
    output_file_ts_rev = "TS_Results_rev.xlsx"
    output_file_ts_sug = "TS_Results_sug.xlsx"
    process_text_analysis_main(score_comment_directory, suggestion_directory,
                      output_file_tar_rev, output_file_tar_sug,
                      output_file_ts_rev,output_file_ts_sug)

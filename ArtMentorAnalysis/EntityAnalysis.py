import os
import json

def process_json_files(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    E = json_data.get("original", [])
    R = json_data.get("added", [])
    W = json_data.get("removed", [])
    TP = len(E) - len(W)
    MR = min(len(W), len(R))
    FP = max(0, len(W) - MR)
    FN = max(0, len(R) - MR)
    return TP, FP, FN, MR

def get_Entity_Accuracy(TP, FP, FN, MR):
    Accuracy = TP / (TP + FP + FN + MR) if (TP + FP + FN + MR) > 0 else 0
    return Accuracy

def get_Entity_Precision(TP, FP, MR):
    Precision = TP / (TP + FP + MR) if (TP + FP + MR) > 0 else 0
    return Precision

def get_Entity_Recall(TP, FN, MR):
    Recall = TP / (TP + FN + MR) if (TP + FN + MR) > 0 else 0
    return Recall

def get_Entity_F1(Precision, Recall):
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    return F1
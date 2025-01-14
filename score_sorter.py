import os
import pandas as pd

def process_csv_files(input_folder, output_file):
    label_scores = {}

    # Iterate over all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('regr.csv'):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Combine the first two rows to create experiment names
            experiment_names = [
                f"{col1} + {col2}" if pd.notna(col2) else f"{col1}"
                for col1, col2 in zip(df.iloc[0], df.iloc[1])
            ]

            # Locate the 'score_average' row
            avg_score_row_index = df[df.iloc[:, 0].str.contains('score_average', na=False)].index
            if len(avg_score_row_index) == 0:
                continue

            avg_score_row = df.iloc[avg_score_row_index[0], 1:]  # Exclude the first column

            # Map scores to experiment names
            for experiment, score_avg in zip(experiment_names, avg_score_row):
                try:
                    if score_avg != 'nan':  # Skip string 'nan' explicitly
                        score_avg = float(score_avg)  # Ensure the score is numeric
                        if experiment not in label_scores:
                            label_scores[experiment] = 0
                        label_scores[experiment] += score_avg
                except ValueError:
                    print(f"Skipping non-numeric score in {experiment}")

        if filename.endswith('class.csv'):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Combine the first two rows to create experiment names
            experiment_names = [
                f"{col1} + {col2}" if pd.notna(col2) else f"{col1}"
                for col1, col2 in zip(df.iloc[0], df.iloc[1])
            ]

            # Locate the 'mcc' row
            mcc_score_row_index = df[df.iloc[:, 0].str.contains('mcc', na=False)].index

            if len(mcc_score_row_index) == 0:
                continue

            mcc_score_row = df.iloc[mcc_score_row_index[0], 1:]  # Exclude the first column

            # Map scores to experiment names
            for experiment, score_mcc in zip(experiment_names, mcc_score_row):
                try:
                    if score_mcc != 'nan':  # Skip string 'nan' explicitly
                        score_mcc = float(score_mcc)  # Ensure the score is numeric
                        if experiment not in label_scores:
                            label_scores[experiment] = 0
                        label_scores[experiment] += score_mcc
                except ValueError:
                    print(f"Skipping non-numeric score in {experiment}")

    # Convert the results to a DataFrame
    sorted_results = pd.DataFrame(
        list(label_scores.items()), columns=['Experiment', 'Total Score']
    )

    # Sort by the highest scores
    sorted_results = sorted_results.sort_values(by='Total Score', ascending=False)
    sorted_results = sorted_results[~sorted_results.isin(['nan']).any(axis=1)]

    # Save the sorted results to the output file
    sorted_results.to_csv(output_file, index=False)

# Specify input folder and output file
input_folder = r'C:\Users\cnhor\Downloads\pooling_exp_results'
output_file = r'C:\Users\cnhor\Downloads\pooling_exp_results\sorted_label_scores.csv'

process_csv_files(input_folder, output_file)

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

            # Ensure there are at least two rows for headers
            if df.shape[0] < 2:
                print(f"Skipping {filename}: Not enough rows for headers.")
                continue

            # Combine the first two rows to create experiment names
            experiment_names = [
                f"{col1}_{col2}" if pd.notna(col2) else f"{col1}"
                for col1, col2 in zip(df.iloc[0], df.iloc[1])
            ]
            # Remove '(', ')', and '_None' from experiment names
            experiment_names = [
                name.replace("(", "").replace(")", "").replace("_None", "").replace("'", "").replace(",", "_").rstrip("_").replace(" ", "")
                for name in experiment_names
            ]
            # Locate the 'score_average' row
            avg_score_row_index = df[df.iloc[:, 0].str.contains('score_average', na=False)].index
            if len(avg_score_row_index) == 0:
                print(f"Skipping {filename}: No 'score_average' row found.")
                continue

            avg_score_row = df.iloc[avg_score_row_index[0], 1:]  # Exclude the first column

            # Convert scores to a numeric format and rank them
            scores = []
            for score in avg_score_row:
                try:
                    scores.append(float(score) if score != 'nan' else None)
                except ValueError:
                    scores.append(None)

            ranks = pd.Series(scores).rank(ascending=False, method='min', na_option='bottom').tolist()

            # Map ranks to experiment names
            for experiment, rank in zip(experiment_names, ranks):
                if rank is not None:  # Skip None values
                    if experiment not in label_scores:
                        label_scores[experiment] = 0
                    label_scores[experiment] += rank
                
        if filename.endswith('class.csv'):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Ensure there are at least two rows for headers
            if df.shape[0] < 2:
                print(f"Skipping {filename}: Not enough rows for headers.")
                continue

            # Combine the first two rows to create experiment names
            experiment_names = [
                f"{col1}_{col2}" if pd.notna(col2) else f"{col1}"
                for col1, col2 in zip(df.iloc[0], df.iloc[1])
            ]
            # Remove '(', ')', and '_None' from experiment names
            experiment_names = [
                name.replace("(", "").replace(")", "").replace("_None", "").replace("'", "").replace(",", "_").rstrip("_").replace(" ", "")
                for name in experiment_names
            ]

            # Locate the 'score_average' row
            mcc_score_row_index = df[df.iloc[:, 0].str.contains('mcc', na=False)].index
            if len(mcc_score_row_index) == 0:
                print(f"Skipping {filename}: No 'mcc_average' row found.")
                continue

            mcc_score_row = df.iloc[mcc_score_row_index[0], 1:]  # Exclude the first column

            # Convert scores to a numeric format and rank them
            scores = []
            for score in mcc_score_row:
                try:
                    scores.append(float(score) if score != 'nan' else None)
                except ValueError:
                    scores.append(None)

            ranks = pd.Series(scores).rank(ascending=False, method='min', na_option='bottom').tolist()

            # Map ranks to experiment names
            for experiment, rank in zip(experiment_names, ranks):
                if rank is not None:  # Skip None values
                    if experiment not in label_scores:
                        label_scores[experiment] = 0
                    label_scores[experiment] += rank



    # Convert the results to a DataFrame
    sorted_results = pd.DataFrame(
        list(label_scores.items()), columns=['Experiment', 'Total Rank Score']
    )

    # Sort by the lowest total rank score (higher rank)
    sorted_results = sorted_results.sort_values(by='Total Rank Score')

    sorted_results = sorted_results[~sorted_results.isin(['nan']).any(axis=1)]

    # Save the sorted results to the output file
    sorted_results.to_csv(output_file, index=False)

# Specify input folder and output file
input_folder = r'C:\Users\cnhor\Documents\GitHub\pooling_exp\results\pooling_exp_results_fastesm650'
output_file = r'C:\Users\cnhor\Documents\GitHub\pooling_exp\results\pooling_exp_results_fastesm650\FastESM650_Total_Rank_Scores.csv'

process_csv_files(input_folder, output_file)

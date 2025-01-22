import os
import pandas as pd

def process_csv_files(input_folder, output_file):
    experiment_ranks = {}

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

            scores = df.iloc[4].tolist()  # Fifth row (scores)

            # Rank the scores
            ranks = pd.Series(scores).rank(ascending=False, method='min').tolist()

            # Update cumulative ranks
            for experiment, rank in zip(experiment_names, ranks):
                if experiment not in experiment_ranks:
                    experiment_ranks[experiment] = 0
                experiment_ranks[experiment] += rank

            final_ranks_df = pd.DataFrame(list(experiment_ranks.items()), columns=['Experiment', 'Cumulative Rank'])
            final_ranks_df = final_ranks_df.sort_values(by='Cumulative Rank')

                
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
                f"{row1}_{row2}" if pd.notna(row2) else f"{row1}"
                for row1, row2 in zip(df.iloc[0], df.iloc[1])
            ]
            # Remove '(', ')', and '_None' from experiment names
            experiment_names = [
                name.replace("(", "").replace(")", "").replace("_None", "").replace("'", "").replace(",", "_").rstrip("_").replace(" ", "")
                for name in experiment_names
            ]

            scores = df.iloc[2].tolist()  # Second row (scores)

            # Rank the scores
            ranks = pd.Series(scores).rank(ascending=False, method='min').tolist()

            # Update cumulative ranks
            for experiment, rank in zip(experiment_names, ranks):
                if experiment not in experiment_ranks:
                    experiment_ranks[experiment] = 0
                experiment_ranks[experiment] += rank

            final_ranks_df = pd.DataFrame(list(experiment_ranks.items()), columns=['Experiment', 'Cumulative Rank'])
            final_ranks_df = final_ranks_df.sort_values(by='Cumulative Rank')

    # Convert the results to a DataFrame
    sorted_results = pd.DataFrame(
        list(experiment_ranks.items()), columns=['Experiment', 'Total Rank Score']
    )

    # Sort by the lowest total rank score (higher rank)
    sorted_results = sorted_results.sort_values(by='Total Rank Score')

    sorted_results = sorted_results[~sorted_results.isin(['nan']).any(axis=1)]

    # Save the sorted results to the output file
    sorted_results.to_csv(output_file, index=False)

# Specify input folder and output file
input_folder = r'C:\Users\cnhor\Documents\GitHub\pooling_exp\results\sum_product'
output_file = r'C:\Users\cnhor\Documents\GitHub\pooling_exp\results\sum_product\ESM2-650M_Total_Rank_Scores.csv'

process_csv_files(input_folder, output_file)

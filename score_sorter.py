import os
import pandas as pd


def process_csv_files(input_folder, output_file):
    experiment_ranks = {}
    experiment_r2_scores = {}
    experiment_mcc_scores = {}
    experiment_r2_counts = {}
    experiment_mcc_counts = {}

    # Iterate over all CSV files in the input folder
    print(os.listdir(input_folder))
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
            # Clean up experiment names
            experiment_names = [
                name.replace("(", "").replace(")", "").replace("_None", "").replace("'", "").replace(",", "_").rstrip("_").replace(" ", "")
                for name in experiment_names
            ]

            scores = df.iloc[4].tolist()  # Fifth row (scores)

            # Rank the scores
            ranks = pd.Series(scores).rank(ascending=False, method='min').tolist()

            # Update cumulative ranks, R2-Pearson scores, and counts
            for experiment, rank, score in zip(experiment_names, ranks, scores):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    continue  # Skip invalid scores

                if experiment not in experiment_ranks:
                    experiment_ranks[experiment] = 0
                    experiment_r2_scores[experiment] = 0.0
                    experiment_r2_counts[experiment] = 0
                experiment_ranks[experiment] += rank
                experiment_r2_scores[experiment] += score
                experiment_r2_counts[experiment] += 1

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
            # Clean up experiment names
            experiment_names = [
                name.replace("(", "").replace(")", "").replace("_None", "").replace("'", "").replace(",", "_").rstrip("_").replace(" ", "")
                for name in experiment_names
            ]

            scores = df.iloc[2].tolist()  # Second row (scores)

            # Rank the scores
            ranks = pd.Series(scores).rank(ascending=False, method='min').tolist()

            # Update cumulative ranks, MCC scores, and counts
            for experiment, rank, score in zip(experiment_names, ranks, scores):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    continue  # Skip invalid scores

                if experiment not in experiment_ranks:
                    experiment_ranks[experiment] = 0
                    experiment_mcc_scores[experiment] = 0.0
                    experiment_mcc_counts[experiment] = 0
                experiment_ranks[experiment] += rank
                experiment_mcc_scores[experiment] += score
                experiment_mcc_counts[experiment] += 1

    # Calculate averages
    experiment_r2_averages = {
        experiment: experiment_r2_scores[experiment] / experiment_r2_counts[experiment]
        for experiment in experiment_r2_scores
    }
    experiment_mcc_averages = {
        experiment: experiment_mcc_scores[experiment] / experiment_mcc_counts[experiment]
        for experiment in experiment_mcc_scores
    }

    # Combine results into a DataFrame
    sorted_results = pd.DataFrame({
        'Experiment': experiment_ranks.keys(),
        'Total Rank Score': experiment_ranks.values(),
        'Average R2-Pearson Score': [experiment_r2_averages.get(exp, None) for exp in experiment_ranks.keys()],
        'Average MCC Score': [experiment_mcc_averages.get(exp, None) for exp in experiment_ranks.keys()],
    })

    # Sort by the lowest total rank score
    sorted_results = sorted_results.sort_values(by='Total Rank Score')

    # Save the sorted results to the output file
    sorted_results.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='results', type=str)
    parser.add_argument('--output_file', default='results', type=str)
    args = parser.parse_args()

    process_csv_files(args.input_folder, args.output_file)
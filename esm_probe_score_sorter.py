import os
import pandas as pd
import ast

def organize_metrics(folder_path, output_csv):
    """
    Organizes metrics from .tsv files in the specified folder and saves the results to a CSV file.

    Parameters:
        folder_path (str): Path to the folder containing .tsv files.
        output_csv (str): Path to save the resulting CSV file.
    """
    all_data = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(folder_path, file_name)
            # Extract pooling method from the file name by taking everything after the first underscore and before the file extension
            pooling_method = file_name.split('_', 1)[-1].rsplit('.', 1)[0]
            
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file_path, sep='\t')

            for _, row in df.iterrows():
                dataset = row['dataset']
                for column, metrics in row.items():
                    if column != 'dataset':
                        # Parse the metrics string into a dictionary
                        metrics_dict = ast.literal_eval(metrics)
                        # Add the dataset, model name, and pooling method to the dictionary
                        metrics_dict['dataset'] = dataset
                        metrics_dict['model'] = column
                        metrics_dict['pooling_method'] = pooling_method
                        all_data.append(metrics_dict)

    # Create a DataFrame from the collected data
    metrics_df = pd.DataFrame(all_data)

    # Sort the DataFrame by dataset and pooling method
    metrics_df.sort_values(by=['dataset', 'pooling_method'], inplace=True)

    # Save the DataFrame to a CSV file
    metrics_df.to_csv(output_csv, index=False)

input_path = r"C:\Users\cnhor\Documents\GitHub\pooling_exp\results\esm_probe_results"
output_name = "esm_probe_results.csv"

# Example usage
organize_metrics(input_path, output_name)

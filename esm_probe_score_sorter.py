import os
import pandas as pd
import ast

def organize_metrics(folder_path, output_csv, classification=True):
    """
    Organizes metrics from .tsv files in the specified folder. If classification is True, it outputs a CSV file
    containing the averaged MCC scores for all classification datasets. If classification is False, it outputs a CSV
    file containing only the R² scores for the enzyme_kcat dataset.

    Parameters:
        folder_path (str): Path to the folder containing .tsv files.
        output_csv (str): Path to save the resulting CSV file.
        classification (bool): If True, outputs averaged MCC scores for classification datasets; if False, outputs only
                               R² scores for enzyme_kcat.
    """
    classification_data = []
    regression_data = []

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
                        # Add dataset, model name, and pooling method to the dictionary
                        metrics_dict['model'] = column
                        metrics_dict['pooling_method'] = pooling_method
                        
                        if dataset == 'enzyme_kcat':
                            if 'eval_r_squared' in metrics_dict:
                                regression_data.append({
                                    'model': column,
                                    'pooling_method': pooling_method,
                                    'r2_score': metrics_dict['eval_r_squared']
                                })
                        else:
                            if 'eval_mcc' in metrics_dict:
                                classification_data.append({
                                    'model': column,
                                    'pooling_method': pooling_method,
                                    'mcc_score': metrics_dict['eval_mcc']
                                })
    
    if classification:
        # Process and save classification dataset averages (only MCC scores)
        class_df = pd.DataFrame(classification_data)
        class_grouped_df = class_df.groupby(['model', 'pooling_method']).mean().reset_index()
        class_grouped_df.to_csv(output_csv, index=False)
    else:
        # Process and save enzyme_kcat dataset (only R² scores)
        enzyme_df = pd.DataFrame(regression_data)
        enzyme_df.to_csv(output_csv, index=False)

input_path = r"C:\Users\cnhor\Documents\GitHub\pooling_exp\results\esm_probe_results"
output_name = "esm_probe_results.csv"

# Example usage
organize_metrics(input_path, output_name, classification=False)

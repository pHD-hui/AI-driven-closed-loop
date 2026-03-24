import pandas as pd
import os


def extract_features_and_merge(fit_params_path, process_data_path, output_path):
    """
        Extract features from fitted parameters and process data, then merge them.

        Parameters:
            fit_params_path: Path to fitted parameter CSV file
            process_data_path: Path to process data Excel file
            output_path: Output file path

        Returns:
            Merged DataFrame
        """
    # 1. Read data
    fit_params = pd.read_csv(fit_params_path)
    process_data = pd.read_excel(process_data_path)

    # 2. Check whether the output file already exists
    if os.path.exists(output_path):
        existing_data = pd.read_excel(output_path)
        last_exp_id = existing_data['Experiment ID'].max()

        # Prepare process parameter column names
        # (exclude experiment ID and feature columns)
        process_cols = [col for col in process_data.columns if col not in ['Experiment ID']]
        feature_cols = ['Maximum Peak Height', 'Maximum Peak Height_std']
        all_cols = process_cols + feature_cols

        # Check duplicate processes
        # (based on process parameters only, ignoring experiment ID and feature values)
        def create_process_signature(row):
            return tuple(row[col] for col in process_cols)

        # Get process signatures from existing data
        existing_signatures = set(existing_data[process_cols].apply(create_process_signature, axis=1))

        # Filter out duplicate processes
        new_process_data = process_data[
            ~process_data[process_cols].apply(create_process_signature, axis=1).isin(existing_signatures)
        ]

        # If there are no new processes to add, return existing data directly
        if new_process_data.empty:
            print("All processes already exist. Proceeding directly to optimization.")
            return existing_data

        # Reassign experiment IDs for new data
        # (starting from last_exp_id + 1)
        new_process_data = new_process_data.copy()
        new_process_data['Experiment ID'] = range(last_exp_id + 1, last_exp_id + 1 + len(new_process_data))
    else:
        existing_data = None
        new_process_data = process_data.copy()
        new_process_data['Experiment ID'] = range(1, 1 + len(new_process_data))

    # 3. Initialize feature dictionary
    features = {}
    for exp_id in new_process_data['Experiment ID']:
        features[exp_id] = {
            'Maximum Peak Height': None,
            'Maximum Peak Height_std': None,
        }

    # 4. Group by experiment ID for processing
    # (using the original experiment ID in fit_params)
    # Assume experiment IDs in fit_params are continuously numbered starting from 1
    grouped = fit_params.groupby(fit_params.columns[0])

    for exp_id, group in grouped:
        # Only process experiment IDs that exist in new_process_data
        # Here it is assumed that the experiment IDs in fit_params
        # are consistent with those in new_process_data
        if exp_id in features:
            # Extract peak-height-related features
            peak_cols = [col for col in group.columns if 'Peak Height' in col]

            # Calculate maximum peak height
            max_peak_value = group[peak_cols].max().max()
            features[exp_id]['Maximum Peak Height'] = max_peak_value

            # Extract the standard deviation corresponding to the maximum peak
            max_peak_pos = group[peak_cols].stack().idxmax()
            peak_num = max_peak_pos[1][-1]
            std_col = f'Stddev{peak_num}'
            features[exp_id]['Maximum Peak Height_std'] = group.loc[max_peak_pos[0], std_col]

            # Add future features here if needed

    # 5. Convert to DataFrame and merge
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = 'Experiment ID'
    merged_df = new_process_data.merge(features_df, on='Experiment ID', how='left')

    # 6. Save results (append or create new)
    if existing_data is not None:
        # Ensure there are no duplicate experiment IDs
        final_df = pd.concat([existing_data, merged_df], ignore_index=True)
    else:
        final_df = merged_df

    final_df.to_excel(output_path, index=False)
    return final_df


# Usage example
if __name__ == "__main__":
    # Basic usage
    df = extract_features_and_merge(
        fit_params_path="fitted_parameters_round1.csv",
        process_data_path="bayesian_optimization/data_backup/initial_data.xlsx",
        output_path="initial_data_with_features.xlsx"
    )
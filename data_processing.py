import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def analyze_spectral_uniformity(file_path,
                                piece_each_process: int,
                                show_plots=True,
                                save_results=True,
                                save_path="representative_spectra"):
    """
    Analyze spectral data uniformity and identify outliers

    Parameters:
        file_path (str): Path to the CSV file
        piece_each_process (int): Number of samples produced by one process: 1/2
        show_plots (bool): Whether to display analysis plots (default True)
        save_results (bool): Whether to save result files (default True)
        save_path (str): Path to save the result file

    Returns:
        dict: A dictionary containing analysis results, including:
            - 'cleaned_data': cleaned spectral data
            - 'representative_spectrum': representative spectrum
            - 'uniformity_scores': uniformity score of each group
            - 'avg_uniformity': average uniformity
            - 'cv_by_wavelength': coefficient of variation at each wavelength

    File outputs:
        representative_spectrum.csv: representative spectral data for all groups
        spectral_analysis_result: result plots
            (Figure 1 and Figure 2: data cleaning example using the first group;
             Figure 3: uniformity scores of all groups)
        uniformity_scores: uniformity scores
    """

    # 1. Data loading and preprocessing
    def load_and_preprocess_data(file_path):
        """
        Load and preprocess spectral data
        wavelengths: wavelengths of light
        sample_data: original intensity data
        grouped_data: grouped data (4 samples per group)
        """
        data = pd.read_csv(file_path, index_col=0)
        wavelengths = data.columns.astype(int)
        sample_data = data.values
        grouped_data = np.array(np.split(sample_data, len(sample_data) // 4))
        index_data = np.array(data.index[i*4] for i in range(len(sample_data) // 4))
        return wavelengths, sample_data, grouped_data, index_data

    # 2. Outlier detection and processing
    def detect_and_clean_outliers(grouped_data: np.ndarray):
        """Detect and process outliers"""

        def process_group(group):
            median = np.median(group, axis=0)  # median
            q1 = np.percentile(group, 25, axis=0)  # 25th percentile
            q3 = np.percentile(group, 75, axis=0)  # 75th percentile
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            cleaned_group = group.copy()  # deep copy
            for i in range(group.shape[0]):
                outlier_mask = (group[i] < lower_bound) | (group[i] > upper_bound)
                cleaned_group[i, outlier_mask] = median[outlier_mask]

            return cleaned_group

        return np.vstack([process_group(group) for group in grouped_data])

    # 3. Uniformity evaluation
    def calculate_uniformity_metrics(cleaned_data, grouped_data_shape):
        """Calculate uniformity metrics
            Use coefficient of variation to measure the dispersion of four spectra
        """
        # regroup cleaned data
        cleaned_groups = np.array(np.split(cleaned_data, len(cleaned_data) // 4))

        # calculate group uniformity
        def group_uniformity(group):
            # We use the coefficient of variation to evaluate the dispersion of four points,
            # then determine the average variation coefficient of each spectrum line in one group.
            # This is used to evaluate the uniformity of this sample:
            # the more uniform => the lower the dispersion
            std_dev = np.std(group, axis=0)  # calculate std of each column
            mean_val = np.mean(group, axis=0)  # calculate mean of each column
            cv = np.where(mean_val != 0, std_dev / mean_val, 0)
            return 1 - np.mean(cv)

        uniformities = [group_uniformity(group) for group in cleaned_groups]

        # calculate representative spectrum (median of each group)
        representative_spectra = np.median(cleaned_groups, axis=1)

        # calculate coefficient of variation at each wavelength
        cv_by_wavelength = np.std(cleaned_data, axis=0) / np.mean(cleaned_data, axis=0)

        return {
            'uniformity_scores': uniformities,
            'avg_uniformity': np.mean(uniformities),
            'representative_spectrum': representative_spectra,  # group medians
            'cv_by_wavelength': cv_by_wavelength
        }

    # 4. Process quality evaluation
    def evaluate_process_quality(metrics):
        uniformity_scores = metrics["uniformity_scores"]
        data_processes = np.split(
            metrics['representative_spectrum'],
            len(metrics['representative_spectrum']) // piece_each_process
        )  # split data assuming two samples per process
        index_processes = np.array([i // piece_each_process for i in range(len(uniformity_scores))])
        delete_index = []  # indices to be removed
        length_of_ws = len(data_processes[0][0])

        for process in range(len(data_processes)):
            # if data is missing, marked as 0, then remove it
            if piece_each_process == 2:
                # only used when there are two samples
                frist_piece_nan = (data_processes[process][0][0] == 0)  # first sample missing
                second_piece_nan = (data_processes[process][1][0] == 0)  # second sample missing
                frist_piece_nonuniform = (uniformity_scores[process * 2] < 0.7)  # first sample non-uniform
                second_piece_nonuniform = (uniformity_scores[process * 2 + 1] < 0.7)  # second sample non-uniform

                if frist_piece_nan or second_piece_nan:
                    # one sample in the group is missing
                    if frist_piece_nan and second_piece_nan:
                        # remove both directly
                        data_processes[process] = np.array([[]])
                        delete_index.append(process * 2)
                        delete_index.append(process * 2 + 1)
                    elif frist_piece_nan:
                        data_processes[process] = data_processes[process][1].reshape((1, length_of_ws))
                        delete_index.append(process * 2)
                    elif second_piece_nan:
                        data_processes[process] = data_processes[process][0].reshape((1, length_of_ws))
                        delete_index.append(process * 2 + 1)

                # remove those that are not uniform enough
                if frist_piece_nonuniform or second_piece_nonuniform:
                    # first discuss the case where both are non-uniform
                    if frist_piece_nonuniform and second_piece_nonuniform:
                        # replace with mean, remove the later one
                        data_processes[process] = (
                            np.mean(
                                [data_processes[process][0], data_processes[process][1]],
                                axis=0
                            ).reshape(1, length_of_ws)
                        )
                        delete_index.append(process * 2)
                    elif frist_piece_nonuniform:
                        data_processes[process] = data_processes[process][1].reshape((1, length_of_ws))
                        delete_index.append(process * 2)
                    elif second_piece_nonuniform:
                        data_processes[process] = data_processes[process][0].reshape((1, length_of_ws))
                        delete_index.append(process * 2 + 1)
            else:
                # with only one sample, remove directly if not uniform enough
                if uniformity_scores[process] < 0.7:
                    delete_index.append(process)

        if piece_each_process == 1:
            # when only one sample is made, delete separately
            data_processes = np.delete(np.array(data_processes), delete_index, axis=0)

        # processing the remaining data is the same for one or two samples
        uniformity_scores_cleaned = np.delete(uniformity_scores, delete_index)
        index_processes = np.delete(index_processes, delete_index)
        return {
            'processes_data': data_processes,  # dtype: list
            'cleaned_uniformity': np.array(uniformity_scores_cleaned),
            'index_processes': np.array(index_processes)  # used together with cleaned_uniformity to get average uniformity of each sample
        }

    # 5. Visualization
    def create_visualizations(wavelengths, original_data, cleaned_data, metrics, processes):
        """Create analysis plots"""
        plt.figure(figsize=(18, 12))
        # adjust subplot spacing
        plt.subplots_adjust(
            left=0.062,
            right=0.985,
            bottom=0.07,
            top=0.959,
            wspace=0.168,
            hspace=0.252
        )

        # Figure 1: comparison between original and cleaned data (first group)
        plt.subplot(2, 2, 1)
        for i in range(4):
            plt.plot(wavelengths, original_data[i], alpha=0.5, label=f'Original {i + 1}')
        for i in range(4):
            plt.plot(wavelengths, cleaned_data[i], '--', alpha=0.8, label=f'Cleaned {i + 1}')
        plt.title('Original vs Cleaned Spectra (Example: First Group)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

        # Figure 2: error display (using Figure 1 as an example), 25% and 75% percentiles
        plt.subplot(2, 2, 2)
        # original data - percentile shading
        q1_original = np.percentile(original_data[:4], 25, axis=0)
        q3_original = np.percentile(original_data[:4], 75, axis=0)
        plt.fill_between(wavelengths, q1_original, q3_original,
                         color='blue', alpha=0.15, label='Original IQR')
        # cleaned data - error bars
        mean_cleaned = np.mean(cleaned_data[:4], axis=0)
        std_cleaned = np.std(cleaned_data[:4], axis=0)
        plt.errorbar(wavelengths[::10], mean_cleaned[::10], yerr=std_cleaned[::10],
                     fmt='ro', markersize=4, capsize=3, label='Cleaned Mean±Std')
        # overlay median and mean lines
        plt.plot(wavelengths, np.median(original_data[:4], axis=0), 'b-',
                 linewidth=1, alpha=0.7, label='Original Median')
        plt.plot(wavelengths, mean_cleaned, 'r--',
                 linewidth=1, label='Cleaned Mean')
        plt.title('Hybrid Visualization: IQR vs Error Bars')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # Figure 3: uniformity metric distribution
        plt.subplot(2, 2, 3)
        # color setting: >=0.7 acceptable (blue-green), <0.7 unacceptable (orange-red)
        colors = ['#3498db' if score >= 0.7 else '#e74c3c' for score in metrics['uniformity_scores']]
        plt.bar(
            range(1, len(metrics['uniformity_scores']) + 1),
            [metrics['uniformity_scores'][idx] if metrics['uniformity_scores'][idx] != 1 else 0
             for idx in range(len(metrics['uniformity_scores']))],
            color=colors
        )
        plt.axhline(y=metrics['avg_uniformity'], color='r', linestyle='--',
                    label=f'Average: {metrics["avg_uniformity"]:.3f}')
        plt.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        plt.axhspan(0.7, 1.0, alpha=0.1, hatch='//')
        plt.title('Uniformity Index for Each Sample Group')
        plt.xlabel('Sample Group')
        plt.ylabel('Uniformity Index (0-1)')
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        plt.grid(True)

        # Figure 4: all spectra collection
        plt.subplot(2, 2, 4)
        peak_intensity = []  # list of maximum intensity for each fabrication method
        processes_data = processes["processes_data"]
        index_process = processes["index_processes"]
        uniformity_process = processes["cleaned_uniformity"]
        avg_cleaned_uniformity = []  # average uniformity of each fabrication method

        for process in range(len(processes_data)):
            peak_intensity.append(np.max(processes_data[process]))
            avg_cleaned_uniformity.append(
                np.mean(uniformity_process[np.where(index_process == process)])
            )
        x = np.arange(len(peak_intensity))

        # set color mapping (from red to green indicates better overall quality)
        avg_cleaned_uniformity = np.array(avg_cleaned_uniformity)
        peak_intensity = np.array(peak_intensity)

        mask = avg_cleaned_uniformity > 0.4  # filter out extremely low-intensity points
        filtered_uniformity = np.array(avg_cleaned_uniformity)[mask]
        filtered_intensity = np.array(peak_intensity)[mask]

        # dynamically calculate point sizes (excluding minimum values)
        min_size = 50
        max_size = 500
        sizes = np.interp(
            filtered_intensity,
            (np.min(filtered_intensity), np.max(filtered_intensity)),
            (min_size, max_size)
        )

        # color mapping (from red to green)
        quality_score = filtered_uniformity * filtered_intensity / np.max(filtered_intensity)
        cmap = plt.cm.get_cmap('RdYlGn')

        # scatter plot
        sc = plt.scatter(
            x=filtered_uniformity,
            y=filtered_intensity,
            c=quality_score,
            cmap=cmap,
            s=sizes,
            alpha=0.8,
            edgecolors='k',
            linewidths=0.8,
            zorder=3
        )

        # axis settings (log scale)
        plt.yscale('log')
        plt.xlim(0.3, 1.0)
        plt.ylim(np.min(filtered_intensity) * 0.9, np.max(filtered_intensity) * 1.3)

        # reference line
        plt.axvline(0.7, color='gray', linestyle=':', alpha=0.5)

        # colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Quality Score\n(Uniformity × Normalized Intensity)',
                       rotation=270, labelpad=20)

        # labels and title
        plt.title('Spectral Quality Analysis (Filtered)\n[Size ∝ Intensity]', pad=20, fontsize=14)
        plt.xlabel('Average Uniformity Score (0-1)', fontsize=12)
        plt.ylabel('Peak Intensity (log scale)', fontsize=12)

        # grid and style optimization
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.gca().set_facecolor('#f8f8f8')

        # highlight the best-quality point
        max_idx = np.argmax(quality_score)
        plt.scatter(filtered_uniformity[max_idx], filtered_intensity[max_idx],
                    s=max_size * 1.2, facecolors='none', edgecolors='gold',
                    linewidths=2, label='Best Quality')
        plt.legend(loc='upper left')

        if save_results:
            plt.savefig('spectral_analysis_results.svg', dpi=300, format='svg')
        if show_plots:
            plt.show()
        else:
            plt.close()

    # Main execution workflow
    wavelengths, original_data, grouped_data, index_data = load_and_preprocess_data(file_path)
    cleaned_data = detect_and_clean_outliers(grouped_data)
    metrics = calculate_uniformity_metrics(cleaned_data, grouped_data.shape)
    processes = evaluate_process_quality(metrics)

    if show_plots:
        create_visualizations(wavelengths, original_data, cleaned_data, metrics, processes)

    # Save results
    if save_results:
        # Save all retained spectra after filtering
        flattened = np.concatenate([arr.reshape(-1) for arr in processes["processes_data"]])
        indexs = [p + 1 for p in processes["index_processes"]]
        len_wavelength = len(wavelengths)
        representative_spectrum = flattened.reshape(len(flattened) // len_wavelength, len_wavelength)
        result_df = pd.DataFrame(
            representative_spectrum,
            columns=wavelengths,
            index=indexs
        )
        result_df.to_csv(save_path)

        # Save uniformity scores
        uniformity_df = pd.DataFrame({
            'Group': range(1, len(metrics['uniformity_scores']) + 1),
            'Uniformity_Score': metrics['uniformity_scores']
        })
        uniformity_df.to_csv('uniformity_scores.csv', index=False)

    return {
        'wavelengths': wavelengths,
        'original_data': original_data,
        'cleaned_data': cleaned_data,
        'representative_spectrum': metrics['representative_spectrum'],
        'uniformity_scores': metrics['uniformity_scores'],
        'avg_uniformity': metrics['avg_uniformity'],
        'cv_by_wavelength': metrics['cv_by_wavelength'],
        'processes_data': processes['processes_data'],  # dtype: list
        'cleaned_uniformity': processes['cleaned_uniformity'],
        'index_processes': processes['index_processes']  # indicator, used together with cleaned_uniformity to calculate average uniformity of each sample
    }


# Usage example
if __name__ == "__main__":
    results = analyze_spectral_uniformity('data/validation_data.csv', piece_each_process=1)

    # Print key results
    print(f"\nAnalysis completed, average uniformity score: {results['avg_uniformity']:.3f}")
    result = np.array(results['uniformity_scores'])
    filter = result[result != 1]
    print(f"Best uniformity group: {np.argmax(results['uniformity_scores']) + 1} (score: {filter.max():.3f})")
    print(f"Worst uniformity group: {np.argmin(results['uniformity_scores']) + 1} (score: {filter.min():.3f})")
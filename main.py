import pytz
from data_processing import analyze_spectral_uniformity
from get_features import extract_features_and_merge
from BO_function import bayesian_optimization_and_suggest
from guass_function import guassFitData



if __name__ == "__main__":
    # Configuration parameters: number of Gaussian peaks and samples per process
    guass_number = 2
    piece_each_process = 2

    # Input file paths
    spectrum_path_origin = 'data/First_round_data-PL(raw).csv'  # Raw spectral data input
    process_data_path = "data/initial_data_R2.xlsx"  # Process parameter input file

    # Intermediate data storage paths
    spectrum_path_processed = "representative_spectra.csv"  # Store spectra of each sample
    fit_params_path = "fitted_parameters_round1.csv"  # Spectral fitting parameters
    input_of_bayesianOpt = "initial_data_with_features.xlsx"  # Input for Bayesian optimization

    # Output result path
    final_output_path = "BO_Round1.xlsx"  # Final output results

    # Main workflow
    # 1. Data processing
    processing_datas = analyze_spectral_uniformity(
        file_path=spectrum_path_origin,
        piece_each_process=piece_each_process,
        show_plots=True,
        save_results=True,
        save_path=spectrum_path_processed
    )

    # 2. Spectral fitting
    finial_params = guassFitData(
        guass_number,
        spectrum_path_processed,
        set_bounds="default",
        save_params_path=fit_params_path,
        save_and_view_training_path=None
    )

    # 3. Extract spectral features and merge with process parameters
    df = extract_features_and_merge(
        fit_params_path=fit_params_path,
        process_data_path=process_data_path,
        output_path=input_of_bayesianOpt
    )

    # 4. Bayesian optimization to generate final results
    bayesian_optimization_and_suggest(
        filepath=input_of_bayesianOpt,
        savepath=final_output_path,
        target_params="Maximum Peak Height",
        kind='ucb',
        kappa=10,
        n_points=20,
    )

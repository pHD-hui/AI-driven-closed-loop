from bayes_opt import BayesianOptimization, UtilityFunction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(114514)


def bayesian_optimization_and_suggest(
        filepath,
        savepath,
        target_params="Maximum Peak Height",
        kind='ucb',
        kappa=10,
        n_points=5,
        param_config=None,
        recent_influence=0.5,
        visualize=False,
        **kwargs
):

    """
    Parameters:
        filepath (str): Path to historical data file (Excel format)
        savepath (str): Path to save suggested parameters (Excel format)
        kind (str): Type of utility function, default 'ucb'
        kappa (float): Exploration coefficient, default 10
        n_points (int): Number of suggested points, default 5
        param_config (None or dict): Parameter space configuration (e.g., bounds, step size)
        recent_influence:
            0 - Treat all historical data equally
            0.5 - Moderate influence of recent data (default)
            1 - Recent data dominates suggestion generation
        visualize: Whether to display optimization progress plot
    """

    # Load data and add temporal weights
    df = pd.read_excel(filepath).sort_values('Maximum Peak Height')
    df['Time Weight'] = np.linspace(recent_influence, 1, len(df))

    # Default parameter configuration
    default_config = {
        "Precursor Volume/ul": {"bounds": (10, 60), "step": 5, "dtype": "int"},
        "Spin Speed/rpm": {"bounds": (1000, 5000), "step": 100, "dtype": "int"},
        "Spin Time/s": {"bounds": (10, 60), "step": 5, "dtype": "int"},
        "Spin Acceleration": {"bounds": (1000, 5000), "step": 100, "dtype": "int"},
        "Annealing Time/min": {"bounds": (1, 10), "step": 1, "dtype": "int"},
        "Additive": {"bounds": (1, 12), "step": 1, "dtype": "int"},
        "Additive Amount%": {"bounds": (5, 20), "step": 5, "dtype": "int"}
    }
    param_config = param_config or default_config

    # Data validation
    missing = [p for p in param_config if p not in df.columns]
    if missing:
        raise ValueError(f"Missing parameter columns: {missing}")
    if target_params not in df.columns:
        raise ValueError(f"Target column '{target_params}' does not exist")

    # Normalize target values
    scaler = StandardScaler()
    df['Normalized Target'] = scaler.fit_transform(df[[target_params]]) if not df[target_params].isnull().all() else 0

    # Initialize optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds={k: v["bounds"] for k, v in param_config.items()},
        verbose=0,
        random_state=42
    )
    for _, row in df.dropna(subset=[target_params]).iterrows():
        optimizer.register(
            params={k: row[k] for k in param_config.keys()},
            target=float(row[target_params])
        )

    # Weighted acquisition function
    def weighted_acquisition(x):
        x_array = np.array([x[p] for p in param_config.keys()]).reshape(1, -1)
        utility = UtilityFunction(kind=kind, kappa=kappa, xi=0.01)
        base_ei = utility.utility(x_array, optimizer._gp, optimizer._space.target.min())

        if recent_influence > 0:
            recent_data = df.iloc[-int(len(df) * 0.3):]
            if len(recent_data) > 0:
                distances = np.mean([
                    np.linalg.norm(x_array - row[list(param_config.keys())].values)
                    for _, row in recent_data.iterrows()
                ], axis=0)
                recent_weight = 1 / (1 + distances) * recent_influence
                return float(base_ei * (1 + recent_weight))
        return float(base_ei)

    # Generate suggestion points (key modification)
    suggestions = []
    param_names = list(param_config.keys())

    # Preprocess historical parameters
    existing = set()
    for p in optimizer._space.params:
        aligned_params = {}
        for param in param_names:
            config = param_config[param]
            raw_val = p[param]
            step = config["step"]

            # Alignment processing
            aligned_val = round(raw_val / step) * step
            aligned_val = np.clip(aligned_val, *config["bounds"])

            # Type conversion
            if config["dtype"] == "int":
                aligned_val = int(aligned_val)
            else:
                aligned_val = float(aligned_val)

            aligned_params[param] = aligned_val
        existing.add(tuple(aligned_params[param] for param in param_names))

    bounds = [v["bounds"] for v in param_config.values()]

    while len(suggestions) < n_points:
        try:
            # Generate intelligent initial point
            best_idx = df[target_params].idxmax()
            best_params = df.iloc[best_idx][param_names].values

            # Preprocess initial point
            x0 = []
            for i, (param, config) in enumerate(param_config.items()):
                step = config["step"]
                raw_val = best_params[i]
                aligned_val = round(raw_val / step) * step
                aligned_val = np.clip(aligned_val, *config["bounds"])
                x0.append(aligned_val)
            x0 = np.array(x0) * (1 + 0.1 * np.random.randn(len(bounds)))

            # Optimize acquisition function
            res = minimize(
                lambda x: -weighted_acquisition(dict(zip(param_names, x))),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-4}
            )

            # Post-process parameters
            aligned = {}
            for i, (param, config) in enumerate(param_config.items()):
                step = config["step"]
                raw_val = res.x[i]

                # Alignment processing
                val = round(raw_val / step) * step
                val = np.clip(val, *config["bounds"])

                # Enforce type conversion
                val = int(val) if config["dtype"] == "int" else float(val)
                aligned[param] = val

            # Strict uniqueness check
            param_tuple = tuple(aligned[param] for param in param_names)
            if param_tuple not in existing:
                suggestions.append(aligned)
                existing.add(param_tuple)
                print(f"Successfully generated suggestion {len(suggestions)}/{n_points}")
            else:
                print("Duplicate parameter combination detected, regenerating...")

        except Exception as e:
            print(f"Error occurred during generation: {str(e)}")
            continue

    # Save results
    df_suggest = pd.DataFrame(suggestions)
    if 'Experiment ID' in df.columns:
        start_num = df['Experiment ID'].max() + 1
        df_suggest.insert(0, 'Experiment ID', range(start_num, start_num + n_points))
    df_suggest.to_excel(savepath, index=False)

    # Visualization
    if visualize and len(optimizer._space.target) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.maximum.accumulate(optimizer._space.target), 'o-')
        plt.xlabel('Number of Experiments')
        plt.ylabel('Target Value')
        plt.title(f'Optimization Progress (Best: {np.max(optimizer._space.target):.2f})')
        plt.grid(True)
        plt.show()

    return df_suggest


if __name__ == "__main__":
    result = bayesian_optimization_and_suggest(
        filepath='training_data/initial_data_with_features_new.xlsx',
        savepath='newProcesses/BO_newRound2_1.xlsx',
        target_params="Maximum Peak Height",
        recent_influence=0.6,
        visualize=True,
        n_points=8
    )
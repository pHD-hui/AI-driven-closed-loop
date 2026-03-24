import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# Encapsulated function
def guassFitData(guass_number,
                 file_path,
                 set_bounds="default",
                 save_params_path="fitted_parameters.csv",
                 save_and_view_training_path=None,
                 print_training_params=False
                 ):
    # Function parameter description:
    # guass_number: number of Gaussian peaks used to fit the whole curve
    # file_path: input file path, must be a CSV file
    # bounds: parameter range, can be customized
            # default: bounds generated automatically
            # custom input should follow the reference format:
            # ([mean], [standard deviation], [amplitude]) * number of Gaussian peaks
    # save_params_path: output filename for fitted parameters, default is "fitted_parameters.csv" in the same folder, must be saved as CSV
    # save_and_view_training_path: visualization of training process; input should be the target folder name for saving results, created automatically if it does not exist
            # training process outputs include:
            # 1. image of each spectrum and its fitted curve, saved in the input folder
            # 2. R^2 of each spectrum, saved as folder_path + "R2.csv"
    # return value: pd.DataFrame containing all fitted parameters
    maxepoch = 50

    def gaussian_sum(x, *params):
        # params should be guass_number groups of (mean, standard deviation, amplitude)
        assert len(params) == 3 * guass_number  # guass_number Gaussian distributions, each with 3 parameters
        y = np.zeros_like(x, dtype=np.float64)
        for i in range(guass_number):
            mean = params[3 * i]
            stddev = params[3 * i + 1]
            amplitude = params[3 * i + 2]
            y += amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
        return y

    def exp(x, mean, stddev, amplitude):
        y = amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
        return y

    def plot_exp(x, mean, stddev, amplitude):
        y = exp(x, mean, stddev, amplitude)
        plt.plot(y, x)

    def R_square(actually_data, predict_data):
        actually_data = np.array(actually_data)
        predict_data = np.array(predict_data)
        RSS = np.sum(np.square(actually_data - predict_data))
        TSS = np.sum(np.square(actually_data - np.mean(predict_data)))
        R2 = 1 - RSS / TSS
        return R2

    if file_path[-3:] != "csv":
        print(file_path)
        # file must be in CSV format to ensure stable reading
        # optionally, preprocessing code could be added to detect file type,
        # read with openpyxl or pandas, and automatically save as CSV
        raise TypeError("The file must be in CSV format. Please resave it as a CSV file in Excel and try again.")

    f = open(file_path, "r", encoding='utf-8')
    xdata = []
    ydata = []
    ytitle = []
    R2_list = []
    title = True

    # for i in f.readlines():
    #     if title:
    #         i = i.replace('\n','')
    #         i = i.split(',')
    #         for j in i[2:]:
    #             xdata.append(eval(j))
    #         title=False
    #     else:
    #         i = i.replace('\n', '')
    #         i = i.split(',')
    #         ytitle.append(i[0]+'-'+i[1])
    #         for j in i[2:]:
    #             ydata.append(eval(j))

    for i in f.readlines():
        if title:
            i = i.replace('\n', '')
            i = i.split(',')
            for j in i[1:]:
                xdata.append(eval(j))
            title = False
        else:
            i = i.replace('\n', '')
            i = i.split(',')
            # ytitle.append(i[0]+'-'+i[1])
            ytitle.append(i[0])
            for j in i[1:]:
                ydata.append(eval(j))

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ytitle = np.array(ytitle)
    ydata = ydata.reshape(ydata.shape[0] // xdata.shape[0], xdata.shape[0])
    xdata = xdata.T
    ydata = ydata.T
    ytitle = ytitle.reshape(ytitle.shape[0], 1)

    def get_params(y):
        initial_params = np.zeros((3 * guass_number))
        for i in range(guass_number):  # random initialization adjustment
            initial_params[i * 3] = min(xdata) + (i) / guass_number * (max(xdata) - min(xdata))
            initial_params[i * 3 + 1] = np.random.rand() * 50
            initial_params[i * 3 + 2] = np.random.rand() * 0.5 * max(y)
        return initial_params

    def get_bounds(x, y):
        # Set parameter bounds:
        # Simple version: [0, +infinity]
        # bounds = ([0 for i in range(guass_number * 3)], [np.inf for i in range(guass_number * 3)])

        # Enhanced version:
        # mean ∈ [minimum spectral wavelength x, maximum spectral wavelength x]
        # standard deviation ∈ [1, 50] (avoid std = 0 by setting minimum to 1)
        # amplitude ∈ [0, 2 * maximum spectral intensity y] (allow peak height to exceed current max)
        lower_bounds = []
        upper_bounds = []
        for i in range(guass_number):
            lower_bounds.extend([min(x), 0, 0])  # mean, standard deviation, amplitude
            upper_bounds.extend([max(x), 50, max(y)])
        return (lower_bounds, upper_bounds)

    finial_params = np.array([])
    y_fits = []

    # Fit data
    for index in range(ydata.shape[1]):
        R2 = 0
        epoch = 0
        while R2 < 0.8:
            if epoch > maxepoch:
                # if time constraint is exceeded
                popt = np.zeros((guass_number * 3))  # temporarily set popt to all zeros
                # output error report
                print("Fitting error (Spectrum No.{}): no optimal solution was found, or this spectrum has no peak. It will be discarded and the fitted data will be recorded as all zeros.".format(ytitle[index][0]))
                break
            else:
                # otherwise, count one more constrained round
                epoch += 1

            # randomly generate initial parameters
            initial_params = get_params(ydata[:, index])

            if set_bounds == "default":
                # if default parameter range is selected (all > 0)
                # generate directly
                bounds = get_bounds(xdata, ydata[:, index])
            else:
                bounds = set_bounds

            try:
                popt, pcov = curve_fit(gaussian_sum, xdata, ydata[:, index], p0=initial_params, maxfev=2000, bounds=bounds)
            except RuntimeError:
                R2 = 0
                continue
            except ValueError:
                print(initial_params)
                print(bounds)

            # prediction
            y_fit = gaussian_sum(xdata, *popt)

            # calculate and save fitting performance
            R2 = R_square(ydata[:, index], y_fit)
            R2_list.append(R2)

            # output fitted parameters (if needed)
            if print_training_params:
                print("Fitted parameters of {}:".format(ytitle[index][0]))
                for i in range(guass_number):
                    print(f"Gaussian {i+1}: Mean = {popt[3*i]}, Stddev = {popt[3*i+1]}, Amplitude = {popt[3*i+2]}")

                print("R^2 of {}: {}".format(ytitle[index][0], R2))

            # plotting
            if save_and_view_training_path == None:
                pass
            else:
                # need to output and inspect training process
                folder_path = save_and_view_training_path
                if not os.path.exists(folder_path):
                    # create if it does not exist
                    os.makedirs(folder_path)
                # plot results
                plt.figure(figsize=(10, 5))
                plt.scatter(xdata, ydata[:, index], color='r', label='Observations', s=10)
                plt.plot(xdata, y_fit, color='b', label='Fitted Gaussian Sum', linewidth=2)
                for i in range(guass_number):
                    plt.plot(xdata, exp(xdata, popt[3 * i + 0], popt[3 * i + 1], popt[3 * i + 2]), "--")
                plt.title('Least Squares Fit of Gaussian Sum of Process No.' + ytitle[index][0])
                plt.legend()
                # fig_name = ytitle[index][0] + ".svg"  # str
                fig_name = str(ytitle[index][0]) + ".svg"
                plt.savefig(os.path.join(folder_path, fig_name), dpi=300)
                # plt.show()
                plt.close()

        # record parameters
        finial_params = np.append(finial_params, popt)

    # save parameters
    finial_params = finial_params.reshape(len(finial_params) // (guass_number * 3), guass_number * 3)
    # finial_params = np.concatenate((ytitle, finial_params), axis = 1)
    # columns_name = ["Mean1", "Stddev1", "Peak Height1", "Mean2", "Stddev2", "Peak Height2"]
    columns_name = [f"{name}{i}" for i in range(1, guass_number + 1) for name in ["Mean", "Stddev", "Peak Height"]]
    finial_params = pd.DataFrame(finial_params, columns=columns_name, index=ytitle.reshape(-1))
    R2_list = pd.DataFrame(R2_list, index=ytitle.reshape(-1))

    if save_params_path[-3:] != "csv":
        raise TypeError("The output should be saved as a CSV file.")
    else:
        finial_params.to_csv(save_params_path, encoding='utf-8_sig')

    if save_and_view_training_path != None:
        R2_path = folder_path + "R2.csv"
        R2_list.to_csv(R2_path, encoding='utf-8_sig')

    return finial_params

# Example
if __name__ == "__main__":
    # guassFitData(2, 'first_round_data-PL-processed.csv', save_params_path="fitted_parameters_try.csv")
    # guassFitData(2, "second_round_data-PL(corrected).csv", save_params_path="fitted_parameters_round2.csv", save_and_view_training_path="image")
    guassFitData(2,
                 "representative_spectra.csv",
                 save_params_path="fitted_parameters_round1.csv",
                 save_and_view_training_path="image")
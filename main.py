import numpy as np
import matplotlib.pyplot as plt
from lin_reg_ecg import test_fit_line, find_new_peak, check_if_improved
from lin_reg_smartwatch import pearson_coeff, fit_predict_mse, multilinear_fit_predict_mse, scatterplot_and_line, scatterplot_and_curve
from gradient_descent import ackley, gradient_ackey, gradient_descent, plot_ackley_function
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def task_1_1():

    test_fit_line()

    # Load ecg signal from 'data/ecg.npy' using np.load
    ecg = np.load('data/ecg.npy')

    # Load indices of peaks from 'indices_peaks.npy' using np.load. There are 83 peaks.
    peaks = np.load('data/indices_peaks.npy')

    # Create a "timeline". The ecg signal was sampled at sampling rate of 180 Hz, and in total 50 seconds.
    # Datapoints are evenly spaced. Hint: shape of time signal should be the same as the shape of ecg signal.
    time = np.arange(0, 50, (1/180)) #180Hz
    print(f'time shape: {time.shape}, ecg signal shape: {ecg.shape}')
    print(f'First peak: ({time[peaks[0]]:.3f}, {ecg[peaks[0]]:.3f})') # (0.133, 1.965)

    # Plot of ecg signal (should be similar to the plot in Fig. 1A of HW1, but shown for 50s, not 8s)
    plt.plot(time, ecg)
    plt.plot(time[peaks], ecg[peaks], "x")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.show()

    new_peaks = np.zeros(peaks.size)
    new_sig = np.zeros(peaks.size)
    improved = np.zeros(peaks.size)

    for i, peak in enumerate(peaks):
        x_new, y_new = find_new_peak(peak, time, ecg)
        new_peaks[i] = x_new
        new_sig[i] = y_new
        improved[i] = check_if_improved(x_new, y_new, peak, time, ecg)

    print(f'Improved peaks: {np.sum(improved)}, total peaks: {peaks.size}')
    print(f'Percentage of peaks improved: {np.sum(improved) / peaks.size :.4f}')


def task_1_2():

    # COLUMN NAMES: hours_sleep, hours_work, avg_pulse, max_pulse, duration, exercise_intensity, fitness_level, calories
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, "duration": 4,
                    "exercise_intensity": 5, "fitness_level": 6, "calories": 7}
    # Load the data from 'data/smartwatch_data.npy' using np.load
    smartwatch_data = np.load('data/smartwatch_data.npy')

    # Now you can access it, for example,  smartwatch_data[:, column_to_id["hours_sleep"]]

    # Meaningful relations
    # duration(x) => calories(y)
    print("------------------------Meaningful Relations----------------------------")
    theta1, mse1 = fit_predict_mse(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["calories"]], 1)
    corrcoef1 = pearson_coeff(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["calories"]])
    print("For calories depends on duration, ")
    print(f'Theta: {theta1}, MSE: {mse1}, Pearson Coefficient: {corrcoef1}')

    # duration(x) => fitness_level(y)
    theta2, mse2 = fit_predict_mse(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["fitness_level"]], 1)
    corrcoef2 = pearson_coeff(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["fitness_level"]])
    print("For fitness_level depends on duration, ")
    print(f'Theta: {theta2}, MSE: {mse2}, Pearson Coefficient: {corrcoef2}')

    # avg_pulse(x) => max_pulse(y)
    theta6, mse6 = fit_predict_mse(smartwatch_data[:, column_to_id["avg_pulse"]], smartwatch_data[:, column_to_id["max_pulse"]], 1)
    corrcoef6 = pearson_coeff(smartwatch_data[:, column_to_id["avg_pulse"]], smartwatch_data[:, column_to_id["max_pulse"]])
    print("For max_pulse depends on avg_pulse, ")
    print(f'Theta: {theta6}, MSE: {mse6}, Pearson Coefficient: {corrcoef6}')

    print("------------------------Non Linear Relations----------------------------")
    # No linear relations
    # duration(x) => hours_sleep(y)
    theta4, mse4 = fit_predict_mse(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["hours_sleep"]], 1)
    corrcoef4 = pearson_coeff(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["hours_sleep"]])
    print("For hours_sleep depends on duration, ")
    print(f'Theta: {theta4}, MSE: {mse4}, Pearson Coefficient: {corrcoef4}')

    # hours_work(x) => calories(y)
    theta5, mse5 = fit_predict_mse(smartwatch_data[:, column_to_id["hours_work"]], smartwatch_data[:, column_to_id["calories"]], 1)
    corrcoef5 = pearson_coeff(smartwatch_data[:, column_to_id["hours_work"]], smartwatch_data[:, column_to_id["calories"]])
    print("For calories depends on hours_work, ")
    print(f'Theta: {theta5}, MSE: {mse5}, Pearson Coefficient: {corrcoef5}')

    # max_pulse(x) => exercise_intensity(y)
    theta3, mse3 = fit_predict_mse(smartwatch_data[:, column_to_id["max_pulse"]], smartwatch_data[:, column_to_id["exercise_intensity"]], 1)
    corrcoef3 = pearson_coeff(smartwatch_data[:, column_to_id["max_pulse"]], smartwatch_data[:, column_to_id["exercise_intensity"]])
    print("For exercise_intensity depends on max_pulse, ")
    print(f'Theta: {theta3}, MSE: {mse3}, Pearson Coefficient: {corrcoef3}')

    # Polynomial regression
    # TODO (use fit_predict_mse)
    print("------------------------Polynomial regression----------------------------")
    theta1, mse1 = fit_predict_mse(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["calories"]], 4)
    corrcoef1 = pearson_coeff(smartwatch_data[:, column_to_id["duration"]], smartwatch_data[:, column_to_id["calories"]])
    print("For calories depends on duration, ")
    print(f'Theta: {theta1}, MSE: {mse1}, Pearson Coefficient: {corrcoef1}')

    # Multilinear
    # TODO (use multilinear_fit_predict_mse)
    print("------------------------Multilinear----------------------------")
    theta, mse = multilinear_fit_predict_mse(smartwatch_data[:, column_to_id["avg_pulse"]], smartwatch_data[:, column_to_id["exercise_intensity"]], smartwatch_data[:, column_to_id["max_pulse"]])
    print("For max_pulse depends on avg_pulse and exercise_intensity, ")
    print(f'Theta: {theta}, MSE: {mse}')

    # When choosing two variables for polynomial regression, use a pair that you used for Meaningful relations, so you can check if the MSE decreases.
    # When choosing a few variables for multilinear regression, use a pair that you used for Meaningful relations, so you can check if the MSE decreases.


def task_2():
    
    # Load data and normalize
    heart_data = np.load('data/heart_data.npy')
    heart_data_targets = np.load('data/heart_data_targets.npy')

    sc = StandardScaler() 
    X_normalized = sc.fit_transform(heart_data) 

    # Spilit data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_normalized, heart_data_targets,
                                                        test_size=0.2, random_state=0)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Create a classifier
    # Solver lbfgs supports only 'l2' or 'none' penalties
    # Try 'l2' penalty
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)
    acc_train = clf.score(x_train, y_train)
    acc_test = clf.score(x_test, y_test)
    print(f'Train accuracy: {acc_train:.4f}. Test accuracy: {acc_test:.4f}.')
    
    # Try 'none' penalty
    clf_none = LogisticRegression(penalty='none')
    clf_none.fit(x_train, y_train)
    acc_train_none = clf_none.score(x_train, y_train)
    acc_test_none = clf_none.score(x_test, y_test)
    print(f'Train accuracy: {acc_train_none:.4f}. Test accuracy: {acc_test_none:.4f}.')
    
    # Since both penalties have the same test accuracy and 'l2' penalty has higher train accuracy, I would rather choose 'l2' penalty.
    
    # Calculate predictions and log_loss
    y_train_pred = clf.predict_proba(x_train)
    y_test_pred = clf.predict_proba(x_test)
    loss_train = log_loss(y_train, y_train_pred)
    loss_test = log_loss(y_test, y_test_pred)
    print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

    # Print theta vector (and also the bias term).
    print(f'Theta vector: {clf.coef_}.')
    print(f'bias: {clf.intercept_}.')


def task_3():
    # Plot the Function, to see how it looks like
    plot_ackley_function(ackley)

    # Choose a random starting point
    x0 = np.random.uniform() # choose a random starting x-coordinate, use rand function from np.random
    y0 = np.random.uniform() # choose a random starting y-coordinate, use rand function from np.random
    print(f'Initial value x = {x0}. y = {y0}.')

    # Call the function gradient_descent
    # Choose max_iter, learning_rate, lr_decay (first see what happens with lr_decay=1, then change it to a lower value)
    max_iter = 1000
    learning_rate = 0.1
    lr_decay = 0.75
    x, y, E_list = gradient_descent(ackley, gradient_ackey, x0, y0, learning_rate, lr_decay, max_iter)

    # Print the point that is the best found solution
    print(f'Best found solution point = {x:.4f}, {y:.4f}')

    # Make a plot of the cost over iteration. Do not forget to label the plot (xlabel, ylabel, title)
    plt.plot(E_list)
    plt.title('The cost over iteration')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()

    print(f'Solution found: f({x:.4f}, {y:.4f})= {ackley(x,y):.4f}' )
    print(f'Global optimum: f(0, 0)= {ackley(0,0):.4f}')



def main():
    task_1_1()
    task_1_2()
    task_2()
    task_3()


if __name__ == '__main__':
    main()

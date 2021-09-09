import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

def plot_residuals(x, y, yhat):
    """ Creates a residual plot from one variable """

    # Create dataframe of input values
    df = pd.DataFrame({'x':x, 'y':y, 'yhat':yhat})
    # Calculate residuals
    df['residual'] = df['y'] - df['yhat']

    # Plot residuals
    return sns.relplot(df['x'], df['residual'], kind='scatter')

def regression_errors(y, yhat):
    """ Returns SSE, ESS, TSS, MSE, and RMSE from two arrays """

    # Create dataframe of input values
    df = pd.DataFrame({'y':y, 'yhat':yhat})
    # Calculate errors
    sse = round(mean_squared_error(df.y, df.yhat) * len(y), 2)
    ess = round(sum((df.yhat - df.y.mean())**2), 2)
    tss = round(ess + sse, 2)
    mse = round(mean_squared_error(df.y, df.yhat), 2)
    rmse = round(mse ** 0.5, 2)
    # Organize error calculations in dict
    return_dict = {
        'SSE':sse,
        'ESS':ess,
        'TSS':tss,
        'MSE':mse,
        'RMSE':rmse
    }

    # Return error calculations
    return return_dict

def baseline_mean_errors(y):
    """ Returns baseline model's SSE, MSE, and RMSE from array """

    # Create dataframe
    df = pd.DataFrame({'y':y, 'baseline':y.mean()})
    # Calculate errors
    sse_baseline = round(mean_squared_error(df.y, df.baseline) * len(df), 2)
    mse_baseline = round(mean_squared_error(df.y, df.baseline), 2)
    rmse_baseline = round(mse_baseline ** 0.5, 2)
    # Assign to dict
    return_dict = {
        'Baseline_SSE':sse_baseline,
        'Baseline_MSE':mse_baseline,
        'Baseline_RMSE':rmse_baseline
    }

    # Return error calculations
    return return_dict

def better_than_baseline(y, yhat):
    """ Returns True if your regression model performs better than the mean baseline"""

    # Run calculations
    model_errors = regression_errors(y, yhat)
    baseline_errors = baseline_mean_errors(y)
    # Compare model and baseline errors
    better = model_errors['RMSE'] < baseline_errors['Baseline_RMSE']

    # Return True or False for model performing better than baseline
    return better
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(df):
    """ Run a Seaborn pairplot with red regression line """
    sns.pairplot(df, kind="reg", plot_kws={'line_kws':{'color':'red'}})

def months_to_years(df):
    """ Convert tenure from months to years """
    df['tenure_years'] = df.tenure / 12
    df['tenure_years'] = df.tenure_years.astype('int64')
    return df

def plot_categorical_and_continuous_vars(df, cats, conts):
    """ 
        Plot a distribution and all boxplots for each
            categorical variable and its relation to all continuous variables
    """
    for cat in cats:
        print("| -----------------", cat, "----------------- |")
        plt.hist(df[cat])
        plt.show()
        for cont in conts:
            sns.boxplot(data=df, x=cat, y=cont)
            plt.show()
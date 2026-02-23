"""
Statistics and Trends Assignment - Applied Data Science
Module: 7PAM2000 - Applied Data Science
Module Leader: Dr. William Cooper

This script analyses the Global Mental Health and Well-being
Indicators 2023 dataset, exploring relationships between depression
and anxiety prevalence, mental health spending, psychiatrist
availability, therapy access, GDP, life expectancy, unemployment,
and suicide rates across world regions.
"""
try:
    from corner import corner
except ImportError:
    corner = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Create a scatter plot showing the relationship between
    Mental Health Spending per capita (USD) and the Therapy
    Access Index, coloured by Region, with a linear trend line.

    Args:
        df (pd.DataFrame): The preprocessed Mental Health dataframe.

    Returns:
        None. Saves the figure as 'relational_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    regions = df['Region'].unique()
    palette = sns.color_palette('tab10', len(regions))
    region_color = dict(zip(regions, palette))

    for region in regions:
        subset = df[df['Region'] == region]
        ax.scatter(
            subset['Mental_Health_Spending_USD'],
            subset['Therapy_Access_Index'],
            label=region,
            color=region_color[region],
            s=75,
            alpha=0.85,
            edgecolors='white',
            linewidth=0.6
        )

    # Annotate top 5 spenders
    top5 = df.nlargest(5, 'Mental_Health_Spending_USD')
    for _, row in top5.iterrows():
        ax.annotate(
            row['Country'],
            xy=(row['Mental_Health_Spending_USD'],
                row['Therapy_Access_Index']),
            xytext=(5, 3),
            textcoords='offset points',
            fontsize=8,
            color='#222222'
        )

    # Linear trend line
    x = df['Mental_Health_Spending_USD'].values
    y = df['Therapy_Access_Index'].values
    slope, intercept, r_value, _, _ = ss.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color='black',
        linestyle='--',
        linewidth=1.8,
        label=f'Trend line (r = {r_value:.2f})'
    )

    ax.set_xlabel('Mental Health Spending per Capita (USD)', fontsize=13)
    ax.set_ylabel('Therapy Access Index (0-100)', fontsize=13)
    ax.set_title(
        'Mental Health Spending vs Therapy Access by Region (2023)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.7)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=150)
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create a grouped horizontal bar chart comparing mean
    Depression Rate (%) and mean Anxiety Rate (%) across
    world regions, ordered by depression prevalence.

    Args:
        df (pd.DataFrame): The preprocessed Mental Health dataframe.

    Returns:
        None. Saves the figure as 'categorical_plot.png'.
    """
    region_stats = df.groupby('Region').agg(
        Mean_Depression=('Depression_Rate_Pct', 'mean'),
        Mean_Anxiety=('Anxiety_Rate_Pct', 'mean')
    ).sort_values('Mean_Depression', ascending=True).reset_index()

    y = np.arange(len(region_stats))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.barh(
        y - height / 2,
        region_stats['Mean_Depression'],
        height,
        label='Mean Depression Rate (%)',
        color='#5c6bc0',
        edgecolor='white'
    )
    bars2 = ax.barh(
        y + height / 2,
        region_stats['Mean_Anxiety'],
        height,
        label='Mean Anxiety Rate (%)',
        color='#ef5350',
        edgecolor='white'
    )

    # Value labels
    for bar in bars1:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', ha='left', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', ha='left', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(region_stats['Region'], fontsize=10)
    ax.set_xlabel('Mean Prevalence Rate (%)', fontsize=13)
    ax.set_title(
        'Mean Depression & Anxiety Rate by World Region (2023)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim(0, region_stats['Mean_Anxiety'].max() + 3.5)
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=150)
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Create a correlation heatmap of all key numerical features
    in the Global Mental Health dataset.

    Args:
        df (pd.DataFrame): The preprocessed Mental Health dataframe.

    Returns:
        None. Saves the figure as 'statistical_plot.png'.
    """
    numerical_cols = [
        'Depression_Rate_Pct',
        'Anxiety_Rate_Pct',
        'Suicide_Rate_Per100k',
        'Mental_Health_Spending_USD',
        'Psychiatrists_Per100k',
        'Therapy_Access_Index',
        'GDP_per_Capita_USD',
        'Life_Expectancy',
        'Unemployment_Pct'
    ]
    corr_matrix = df[numerical_cols].corr()

    display_labels = [
        'Depression %',
        'Anxiety %',
        'Suicide Rate',
        'MH Spending',
        'Psychiatrists',
        'Therapy Access',
        'GDP p.c.',
        'Life Expectancy',
        'Unemployment %'
    ]
    corr_matrix.index = display_labels
    corr_matrix.columns = display_labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 8.5}
    )
    ax.set_title(
        'Correlation Heatmap — Global Mental Health Indicators (2023)',
        fontsize=13, fontweight='bold', pad=14
    )
    ax.tick_params(axis='x', labelsize=9, rotation=32)
    ax.tick_params(axis='y', labelsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=150)
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Compute the four statistical moments for a given column.

    Args:
        df (pd.DataFrame): The preprocessed Mental Health dataframe.
        col (str): The column name to analyse.

    Returns:
        tuple: (mean, stddev, skewness, excess_kurtosis) as floats.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess the Global Mental Health 2023 dataset: inspect
    structure, produce summary statistics, compute correlation
    matrix, and remove any missing or duplicate rows.

    Args:
        df (pd.DataFrame): The raw Mental Health dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe ready for analysis.
    """
    print("=== Dataset Head ===")
    print(df.head())

    print("\n=== Dataset Tail ===")
    print(df.tail())

    print("\n=== Descriptive Statistics ===")
    print(df.describe().round(3))

    print("\n=== Correlation Matrix (numerical) ===")
    numerical_cols = [
        'Depression_Rate_Pct', 'Anxiety_Rate_Pct',
        'Suicide_Rate_Per100k', 'Mental_Health_Spending_USD',
        'Psychiatrists_Per100k', 'Therapy_Access_Index',
        'GDP_per_Capita_USD', 'Life_Expectancy', 'Unemployment_Pct'
    ]
    print(df[numerical_cols].corr().round(3))

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes > 0:
        print(f"\nRemoved {dupes} duplicate rows.")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values.")

    print(f"\nFinal dataset shape: {df.shape}")
    return df


def writing(moments, col):
    """
    Print and interpret the four statistical moments for a column.

    Args:
        moments (tuple): (mean, stddev, skewness, excess_kurtosis).
        col (str): The column name that was analysed.

    Returns:
        None.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0.5:
        skew_desc = 'right skewed'
    elif moments[2] < -0.5:
        skew_desc = 'left skewed'
    else:
        skew_desc = 'not skewed'

    if moments[3] > 1:
        kurt_desc = 'leptokurtic'
    elif moments[3] < -1:
        kurt_desc = 'platykurtic'
    else:
        kurt_desc = 'mesokurtic'

    print(f'The data was {skew_desc} and {kurt_desc}.')
    return


def main():
    """
    Main entry point: load data, preprocess, generate all plots,
    and compute statistical moments for the chosen column.
    """
    df = pd.read_csv('mental_health_2023.csv')
    df = preprocessing(df)
    col = 'Depression_Rate_Pct'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()

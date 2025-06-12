import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

def plot_means_and_stds(df, output_path):
    """
    Create a table with means and standard deviations for all final model variables.
    """
    means = df.mean()
    stds = df.std()
    
    summary_df = pd.DataFrame({'Mean': means, 'Std': stds})
    summary_df.to_csv(output_path + 'means_and_stds_table.csv')  # Save table as CSV
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df.reset_index(), x='index', y='Mean', yerr=summary_df['Std'])
    plt.xticks(rotation=45)
    plt.title('Means and Standard Deviations of Final Model Variables')
    plt.tight_layout()
    plt.savefig(output_path + 'means_and_stds.png')
    plt.close()

def plot_sample_summary(df, output_path):
    """
    Create a summary plot of sample information:
    - Age distribution
    - sex distribution
    - Diagnosis status distribution
    """
    plt.figure(figsize=(15, 5))
    
    # Age distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df['Age'], kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    
    # Sex distribution
    plt.subplot(1, 3, 2)
    sex_counts = df['Sex'].value_counts()
    sns.barplot(x=sex_counts.index, y=sex_counts.values)
    plt.title('Sex Distribution')
    plt.ylabel('Count')
    plt.xlabel('Sex')
    
    # Diagnosis status distribution
    plt.subplot(1, 3, 3)
    dx_counts = df['Dx'].value_counts()
    sns.barplot(x=dx_counts.index, y=dx_counts.values)
    plt.title('Diagnosis Status Distribution')
    plt.ylabel('Count')
    plt.xlabel('Dx')
    
    plt.tight_layout()
    plt.savefig(output_path + 'sample_summary.png')
    plt.close()

def plot_means_and_stds_with_model(df, ari_features, scared_features, output_path):
    """
    Create a summary table and heatmap showing means and standard deviations of features,
    with color-coded rows by model type (ARI, SCARED, ARI+SCARED) and bolded feature names.
    """

    # Assign model labels
    feature_model = {}
    ordered_features = []
    for feat in ari_features:
        if feat in scared_features:
            feature_model[feat] = 'ARI+SCARED'
        else:
            feature_model[feat] = 'ARI'
        if feat not in ordered_features:
            ordered_features.append(feat)
    for feat in scared_features:
        if feat not in feature_model:
            feature_model[feat] = 'SCARED'
            ordered_features.append(feat)

    means = df.mean()
    stds = df.std()

    summary_df = pd.DataFrame({
        'Mean': means,
        'Std': stds,
    })
    summary_df['Model'] = pd.Series(feature_model)
    summary_df = summary_df.loc[ordered_features].reset_index().rename(columns={'index': 'Feature'})
    summary_df = summary_df[['Feature', 'Mean', 'Std', 'Model']]
    summary_df.to_csv(output_path + 'means_and_stds_with_model.csv', index=False)
    print(f"Saved CSV to {output_path}means_and_stds_with_model.csv")

    # === Optional Heatmap ===
    heatmap_data = summary_df[['Mean', 'Std']].values
    annot_data = summary_df[['Mean', 'Std']].round(2).astype(str).values

    plt.figure(figsize=(7, 0.6 * len(summary_df) + 2))
    ax = sns.heatmap(
        heatmap_data,
        annot=annot_data,
        fmt='',
        cmap='Blues',
        cbar=False,
        linewidths=1.5,
        linecolor='white',
        xticklabels=['Mean', 'Std'],
        yticklabels=summary_df['Feature'],
        annot_kws={"size": 16, "weight": "bold", "color": "#002147"}
    )

    ylabels = [
        f"{feat} ({model})" if model else feat
        for feat, model in zip(summary_df['Feature'], summary_df['Model'])
    ]
    ax.set_yticklabels(ylabels, fontsize=16, weight='bold', color='#002147')
    ax.set_xticklabels(['Mean', 'Std'], fontsize=18, weight='bold', color='#002147')
    plt.title('Means and Standard Deviations of Final Model Variables', fontsize=22, weight='bold', color='#002147', pad=20)
    plt.tight_layout()
    plt.savefig(output_path + 'means_and_stds_heatmap_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === Plotly Table with bluish-greenish colors and bold feature names ===
    summary_df_display = summary_df.copy()
    summary_df_display['Mean'] = summary_df_display['Mean'].round(3)
    summary_df_display['Std'] = summary_df_display['Std'].round(3)

    # Bold the feature names using HTML
    summary_df_display['Feature'] = summary_df_display['Feature'].apply(lambda x: f"<b>{x}</b>")

    model_color_map = {
    'ARI': '#cce5f7',      # Light sky blue
    'SCARED': '#99ccf3',   # Medium blue
    'ARI+SCARED': '#66b2e8'  # Vibrant blue
    }
    row_colors = [model_color_map.get(model, 'white') for model in summary_df_display['Model']]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df_display.columns),
            fill_color='#002147',
            font=dict(color='white', size=24),
            align='center',
            height=50
        ),
        cells=dict(
            values=[summary_df_display[col] for col in summary_df_display.columns],
            fill_color=[row_colors] * len(summary_df_display.columns),
            font=dict(color='#002147', size=24),
            align='center',
            height=40
        )
    )])

    fig.update_layout(
        title_text='Means and Standard Deviations of Final Model Variables',
        title_font_size=24,
        title_x=0.5,
        margin=dict(t=80, b=40)
    )

    fig.write_html(output_path + 'means_and_stds_table_plot_plotly.html')
    print(f"Saved interactive table to {output_path}means_and_stds_table_plot_plotly.html")

def main():
    # Load the data
    data_path = '07 - Ridge/Cleaned_data_imputed.csv'
    df_full = pd.read_csv(data_path)

    selected_ARI_features = pd.read_csv('07 - Ridge/ARI_features_for_ridgereg.csv')['feature'].tolist()
    selected_SCARED_features = pd.read_csv('07 - Ridge/SCARED_features_for_ridgereg.csv')['feature'].tolist()
    # Remove duplicates while preserving order
    seen = set()
    selected_features = []
    for feat in selected_ARI_features + selected_SCARED_features:
        if feat not in seen:
            selected_features.append(feat)
            seen.add(feat)

    df = df_full[selected_features].copy()
    # Convert all "Dwell" columns from ms to sec
    dwell_cols = [col for col in df.columns if col.startswith("Dwell")]
    df[dwell_cols] = df[dwell_cols] / 1000

    output_path = 'plots_for_poster/'

    plot_means_and_stds_with_model(df, selected_ARI_features, selected_SCARED_features, output_path)

    # Plot descriptives for Dx, Sex, Age
    plot_sample_summary(df_full[['Age', 'Sex', 'Dx']], output_path)

if __name__ == "__main__":
    main()
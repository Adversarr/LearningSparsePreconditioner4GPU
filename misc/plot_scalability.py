"""
python misc/plot_scalability.py --input output/heat_tetmesh_rl2_loss_unseen_meshes/all_infer_rL2_heat_tetmesh_8.csv --step=32
CSV header for the input file:
Key,Solve Time (ms),Precond Time (ms),#Iteration,Matrix Size
METHOD,0.0,0.0,324,12345

Features:
1. Log-log scale plots
2. Rounds matrix sizes to nearest specified step
3. Three metrics: solve time, total time, iterations
"""
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def reinterprete(method):
    mapped = {
        'neural-cpu': 'Ours+CPU',
        'neural-cuda': 'Ours+CUDA',
        'ainv-cpu': "AINV+CPU",
        'ainv-cuda': "AINV+CUDA",
        'ic-cpu': "IC+CPU",
        'ic-cuda': "IC+CUDA",
        'diagonal-cpu': "Diag+CPU",
        'diagonal-cuda': "Diag+CUDA",
        'none-cpu': "None+CPU",
        'none-cuda': "None+CUDA",
    }
    if not isinstance(method, str):
        output = [mapped.get(m, m) for m in method]
        return output
    return mapped.get(method, method)


def preprocess_keys(df):
    """Preprocess method names according to specified rules."""
    # Make a copy to avoid modifying original
    processed = df.copy()
    
    # Rule 1: lowercase and replace + with -
    processed['Key'] = processed['Key'].str.lower().str.replace('+', '-')
    
    # Rule 2: Append -cpu to neural
    processed['Key'] = processed['Key'].apply(
        lambda x: f"{x}-cpu" if x == "neural" else x
    )
    # Rule 3: Remove pcg- prefix
    processed['Key'] = processed['Key'].str.replace('^pcg-', '', regex=True)
    
    # Filtering rules
    keep_mask = (
        # Keep CPU methods only if they contain 'ic-cpu'
        processed['Key'].str.contains('ic-cpu') | 
        # OR don't contain 'cpu' at all
        (~processed['Key'].str.contains('cpu')) 
    )

    # keep_mask &= (
    #     # Remove all ic-cuda
    #     ~processed["Key"].str.contains("ic-cuda")
    # )
    
    # keep_mask &= (
    #     # Remove all ic-cuda
    #     ~processed["Key"].str.contains("none")
    # )
    processed["Key"] = processed["Key"].apply(reinterprete)
    return processed[keep_mask]

# def round_to_nearest_step(value, step):
#     """Round value to nearest multiple of step."""
#     return step * round(value / step)
def round_to_nearest_step(value, step):
    """Round value to nearest multiple of step in log space for better log-scale alignment."""
    if value <= 0:
        return value  # protect against log(0)

    # Convert to log space, round, then convert back
    log_value = np.log10(value)
    log_step = np.log10(step)
    rounded_log = log_step * round(log_value / log_step)
    return int(10 ** rounded_log)


def main(cfg):
    # Read the CSV file
    df = pd.read_csv(cfg.input)
    
    # Set global font sizes
    plt.rc('font', size=13)           # Controls default text sizes (e.g., labels)
    plt.rc('axes', titlesize=14)      # Font size of the axes title
    plt.rc('axes', labelsize=14)      # Font size of the x and y labels
    plt.rc('xtick', labelsize=12)     # Font size of the tick labels
    plt.rc('ytick', labelsize=12)     # Font size of the tick labels
    plt.rc('legend', fontsize=12)     # Font size of the legend

    df = preprocess_keys(df)

    # Round matrix sizes
    if cfg.step > 0:
        df['Matrix Size'] = df['Matrix Size'].apply(
            lambda x: round_to_nearest_step(x, cfg.step)
        )

    # Calculate total time
    df['Total Time (ms)'] = df['Solve Time (ms)'] + df['Precond Time (ms)']

    # Create long-form dataframe
    plot_df = df.melt(
        id_vars=["Key", "Matrix Size"],
        value_vars=[
            # "Solve Time (ms)",
            "Total Time (ms)",
            # "#Iteration",
        ],
        var_name="Metric",
        value_name="Value",
    )

    # Create plot with log scales
    g: sns.FacetGrid = sns.relplot(
        data=plot_df,
        x='Matrix Size',
        y='Value',
        hue='Key',
        col='Metric',
        kind='line',
        style='Key',
        facet_kws={'sharey': False},
        legend="full",
    )

    # Set log scales and formatting
    g.set(xscale="log", yscale="log")
    g.fig.set_size_inches(5.5, 6)
    for ax in g.axes.flat:
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Total Time (ms)')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    g.set_titles("{col_name}")
    # plt.legend(
    #     fontsize=10,
    #     frameon=False,
    #     framealpha=1,
    #     title="Method",
    #     loc="upper left",
    # )
    g.legend.set_loc("upper left")
    g.legend.set_bbox_to_anchor((0.12, 0.95))
    g.legend.set_title("Method")
    xmin = plot_df['Matrix Size'].min()
    xmax = plot_df['Matrix Size'].max()
    ymin = plot_df['Value'].min()
    ymax = plot_df['Value'].max()
    print(f"X range: {xmin} - {xmax}, {np.sqrt(xmin * xmax)}")
    g.set(xlim=(xmin, xmax))
    
    g.ax.set_xticks([2000, 4000, 8000, 16000, 32000])
    g.ax.set_yticks([20, 40, 80, 160, 320])
    plt.tight_layout()
    filename = 'plots/scalability.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="scalability.csv",
        help="Input CSV file with scalability data"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=400,
        help="Step size for rounding matrix sizes (default: 400)"
    )
    main(parser.parse_args())
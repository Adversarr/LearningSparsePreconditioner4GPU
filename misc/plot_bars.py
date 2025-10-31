from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def reinterprete(method):
    mapped = {
        'Neural': 'Ours+CPU',
        'Neural+CUDA': 'Ours+CUDA',
        'PCG-ainv-cpu': "AINV+CPU",
        'PCG-ainv-cuda': "AINV+CUDA",
        'PCG-ic-cpu': "IC+CPU",
        'PCG-ic-cuda': "IC+CUDA",
        'PCG-diagonal-cpu': "Diag+CPU",
        'PCG-diagonal-cuda': "Diag+CUDA",
        'PCG-none-cpu': "None+CPU",
        'PCG-none-cuda': "None+CUDA",
    }
    if not isinstance(method, str):
        output = [mapped.get(m, m) for m in method]
        return output
    return mapped.get(method, method)

def main(args):
    # 1. Read input file
    file_path = args.input_file
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return

    try:
        df = pd.read_csv(file_path)
        required_cols = ['Key', 'Precond Time (ms)', 'Solve Time (ms)']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: File {file_path} missing required columns ({required_cols}).")
            return

        # 2. Generate plot (no case/rtol dependency)
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plot to {output_dir}")

        fig, ax = plt.subplots(figsize=(12, 6))
        rtol_data = df.set_index('Key')

        # Plot stacked bars
        methods = rtol_data.index
        precond_times = pd.to_numeric(rtol_data['Precond Time (ms)'], errors='coerce').fillna(0)
        solve_times = pd.to_numeric(rtol_data['Solve Time (ms)'], errors='coerce').fillna(0)

        # Horizontal line for Neural+CUDA (if exists)
        if "Neural+CUDA" in methods:
            neural_cuda_total = precond_times["Neural+CUDA"] + solve_times["Neural+CUDA"]
            ax.axhline(
                y=neural_cuda_total,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                # label='Neural+CUDA Total Time'
                label='Ours+CUDA'
            )

        bar_width = 0.7
        palette = sns.color_palette("pastel")
        ax.bar(reinterprete(methods), precond_times, width=bar_width, color=palette[0], edgecolor='white', linewidth=0.7, label='Construction Time')
        ax.bar(reinterprete(methods), solve_times, width=bar_width, bottom=precond_times, color=palette[1], edgecolor='white', linewidth=0.7, label='Solve Time')
        max_time_ms = args.max_time_ms

        # Add text labels for total time on each bar
        for i, method in enumerate(reinterprete(methods)):
            total_time = precond_times[methods[i]] + solve_times[methods[i]]
            position = min(total_time + 1, max_time_ms - 6)
            ax.text(
                i, 
                position,  # Small offset above the bar
                f'{int(total_time)}',
                ha='center',
                va='bottom',
                fontsize=13,
                color='black'
            )

        # Customize plot (generic title, no rtol/case)
        # ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Total Time (ms)', fontsize=14)
        # ax.set_title('Performance Comparison', fontsize=14, pad=20)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=20, fontsize=13)  # <-- Added this line to rotate xticks

        # Find the method with the second smallest total time (excluding Neural+CUDA)
        total_times = precond_times + solve_times
        if "Neural+CUDA" in total_times:
            neural_time = total_times["Neural+CUDA"]
            other_methods = total_times.drop("Neural+CUDA")
            if len(other_methods) > 0:
                second_min_time = other_methods.min()
                second_min_method = reinterprete(other_methods.idxmin())

                # Plot horizontal line for the second-best method
                ax.axhline(
                    y=second_min_time,
                    color='blue',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'{second_min_method}'
                )

                # Calculate the percentage difference
                percentage_diff = ((second_min_time - neural_time) / neural_time) * 100
                # Get the x-position of Neural+CUDA
                x_neural = list(methods).index("Neural+CUDA")
                # Plot vertical arrow and text
                arrow_mid_x = x_neural + 0.31  # Offset to avoid overlapping bars
                ax.annotate(
                    '', 
                    xy=(arrow_mid_x, neural_time), 
                    xytext=(arrow_mid_x, second_min_time),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                )
                # Place text next to the arrow
                ax.text(
                    arrow_mid_x - 0.9,  # Slight right offset
                    (neural_time + second_min_time) / 2,
                    f"+{percentage_diff:.1f}%",
                    ha='left', 
                    va='center',
                    backgroundcolor='white',
                    fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.0),
                )

        ax.set_ylim(0, max_time_ms)
        ax.set_yticks(np.arange(0, max_time_ms + 1, 100))
        ax.legend(
            fontsize=14, 
            frameon=False, 
            framealpha=1, 
            # loc='upper left',  # Anchor point for bbox_to_anchor
            # bbox_to_anchor=(1, 1),  # Positions legend outside right
            borderaxespad=0.  # Removes padding between legend and plot
        )
        sns.despine()
        plt.tight_layout()

        # Save plot (generic filename)
        plot_filename = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {plot_filename}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot performance comparison of numerical methods')
    parser.add_argument('input_file', type=str, help='csv file to read')
    parser.add_argument("--max-time-ms", type=float, default=500, help="Maximum time to display in milliseconds")
    args = parser.parse_args()
    main(args)

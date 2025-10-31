import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def reinterprete(method):
    mapped = {
        'Ours': 'Ours+CPU',
        'Ours+cuda': 'Ours+CUDA',
        'ainv+cpu': "AINV+CPU",
        'ainv+cuda': "AINV+CUDA",
        'ic+cpu': "IC+CPU",
        'ic+cuda': "IC+CUDA",
        'diagonal+cpu': "Diag+CPU",
        'diagonal+cuda': "Diag+CUDA",
        'none+cpu': "None+CPU",
        'none+cuda': "None+CUDA",
    }
    if not isinstance(method, str):
        output = [mapped.get(m, m) for m in method]
        return output
    return mapped.get(method, method)

def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot performance comparison of numerical methods')
    parser.add_argument('case', type=str, help='Case name')
    parser.add_argument('filename', type=str, help='filename')
    parser.add_argument('rtols', type=str, nargs='+', help='Tolerance values (-log10)')
    args = parser.parse_args()

    # 2. Collect all data
    data_frames = []
    precond_times = {}
    
    for rtol in args.rtols:
        file_path = f"output/{args.case}/infer_{args.filename}_{rtol}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            continue
        
        df = pd.read_csv(file_path)
        df['rtol'] = float(rtol)
        df['error'] = 10 ** (-df['rtol'])
        data_frames.append(df)
        
        # Store preconditioning time for each algorithm
        for algo in df['Key'].unique():
            if algo not in precond_times:
                precond_times[algo] = df[df['Key'] == algo]['Precond Time (ms)'].values[0]
    
    if not data_frames:
        print("Error: No data files found")
        return
    
    all_data = pd.concat(data_frames)
    
    # 3. Prepare plots
    algorithms = all_data['Key'].unique()
    metrics = [
        "#Iteration",
        # "Solve Time (ms)",
        "Total Time (ms)",
    ]
    metric_names = [
        "Iteration Count",
        # "Solve Time (ms)",
        "Total Time (ms)",
    ]

    # Set global font sizes
    plt.rc('font', size=14)           # Controls default text sizes (e.g., labels)
    plt.rc('axes', titlesize=15)      # Font size of the axes title
    plt.rc('axes', labelsize=15)      # Font size of the x and y labels
    plt.rc('xtick', labelsize=12)     # Font size of the tick labels
    plt.rc('ytick', labelsize=12)     # Font size of the tick labels
    plt.rc('legend', fontsize=12)  
    largest_total_time_per_algo = all_data.groupby('Key')['Total Time (ms)'].max()
    # Find the second largest solve time across all algorithms
    second_largest_total_time = largest_total_time_per_algo.nlargest(2).values[-1]
    print(f"Second largest solve time: {second_largest_total_time}")
    total_time_max = second_largest_total_time * 1.2
    
    largest_solve_time_per_algo = all_data.groupby('Key')['Solve Time (ms)'].max()
    # Find the second largest solve time across all algorithms
    second_largest_solve_time = largest_solve_time_per_algo.nlargest(2).values[-1]
    print(f"Second largest solve time: {second_largest_solve_time}")
    solve_time_max = second_largest_solve_time * 1.2

    # Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
    colors = ['g', 'r', 'c', 'y', 'm', 'k']
    handles, labels = [], []
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]

        for i, algo in enumerate(algorithms):
            algo_data = all_data[all_data['Key'] == algo].sort_values(by='rtol')
            if 'cuda' in algo.lower() and 'iteration' in metric.lower():
                continue
            # Prepare data including preconditioning point
            if metric == 'Total Time (ms)':
                x_values = [precond_times[algo]] + list(algo_data[metric])
                y_values = [1.0] + list(algo_data['error'])
            else:
                x_values = [0] + list(algo_data[metric])  # 0 iterations for precond-only
                y_values = [1.0] + list(algo_data['error'])

            if algo.startswith('PCG-'):
                algo = algo[4:]  # Remove 'PCG-' prefix for plotting
            algo = algo.replace('-', '+')
            algo = algo.lower()
            algo = algo.replace('neural', 'Ours')
            if metric == '#Iteration':
                algo = algo.replace('+cpu', '')
            if metric != 'Precond Time (ms)':
                # Plot connected line
                linestyle = '--' if 'cuda' in algo.lower() else '-'
                marker = markers[(i // 2) % len(markers)]
                color = colors[(i // 2) % len(colors)]
                algo = reinterprete(algo)
                ax.plot(x_values, y_values, marker + linestyle, label=algo, color=color, markersize=5)
                if 'Solve Time (ms)' == metric:
                    ax.set_xlim(0, solve_time_max)
                elif 'Total Time (ms)' == metric:
                    ax.set_xlim(0, total_time_max)
            else:
                # Plot bars
                ax.bar(np.arange(len(x_values) - 1), x_values[1:], label=algo, color=colors[i % len(colors)], alpha=0.5)

        ax.set_xlabel(name)
        ax.set_ylabel('Relative Error')
        ax.set_yscale('log')
        ax.set_title(f'Error vs {name}')
        ax.grid(True, which="both", ls="--")
        # if metric == '#Iteration':
        #     ax.legend()
        if metric == 'Total Time (ms)':
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            ax.legend(fontsize='small')
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small')
    plt.tight_layout()
    filename = f'plots/convergence_{args.case}_{args.filename}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.show()
    plt.savefig(filename, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()

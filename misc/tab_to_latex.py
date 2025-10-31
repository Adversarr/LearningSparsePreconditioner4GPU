r"""
input: all_files in main. each file is a CSV file with the following columns:

Key,Total Time (ms),Solve Time (ms),Precond Time (ms),#Iteration
PCG-ainv-cpu,71.7201,51.9084,19.8117,172.3311
PCG-ainv-cuda,48.7994,29.4784,19.321,172.2742
PCG-ic-cpu,49.6197,41.0251,8.5946,109.4357
PCG-ic-cuda,81.7196,79.8105,1.9092,99.954
PCG-diagonal-cpu,37.1003,36.9728,0.1275,265.5446
PCG-diagonal-cuda,37.7182,37.5141,0.2041,265.5599
PCG-none-cpu,69.5831,69.5813,0.0018,474.2616
PCG-none-cuda,66.6622,66.4738,0.1884,474.468
Neural,48.8164,48.6279,0.1886,113.3774
Neural+CUDA,20.4926,20.304,0.1886,113.3774

Output should be:

\begin{table}[htbp]
\centering
\caption{Comparison between different PCG preconditioners. The total time (ms) and iterations for the heat problem are reported here. The best value is in bold, and the lower value indicates better performance.}
\label{tab:time-stats}
\begin{tabular}{ccccccccc}
\toprule
    \multirow{2}{*}{Stage}  & \multicolumn{4}{c}{CPU} & \multicolumn{4}{c}{GPU}\\
    & Diag. & IC & AINV & Ours & Diag. & IC & AINV & Ours\\
\midrule
Construction&  \\
$10^{-2}$   & TOTAL-TIME(ITERATION) & TOTAL-TIME(ITERATION) & TOTAL-TIME(ITERATION) & ... \\ <-- come frome the input file 2
$10^{-4}$   &  \\ <-- come frome the input file 4
$10^{-6}$   &  \\ <-- come frome the input file 6
$10^{-8}$   &  \\ <-- come frome the input file 8
\bottomrule
\end{tabular}
\end{table}
"""

from argparse import ArgumentParser
import csv
import os

def read_csv_data(file_path):
    data = {}
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['Key']] = {
                'total': float(row['Total Time (ms)']),
                'iter': float(row['#Iteration']),
                'construction': float(row['Precond Time (ms)']),
            }
    return data

def find_best_values(file_groups):
    best_values = {}
    for group_name, group_data in file_groups.items():
        best_total = min(d['total'] for d in group_data.values())
        best_iter = min(d['iter'] for d in group_data.values())
        best_values[group_name] = {
            'best_total': best_total,
            'best_iter': best_iter
        }
    return best_values

def format_value(total, iter, best_total, best_iter):
    total_str = f"{total:.0f}"# if total != best_total else f"\\textbf{{{total:.1f}}}"
    iter_str = f"{int(iter)}"# if iter != best_iter else f"\\textbf{{{int(iter)}}}"
    return f"{total_str}({iter_str})"

def main(args):
    all_files = [f"{args.input}_{i}.csv" for i in [2, 4, 6, 8]]
    file_groups = {
        'PCG-ic-cpu': [],
        'PCG-ainv-cpu': [],
        'PCG-diagonal-cpu': [],
        'Neural': [],
        'PCG-ic-cuda': [],
        'PCG-ainv-cuda': [],
        'PCG-diagonal-cuda': [],
        'Neural+CUDA': []
    }

    # Read all files and group data by method
    all_data = []
    for file in all_files:
        all_data.append(read_csv_data(file))

    # Find best values for each column
    best_values = {
        'PCG-diagonal-cpu': {'best_total': min(d['PCG-diagonal-cpu']['total'] for d in all_data),
                           'best_iter': min(d['PCG-diagonal-cpu']['iter'] for d in all_data)},
        'PCG-ic-cpu': {'best_total': min(d['PCG-ic-cpu']['total'] for d in all_data),
                     'best_iter': min(d['PCG-ic-cpu']['iter'] for d in all_data)},
        'PCG-ainv-cpu': {'best_total': min(d['PCG-ainv-cpu']['total'] for d in all_data),
                       'best_iter': min(d['PCG-ainv-cpu']['iter'] for d in all_data)},
        'Neural': {'best_total': min(d['Neural']['total'] for d in all_data),
                 'best_iter': min(d['Neural']['iter'] for d in all_data)},
        'PCG-diagonal-cuda': {'best_total': min(d['PCG-diagonal-cuda']['total'] for d in all_data),
                            'best_iter': min(d['PCG-diagonal-cuda']['iter'] for d in all_data)},
        'PCG-ic-cuda': {'best_total': min(d['PCG-ic-cpu']['total'] for d in all_data),
                      'best_iter': min(d['PCG-ic-cpu']['iter'] for d in all_data)},
        'PCG-ainv-cuda': {'best_total': min(d['PCG-ainv-cuda']['total'] for d in all_data),
                        'best_iter': min(d['PCG-ainv-cuda']['iter'] for d in all_data)},
        'Neural+CUDA': {'best_total': min(d['Neural+CUDA']['total'] for d in all_data),
                      'best_iter': min(d['Neural+CUDA']['iter'] for d in all_data)},
    }

    # Generate LaTeX table
    print(r"""\begin{table}[htbp]
\centering
\caption{Comparison between different PCG preconditioners. The total time (ms) and iterations for the heat problem are reported here. The best value is in bold, and the lower value indicates better performance.}
\label{tab:time-stats}
\begin{tabular}{ccccccccc}
\toprule
    \multirow{2}{*}{Stage}  & \multicolumn{4}{c}{CPU} & \multicolumn{4}{c}{GPU}\\
    & Diag. & IC & AINV & Ours & Diag. & IC & AINV & Ours\\
\midrule""")
    # precompute time:
    row = r"Construction& "
    row += f'{all_data[0]['PCG-diagonal-cpu']['construction']:.3f} & '
    row += f'{all_data[0]['PCG-ic-cpu']['construction']:.3f} & '
    row += f'{all_data[0]['PCG-ainv-cpu']['construction']:.3f} & '
    row += f'{all_data[0]['Neural']['construction']:.3f} & '
    row += f'{all_data[0]['PCG-diagonal-cuda']['construction']:.3f} & '
    row += f'{all_data[0]['PCG-ic-cuda']['construction']:.3f} & '
    row += f'{all_data[0]['PCG-ainv-cuda']['construction']:.3f} & '
    row += f'{all_data[0]['Neural+CUDA']['construction']:.3f}'
    print(row + r" \\")

    # Iterate through each file and format the data

    for i, (file, data) in enumerate(zip(all_files, all_data)):
        tol = file.split('_')[-1].split('.')[0]  # Extract tolerance from filename
        row = f"$10^{{-{tol}}}$ & "
        
        # CPU columns
        row += format_value(data['PCG-diagonal-cpu']['total'], data['PCG-diagonal-cpu']['iter'],
                          best_values['PCG-diagonal-cpu']['best_total'], best_values['PCG-diagonal-cpu']['best_iter']) + " & "
        row += format_value(data['PCG-ic-cpu']['total'], data['PCG-ic-cpu']['iter'],
                          best_values['PCG-ic-cpu']['best_total'], best_values['PCG-ic-cpu']['best_iter']) + " & "
        row += format_value(data['PCG-ainv-cpu']['total'], data['PCG-ainv-cpu']['iter'],
                          best_values['PCG-ainv-cpu']['best_total'], best_values['PCG-ainv-cpu']['best_iter']) + " & "
        row += format_value(data['Neural']['total'], data['Neural']['iter'],
                          best_values['Neural']['best_total'], best_values['Neural']['best_iter']) + " & "
        
        # GPU columns
        row += format_value(data['PCG-diagonal-cuda']['total'], data['PCG-diagonal-cuda']['iter'],
                          best_values['PCG-diagonal-cuda']['best_total'], best_values['PCG-diagonal-cuda']['best_iter']) + " & "
        row += format_value(data['PCG-ic-cuda']['total'], data['PCG-ic-cuda']['iter'],
                          best_values['PCG-ic-cuda']['best_total'], best_values['PCG-ic-cuda']['best_iter']) + " & "
        row += format_value(data['PCG-ainv-cuda']['total'], data['PCG-ainv-cuda']['iter'],
                          best_values['PCG-ainv-cuda']['best_total'], best_values['PCG-ainv-cuda']['best_iter']) + " & "
        row += format_value(data['Neural+CUDA']['total'], data['Neural+CUDA']['iter'],
                          best_values['Neural+CUDA']['best_total'], best_values['Neural+CUDA']['best_iter'])
        
        print(row + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input', type=str, help='Input file prefix')
    main(parser.parse_args())
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
\begin{table}[h!]
\caption{Benchmark result on different datasets. Total time (ms) and total iterations (in parentheses) are listed in the table. The lower value indicates better performance.}
\label{tab:benchmark}
\centering
\begin{tabular}{ccccccccc}
\toprule
    \multirow{2}{*}{Test Case}  & \multicolumn{4}{c}{CPU} & \multicolumn{4}{c}{GPU}\\
    & Diag & IC & AINV & Ours & Diag & IC & AINV & Ours\\
\midrule
    Heat Equation& T(N)     \\  <-- come from the input file 1
    Poisson Equation&   \\      <-- come from the input file 2
    Hyperelasticity&    \\      <-- come from the input file 3
    Synthetic System&   \\      <-- come from the input file 4
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
            print(row)
            data[row['Key']] = {
                'total': float(row['Total Time (ms)']),
                'iter': float(row['#Iteration']),
            }
    return data

def format_value(total, iter):
    total_str = f"{total:.0f}"# if total != best_total else f"\\textbf{{{total:.1f}}}"
    iter_str = f"{int(iter)}"# if iter != best_iter else f"\\textbf{{{int(iter)}}}"
    return f"{total_str}({iter_str})"


def generate_latex_table(file_groups) -> str:
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison between different PCG preconditioners. The total time (ms) and iterations for the heat problem are reported here. The best value is in bold, and the lower value indicates better performance.}
\label{tab:time-stats}
\begin{tabular}{ccccccccc}
\toprule
    \multirow{2}{*}{Test Case}  & \multicolumn{4}{c}{CPU} & \multicolumn{4}{c}{GPU}\\
    & Diag & IC & AINV & Ours & Diag & IC & AINV & Ours\\
\midrule
"""
    latex_table += "Heat Equation& "
    idx = 0
    latex_table += format_value(file_groups['PCG-diagonal-cpu'][idx]['total'], file_groups['PCG-diagonal-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cpu'][idx]['total'], file_groups['PCG-ic-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cpu'][idx]['total'], file_groups['PCG-ainv-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural'][idx]['total'], file_groups['Neural'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-diagonal-cuda'][idx]['total'], file_groups['PCG-diagonal-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cuda'][idx]['total'], file_groups['PCG-ic-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cuda'][idx]['total'], file_groups['PCG-ainv-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural+CUDA'][idx]['total'], file_groups['Neural+CUDA'][idx]['iter']) + r" \\"
    
    latex_table += "\n"
    latex_table += "Poisson Equation& "
    idx = 1
    latex_table += format_value(file_groups['PCG-diagonal-cpu'][idx]['total'], file_groups['PCG-diagonal-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cpu'][idx]['total'], file_groups['PCG-ic-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cpu'][idx]['total'], file_groups['PCG-ainv-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural'][idx]['total'], file_groups['Neural'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-diagonal-cuda'][idx]['total'], file_groups['PCG-diagonal-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cuda'][idx]['total'], file_groups['PCG-ic-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cuda'][idx]['total'], file_groups['PCG-ainv-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural+CUDA'][idx]['total'], file_groups['Neural+CUDA'][idx]['iter']) + r" \\"
    
    latex_table += "\n"
    latex_table += "Hyperelasticity& "
    idx = 2
    latex_table += format_value(file_groups['PCG-diagonal-cpu'][idx]['total'], file_groups['PCG-diagonal-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cpu'][idx]['total'], file_groups['PCG-ic-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cpu'][idx]['total'], file_groups['PCG-ainv-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural'][idx]['total'], file_groups['Neural'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-diagonal-cuda'][idx]['total'], file_groups['PCG-diagonal-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cuda'][idx]['total'], file_groups['PCG-ic-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cuda'][idx]['total'], file_groups['PCG-ainv-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural+CUDA'][idx]['total'], file_groups['Neural+CUDA'][idx]['iter']) + r" \\"
    
    latex_table += "\n"
    latex_table += "Synthetic System& "
    idx = 3
    latex_table += format_value(file_groups['PCG-diagonal-cpu'][idx]['total'], file_groups['PCG-diagonal-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cpu'][idx]['total'], file_groups['PCG-ic-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cpu'][idx]['total'], file_groups['PCG-ainv-cpu'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural'][idx]['total'], file_groups['Neural'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-diagonal-cuda'][idx]['total'], file_groups['PCG-diagonal-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ic-cuda'][idx]['total'], file_groups['PCG-ic-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['PCG-ainv-cuda'][idx]['total'], file_groups['PCG-ainv-cuda'][idx]['iter']) + r" & "
    latex_table += format_value(file_groups['Neural+CUDA'][idx]['total'], file_groups['Neural+CUDA'][idx]['iter']) + r" \\"
    
    
    latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex_table

def main(args):
    all_files = list(args.inputs)
    file_groups = {
        'PCG-none-cpu': [],
        'PCG-ic-cpu': [],
        'PCG-ainv-cpu': [],
        'PCG-diagonal-cpu': [],
        'PCG-none-cuda': [],
        'PCG-ic-cuda': [],
        'PCG-ainv-cuda': [],
        'PCG-diagonal-cuda': [],
        'Neural': [],
        'Neural+CUDA': [],
    }
    
    # Read data from files
    for file_path in all_files:
        data = read_csv_data(file_path)
        for key, value in data.items():
            if 'cpu' in key:
                file_groups[key].append(value)
            elif 'cuda' in key:
                file_groups[key].append(value)
            elif 'Neural' == key:
                file_groups[key].append(value)
            elif 'Neural+CUDA' == key:
                file_groups[key].append(value)

    # Generate LaTeX table
    latex_table = generate_latex_table(file_groups)
    print(latex_table)

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate LaTeX table from CSV files.")
    parser.add_argument("inputs", nargs='+', help="Input CSV files.")
    args = parser.parse_args()
    main(args)
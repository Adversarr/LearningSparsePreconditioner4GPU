"""
Input CSV:

neural,none,diag,ainv,ichol
913.589685921165,139671372.24988067,35245.43634202137,11279.882195408463,2072.655737083117
...

---

Original Plot Code:

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=table)
plt.title("Condition Number Distribution")
plt.ylabel("Condition Number")
if name == 'cond':
    plt.yscale('log')
plt.savefig(f"{name}_cond_{exp_name}.png", dpi=300)

---

Plot with larger font size.
"""

from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_condition_number_distribution(csv_file, save_name, name, visualize=False):
    # Load the data
    table = pd.read_csv(csv_file)
    
    # Set the style and create the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # replace keys:
    # neural -> Neural
    table.columns = [col.replace('neural', 'Neural') for col in table.columns]
    # none -> None
    table.columns = [col.replace('none', 'None') for col in table.columns]
    # diag -> Diag
    table.columns = [col.replace('diag', 'Diag') for col in table.columns]
    # ainv -> AINV
    table.columns = [col.replace('ainv', 'AINV') for col in table.columns]
    # ichol -> IC
    table.columns = [col.replace('ichol', 'IC') for col in table.columns]
    
    sns.boxplot(data=table)
    # plt.title("Condition Number Distribution", fontsize=16)
    plt.ylabel("Condition Number", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if name == 'cond':
        plt.yscale('log')
    plt.savefig(f"{save_name}.png", dpi=300)
    if visualize:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot condition number distribution from CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing condition numbers.")
    parser.add_argument("save_name", type=str, help="Base name for the output files.")
    
    args = parser.parse_args()
    
    plot_condition_number_distribution(args.csv_file, args.save_name, name='kaporin')
"""
Input Frame:

Key,Solve Time (ms),Precond Time (ms),#Iteration,Matrix Size
Neural,1296.751,3.0639,468.0,49152
Neural,...
Neural+CUDA,...
Neural+CUDA,...

Output Frame:
Key,Total Time (ms),Solve Time (ms),Precond Time (ms),#Iteration
Neural,27.8821,24.8441,3.038,56.01
Neural+CUDA,13.7193,10.6813,3.038,56.01
"""
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    df = pd.read_csv(args.csv)
    print(df)
    print(df.describe())
    
    # Filter the DataFrame based on the size range
    filtered_df = df[(df['Matrix Size'] >= args.min_size) & (df['Matrix Size'] <= args.max_size)]
    
    # Group the filtered DataFrame by 'Key' and calculate the mean solve time
    mean_solve_time = filtered_df.groupby('Key')['Solve Time (ms)'].mean()
    mean_precond_time = filtered_df.groupby('Key')['Precond Time (ms)'].mean()
    mean_iteration = filtered_df.groupby('Key')['#Iteration'].mean()
    mean_total_time = mean_solve_time + mean_precond_time

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({
        'Key': mean_solve_time.index,
        'Solve Time (ms)': mean_solve_time,
        'Precond Time (ms)': mean_precond_time,
        'Total Time (ms)': mean_total_time,
        '#Iteration': mean_iteration
    })
    # Print the resulting DataFrame
    print(result_df)
    if args.output is not None:
        result_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv', type=str)
    parser.add_argument('min_size', type=int)
    parser.add_argument('max_size', type=int)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
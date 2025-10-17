import logging
import sys
import os
import time

import pandas as pd
import numpy as np

from bcause.factors import MultinomialFactor, DeterministicFactor
from bcause.models.cmodel import StructuralCausalModel
from bcause.learning.parameter import expectation_maximization as em


log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')

# Import and return the seeds given a path as integer
def import_seeds(path):
    with open(path, 'r') as f:
        seeds = [int(line.strip()) for line in f.readlines()]
    return seeds

def factor_as_list(factor: MultinomialFactor):
    return MultinomialFactor(domain=factor.domain, values=factor.values, left_vars=factor.left_vars,
                                     right_vars=factor.right_vars, vtype="list")

# Create a dataframe with columns model_path, data_path, method, time, iterations if the dataframe does not exist in the path, otherwise load it
def create_benchmark_results_df(path):
    if os.path.exists(os.path.join(path, 'benchmark_results.csv')):
        # Load the dataframe from the csv file
        print("Loading existing benchmark results...")
        df = pd.read_csv(os.path.join(path, 'benchmark_results.csv'))
    else:
        print("Creating new benchmark results dataframe...")
        df = pd.DataFrame(columns=["model_path", "data_path", "method", "time", "iterations", "threshold"])
    return df

# Create a dictionary with the model and data names
def create_model_data_dict(graph_code, path, seeds):
    files = os.listdir(path)
    models_data_dict = {}
    for file in files:
        if file.endswith('.bif') and file.startswith(graph_code) and int(file.split('_')[-1].split('.')[0]) in seeds:
            data_file = file.replace("model", "data").replace('.bif', '.csv')
            models_data_dict[file] = data_file
    return models_data_dict

def get_seeds(selected_seeds:list,df:pd.DataFrame):
    existing_seeds = set()
    for model_path in df['model_path'].unique():
        seed = int(model_path.split('_')[-1].split('.')[0])
        existing_seeds.add(seed)
    return [seed for seed in selected_seeds if seed not in existing_seeds]

# Main function
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Define the path to the models and data
    path = "/Users/antoniogonzalezalves/Documents/BenchMarkWUPES"
    seeds = import_seeds(os.path.join(path, "seeds.txt"))
    step = 1
    iterations = range(1,101,step)
    graph_codes = ["g2"]
    df = create_benchmark_results_df(path)
    selected_seeds = get_seeds(seeds[10:20],df)
    first = True
    for graph_code in graph_codes:
        models_data_dict = create_model_data_dict(graph_code, path, selected_seeds)
        for model_name in models_data_dict.keys():
            print(f"Running model {model_name}")
            model_path = os.path.join(path, model_name)
            data_path = os.path.join(path, models_data_dict[model_name])
            data = pd.read_csv(data_path, dtype=str)
            # Create the model
            model = StructuralCausalModel.read(model_path)
            # Convert all factors to list type
            model_list = model.copy()
            for v in model_list.variables:
                model_list.factors[v] = factor_as_list(model_list.factors[v])

            threshold_list = [0.0, 0.05, 0.1]
            # List of algorithms to run
            algo_list = ["EMP","EMPList", "EMPBtree","EM", "EMList", "Btree"]

            for algo in algo_list:
                if "Btree" in algo:
                    em_instance = em.ExpectationMaximizationTrees(model, ignore_convergence=True, combine_steps=False)
                elif "EMP" in algo and "List" in algo:
                    em_instance = em.ExpectationMaximizationPrecomputed(model_list, ignore_convergence=True, as_list=True)
                elif "EMP" in algo:
                    em_instance = em.ExpectationMaximizationPrecomputed(model, ignore_convergence=True)
                elif "EM" in algo and "List" in algo:
                    em_instance = em.ExpectationMaximization(model_list, ignore_convergence=True, vtype="list")
                else:
                    em_instance = em.ExpectationMaximization(model, ignore_convergence=True)

                if algo == "EMPBtree":
                    for threshold in threshold_list:
                        print(f"Running {algo} for {model_name} with threshold {threshold}")
                        em_instance.threshold = threshold
                        start_time = time.time()
                        em_instance.initialize(data[model.endogenous])
                        em_time_total = time.time() - start_time
                        # print(f"{algo} initialization time: {em_time_total:.4f} seconds")
                        df = pd.concat([df, pd.DataFrame([[model_name, models_data_dict[model_name], algo, em_time_total, 0,threshold]],
                                                         columns=df.columns)], ignore_index=True)

                        for iteration in iterations:
                            # Time the initialization of EM
                            start_time = time.time()
                            em_instance.step()
                            em_time_total += time.time() - start_time
                            # print every 10 iterations
                            # if iteration % 10 == 0:
                                # print(f"{algo} time for iteration {iteration}: {em_time_total:.4f} seconds")

                            # Add the dataframe with the results to df using concat
                            df = pd.concat([df, pd.DataFrame([[model_name, models_data_dict[model_name], algo, em_time_total, iteration,threshold]], columns=df.columns)], ignore_index=True)
                            # Save the dataframe to a csv file
                            df.to_csv(os.path.join(path, 'benchmark_results_3.csv'), index=False)


                else:
                    print(f"Running {algo} for {model_name}")
                    start_time = time.time()
                    em_instance.initialize(data[model.endogenous])
                    em_time_total = time.time() - start_time
                    # print(f"{algo} initialization time: {em_time_total:.4f} seconds")
                    df = pd.concat([df, pd.DataFrame([[model_name, models_data_dict[model_name], algo, em_time_total, 0,np.nan]],
                                                     columns=df.columns)], ignore_index=True)

                    for iteration in iterations:
                        # Time the initialization of EM
                        start_time = time.time()
                        em_instance.step()
                        em_time_total += time.time() - start_time
                        # print every 10 iterations
                        # if iteration % 10 == 0:
                            # print(f"{algo} time for iteration {iteration}: {em_time_total:.4f} seconds")

                        # Add the dataframe with the results to df using concat
                        df = pd.concat([df, pd.DataFrame([[model_name, models_data_dict[model_name], algo, em_time_total, iteration,np.nan]], columns=df.columns)], ignore_index=True)
                        # Save the dataframe to a csv file
                        df.to_csv(os.path.join(path, 'benchmark_results_3.csv'), index=False)
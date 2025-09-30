import os
import json
import torch
import wandb
import shutil
import socket
import logging
import datetime
import argparse
from swarm import Swarm
from utils import data_generation_objective

def log_with_flush(message, level=logging.INFO):
  logging.log(level, message)
  logging.getLogger().handlers[0].flush()

def curret_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def assign_gpu(num_gpus, process_idx, total_processes):
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this data swarms search, also directory name in search/")
    argParser.add_argument("-t", "--task", help="name of the task/dataset") # alpaca, gsm8k, truthfulqa, wikidyk
    argParser.add_argument("-o", "--objective", help="name of the objective") # difficult, separate, novel, stable
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string") # such as 0,1,2,3
    argParser.add_argument("--inertia", default = 0.4, help="inertia of particle weight update")
    argParser.add_argument("--cognitive_coeff", default = 0.3, help="cognitive coefficient of particle weight update")
    argParser.add_argument("--social_coeff", default = 0.3, help="social coefficient of particle weight update")
    argParser.add_argument("--repel_coeff", default = 0.3, help="repel coefficient of particle weight update")
    argParser.add_argument("--step_length", default = 1, help="step length of the search in the direction of velocity")
    argParser.add_argument("-p", "--patience", default = 10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default = 200, help="max iteration of the search")
    argParser.add_argument("--weight_randomness", default = 1, help="whether to use weight randomess") # 0, 1
    argParser.add_argument("-b", "--base_model", default="google/gemma-2-9b-it", help="base model of the lora experts")
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file") # just keep it 1 unless you absolutely know what you're doing
    argParser.add_argument("--project_name_wb", default="data swarms", help="wandb project name") # as you wish
    argParser.add_argument("--starting_velocity_mode", default="random", help="starting velocity mode: zero, random, best") # zero, random, best
    argParser.add_argument("--repel_term", default=1, help="whether to incorporate a repel term with global_worst") # 0, 1
    argParser.add_argument("--step_length_factor", default=0.95, help="step length *= step_length_factor every iteration") # 1 for no scheduling, 0.95 maybe?
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart particles")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end") # 0, 1

    args = argParser.parse_args()
    search_pass_name = args.name
    task = args.task
    objective = args.objective
    gpus = [int(gpu_id) for gpu_id in args.gpus.split(",")]
    inertia = float(args.inertia)
    cognitive_coeff = float(args.cognitive_coeff)
    social_coeff = float(args.social_coeff)
    repel_coeff = float(args.repel_coeff)
    step_length = float(args.step_length)
    patience = int(args.patience)
    max_iteration = int(args.max_iteration)
    weight_randomness = int(args.weight_randomness)
    base_model = args.base_model
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    starting_velocity_mode = args.starting_velocity_mode
    repel_term = int(args.repel_term)
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = int(args.clean_up_on_end)

    search_pass_name += ("_" + socket.gethostname())
    args.name = search_pass_name

    # create search directory
    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += curret_time_string().replace(" ", "_")
        # exit("search directory already exists!")
    os.mkdir(os.path.join("search", search_pass_name))

    # write args to file
    with open(os.path.join("search", search_pass_name, "args.txt"), "w") as f:
        f.write(str(args))

    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)

    test_generator_paths = []
    for i in range(4):
        test_generator_paths.append(f"initial_experts/{task}_cluster_{i}")
    test_taker_paths = ["initial_experts/cot", "initial_experts/lima", "initial_experts/oasst1", "initial_experts/science"]

    log_with_flush("initializing search... "+curret_time_string())
    data_swarm = Swarm(
        swarm_base_path = os.path.join("search", search_pass_name, "data_swarm"),
        model_paths = test_generator_paths,
        base_model = base_model,
        fast_merge = fast_merge,
        starting_velocity_mode = starting_velocity_mode,
        weight_randomness = weight_randomness,
        inertia = inertia,
        cognitive_coeff = cognitive_coeff,
        social_coeff = social_coeff,
        repel_coeff = repel_coeff,
        step_length = step_length,
        repel_term = repel_term,
        step_length_factor = step_length_factor,
        minimum_step_length = minimum_step_length,
        gpus = gpus,
        patience = patience,
        restart_patience = restart_patience,
    )
    log_with_flush("search initialized")

    test_generator_paths = []
    for i in range(4):
        test_generator_paths.append(os.path.join("search", search_pass_name, "data_swarm", "model_"+str(i), "now"))

    # main search iteration
    iter_count = 0
    while iter_count < max_iteration:
        log_with_flush("--------------------------")
        log_with_flush("iteration "+str(iter_count)+" "+curret_time_string())
        
        # evaluating data generators based on the objective

        scores = data_generation_objective(
            objective_name = objective,
            task = task,
            test_generator_paths = test_generator_paths,
            test_taker_paths = test_taker_paths,
            gpu_ids = gpus,
            base_model = base_model
        )

        log_with_flush("objective evaluation done! scores: "+str(scores))

        # update the swarm

        termination_flag = data_swarm.update(scores)
        log_with_flush("swarm updated")

        with open(os.path.join("search", search_pass_name, "data_swarm", "utility_scratchpad.json"), "r") as f:
            utility_scratchpad = json.load(f)

        wandb_log = {
            "g": utility_scratchpad["g"],
            "g_worst": utility_scratchpad["g_worst"],
        }
        for i in range(len(scores)):
            wandb_log["model_"+str(i) + "_now"] = utility_scratchpad["model_"+str(i) + "_now"]

        wandb.log(wandb_log)
        
        if termination_flag:
            break

        iter_count += 1
    
    log_with_flush("ending search... "+curret_time_string())

    if clean_up_on_end:
        shutil.rmtree(os.path.join("search", search_pass_name, "data_swarm", "global_worst"))
        for i in range(len(test_generator_paths)):
            for aux in ["g_x", "p_x", "velocity", "x_w"]:
                shutil.rmtree(os.path.join("search", search_pass_name, "data_swarm", "model_"+str(i), aux))

    log_with_flush("the end of search... "+curret_time_string())
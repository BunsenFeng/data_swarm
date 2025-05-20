# define the swarm class
# managing initialization and update of the swarm

import os
import json
import math
import torch
import shutil
import random
from merge import lora_merge
from multiprocessing import Pool

def assign_gpu(num_gpus, process_idx, total_processes):
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

class Swarm:

    def __init__(self, swarm_base_path, model_paths, base_model, fast_merge, starting_velocity_mode, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, step_length, repel_term, step_length_factor, minimum_step_length, gpus, patience, restart_patience):

        self.swarm_base_path = swarm_base_path
        os.mkdir(self.swarm_base_path)
        self.model_paths = model_paths
        self.base_model = base_model
        self.fast_merge = fast_merge
        self.starting_velocity_mode = starting_velocity_mode
        self.weight_randomness = weight_randomness
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.repel_coeff = repel_coeff
        self.step_length = step_length
        self.repel_term = repel_term
        self.step_length_factor = step_length_factor
        self.minimum_step_length = minimum_step_length
        self.gpus = gpus
        self.patience = patience
        self.restart_patience = restart_patience
        self.restart_counter = [0] * len(self.model_paths)

        # initialize utility scratchpad
        self.utility_scratchpad = {"g": None, "g_worst": None, "g_history": []}
        for i in range(len(self.model_paths)):
            self.utility_scratchpad[f"model_{i}_now"] = None
            self.utility_scratchpad[f"model_{i}_best"] = None
            self.utility_scratchpad[f"model_{i}_history"] = []
        
        with open(os.path.join(swarm_base_path, "utility_scratchpad.json"), "w") as f:
            json.dump(self.utility_scratchpad, f, indent=4)

        # iniitalize the directories for the swarm
        for i in range(len(self.model_paths)):
            os.mkdir(os.path.join(self.swarm_base_path, "model_" + str(i)))
            for checkpoint_type in ["personal_best", "now", "velocity"]:
                os.mkdir(os.path.join(self.swarm_base_path, "model_" + str(i), checkpoint_type))
        os.mkdir(os.path.join(self.swarm_base_path, "global_best"))
        os.mkdir(os.path.join(self.swarm_base_path, "global_worst"))

        # initialize model now weights and personal_best
        for i in range(len(self.model_paths)):
            shutil.copytree(self.model_paths[i], os.path.join(self.swarm_base_path, "model_" + str(i), "now"), dirs_exist_ok=True)
            shutil.copytree(self.model_paths[i], os.path.join(self.swarm_base_path, "model_" + str(i), "personal_best"), dirs_exist_ok=True)

        # initialize model velocity
        if starting_velocity_mode == "random":
            merge_args = []
            for i in range(len(self.model_paths)):
                secret_lover_id = random.randint(0, len(self.model_paths) - 1)
                while secret_lover_id == i:
                    secret_lover_id = random.randint(0, len(self.model_paths) - 1)
                merge_args.append(([-1,1], [os.path.join(self.swarm_base_path, "model_" + str(i), "now"), os.path.join(self.swarm_base_path, "model_" + str(secret_lover_id), "now")], os.path.join(self.swarm_base_path, "model_" + str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(self.model_paths))], fast_merge))

            with Pool(len(gpus)) as p:
                p.starmap(lora_merge, merge_args, chunksize = math.ceil(len(model_paths)/len(gpus)))
        else:
            raise NotImplementedError

        # no starting utility eval: will eval then update in iteration 1

        # initialize global best and global worst
        for checkpoint_type in ["global_best", "global_worst"]:
            random_idx = random.randint(0, len(self.model_paths) - 1)
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(random_idx), "now"), os.path.join(self.swarm_base_path, checkpoint_type), dirs_exist_ok=True)

    def update(self, scores):
        assert len(scores) == len(self.model_paths)

        # update utility scratchpad
        for i in range(len(self.model_paths)):
            self.utility_scratchpad[f"model_{i}_now"] = scores[i]
            self.utility_scratchpad[f"model_{i}_history"].append(scores[i])
            if self.utility_scratchpad[f"model_{i}_best"] is None or scores[i] > self.utility_scratchpad[f"model_{i}_best"]:
                self.utility_scratchpad[f"model_{i}_best"] = scores[i]
                shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(i), "now"), os.path.join(self.swarm_base_path, "model_" + str(i), "personal_best"), dirs_exist_ok=True)
        
        if self.utility_scratchpad["g"] is None or max(scores) > self.utility_scratchpad["g"]:
            self.utility_scratchpad["g"] = max(scores)
            best_idx = scores.index(max(scores))
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(best_idx), "now"), os.path.join(self.swarm_base_path, "global_best"), dirs_exist_ok=True)
        if self.utility_scratchpad["g_worst"] is None or min(scores) < self.utility_scratchpad["g_worst"]:
            self.utility_scratchpad["g_worst"] = min(scores)
            worst_idx = scores.index(min(scores))
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(worst_idx), "now"), os.path.join(self.swarm_base_path, "global_worst"), dirs_exist_ok=True)
        self.utility_scratchpad["g_history"].append(self.utility_scratchpad["g"])

        with open(os.path.join(self.swarm_base_path, "utility_scratchpad.json"), "w") as f:
            json.dump(self.utility_scratchpad, f, indent=4)

        # if "g_history" did not improve in patience iterations, terminate signal
        if len(self.utility_scratchpad["g_history"]) > self.patience and self.utility_scratchpad["g_history"][-1] <= self.utility_scratchpad["g_history"][-self.patience]:
            termination_flag = True
            return termination_flag
        
        for i in range(len(self.model_paths)):
            
            base_path = self.swarm_base_path
            model_path = os.path.join(self.swarm_base_path, "model_" + str(i))
            now_path = os.path.join(model_path, "now")
            best_path = os.path.join(model_path, "personal_best")
            velocity_path = os.path.join(model_path, "velocity")

            # judge restart flag
            if len(self.utility_scratchpad[f"model_{i}_history"]) > self.restart_patience and self.utility_scratchpad[f"model_{i}_history"][-1] <= self.utility_scratchpad[f"model_{i}_history"][-int(self.restart_patience)] and self.restart_counter[i] == 0:
                restart_flag = True
                self.restart_counter[i] = 3
            else:
                restart_flag = False
                self.restart_counter[i] = max(0, self.restart_counter[i] - 1)

            if restart_flag:
                shutil.copytree(best_path, now_path, dirs_exist_ok=True)
                lora_merge([0], [now_path], velocity_path, self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))], self.fast_merge)
            
            # weight randomness
            if self.weight_randomness:
                r_w = random.uniform(0, 1)
                r_p = random.uniform(0, 1)
                r_s = random.uniform(0, 1)
                r_b = random.uniform(0, 1) # b for bad, repel term weight
            else:
                r_w = 1
                r_p = 1
                r_s = 1
                r_b = 1
            
            # weight normalize
            self_weight = r_w * self.inertia
            cognitive_weight = r_p * self.cognitive_coeff
            social_weight = r_s * self.social_coeff
            repel_weight = r_b * self.repel_coeff if self.repel_term else 0
            weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

            self_weight /= weight_sum
            cognitive_weight /= weight_sum
            social_weight /= weight_sum
            repel_weight /= weight_sum

            # p_i-x_i task vector
            lora_merge(
                weights = [1, -1],
                lora_name_list = [os.path.join(model_path, "personal_best"), os.path.join(model_path, "now")],
                output_path = os.path.join(model_path, "p_x"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge
            )

            # g-x_i task vector
            lora_merge(
                weights = [1, -1],
                lora_name_list = [os.path.join(self.swarm_base_path, "global_best"), os.path.join(model_path, "now")],
                output_path = os.path.join(model_path, "g_x"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge
            )

            # x_i-w task vector
            lora_merge(
                weights = [-1, 1],
                lora_name_list = [os.path.join(self.swarm_base_path, "global_worst"), os.path.join(model_path, "now")],
                output_path = os.path.join(model_path, "x_w"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge
            )

            # update velocity
            lora_merge(
                weights = [self_weight, cognitive_weight, social_weight, repel_weight],
                lora_name_list = [
                    os.path.join(model_path, "velocity"),
                    os.path.join(model_path, "p_x"),
                    os.path.join(model_path, "g_x"),
                    os.path.join(model_path, "x_w")
                ],
                output_path = os.path.join(model_path, "velocity"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge
            )

            # update now, current position
            lora_merge(
                weights = [1, self.step_length],
                lora_name_list = [
                    os.path.join(model_path, "now"),
                    os.path.join(model_path, "velocity")
                ],
                output_path = os.path.join(model_path, "now"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge
            )

            # update step length
            self.step_length *= self.step_length_factor
            self.step_length = max(self.step_length, self.minimum_step_length)

        termination_flag = False
        return termination_flag

# demo
# swarm = Swarm(
#     swarm_base_path = "search/temp",
#     model_paths = ["initial_experts/alpaca_cluster_0", "initial_experts/alpaca_cluster_1", "initial_experts/alpaca_cluster_2", "initial_experts/alpaca_cluster_3"],
#     base_model = "google/gemma-2-9b-it",
#     fast_merge = True,
#     starting_velocity_mode = "random",
#     weight_randomness = True,
#     inertia = 0.5,
#     cognitive_coeff = 0.5,
#     social_coeff = 0.5,
#     repel_coeff = 0.5,
#     step_length = 0.5,
#     repel_term = True,
#     gpus = [0,1,2,3],
#     patience = 5,
#     restart_patience = 3
# )

# flag = swarm.update([0.5, 0.6, 0.7, 0.8])
# flag = swarm.update([0.6, 0.7, 0.8, 0.9])
# flag = swarm.update([0.7, 0.8, 0.9, 1.0])
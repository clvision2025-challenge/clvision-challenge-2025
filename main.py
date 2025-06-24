import logging.config
import os
import random
import numpy as np
import gc

import torch
import torch.nn.functional as F
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, load_deepspeed
# from flops_counter.ptflops import get_model_complexity_info

from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime

from strategies.my_trainer_upstream import create_trainer_upstream
from strategies.my_trainer_downstream import create_trainer_downstream
from strategies.my_utils import get_state_dict, configure_online_datastream

def main():
    # =============================================== DO NOT MODIFY THIS PART =========================================================#
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    
    if training_args.scenario == 1:
        train_datalists = json.load(open(f'dataset/Upstream_scenario1/train/dataset-0.json', 'r'))
    elif training_args.scenario == 2:
        train_datalists = json.load(open(f'dataset/Upstream_scenario2/train/dataset-0.json', 'r'))
    else:
        raise ValueError(f"Unknown scenario {training_args.scenario}, only support 1 or 2 now.")
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}

    training_loss = []
    start_time = time.time()
    memory = []
    memory_size = 1000000

    num_iterations = training_args.num_iter
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    torch.cuda.empty_cache()
    
    # ================================================================================================================================#
    
    
    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    model.config.use_cache = False
    
    ##### simulate online memory insertion & get_batch ####
    datalists = configure_online_datastream(train_datalists, num_iterations, training_args, memory, memory_size, total_batchsize)
    
    print("len(train_datalists)", len(train_datalists), "len(datalists)", len(datalists))
    data_module = make_supervised_data_module(client_data=datalists, # sub_dataset
                                        tokenizer=tokenizer,
                                        data_args=copy.deepcopy(data_args))
    # create trainer
    trainer = create_trainer_upstream(model, tokenizer, training_args, data_module)
    
    # flops_dict = get_model_complexity_info(trainer, (3, 336, 336),
    #                                         as_strings=False,
    #                                         print_per_layer_stat=False, verbose=True,
    #                                         criterion=trainer.get_loss_func(),
    #                                         original_opt=trainer.get_optimizer(),
    #                                         opt_name="adam", lr=0.0001, llava=True)
    # trainer.set_flops_dict(flops_dict)
    
    
    results = trainer.train()
    training_loss.append(results.training_loss)
    
    # save local model
    state_dict = get_state_dict(training_args, model)
    os.makedirs(f"{training_args.state_dir}", exist_ok=True)
    output_dir = os.path.join(training_args.state_dir, f"Upstream_{training_args.scenario}.pth")

    if (training_args.local_rank == 0 or training_args.local_rank == -1):
        torch.save(state_dict, output_dir)
    
    # logger.info(f"======== Summary =======")
    # logger.info(f"Total FLOPs {trainer.total_flops:4f}")

    trainer.deepspeed.empty_partition_cache()
    trainer.accelerator.free_memory()
    trainer.delete_accelerator()
    del trainer
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Training loss {training_loss[-1]} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    logger.info("Upstream Continual Learning done\n")
    
    # Run 4 Downstream Tasks
    
    # The finetuning hyperparameters are fixed for all downstream tasks
    # =============================================== DO NOT MODIFY THIS PART =========================================================#
    copied_args = copy.deepcopy(training_args)
    copied_args.per_gpu_train_batch_size = 2
    copied_args.gradient_accumulation_steps = 2
    copied_args.lr_scheduler_type = "constant"
    for downstream_idx in range(4):
        # You can customize how to initialize model weight for downstream tasks
        load_state_dict(model, state_dict, training_args)
        logger.info('model loading done')
        
        train_datalists = json.load(open(f'dataset/Downstream/train/dataset-{downstream_idx}.json', 'r'))
        random.shuffle(train_datalists)
    
        data_module = make_supervised_data_module(client_data=train_datalists, # sub_dataset
                                                tokenizer=tokenizer,
                                                data_args=copy.deepcopy(data_args))
        # ================================================================================================================================#
        
        
        # You are allowed to modify trainer only for optimizer initialization
        trainer = create_trainer_downstream(model,tokenizer,copied_args,data_module)
        
        results = trainer.train()
        
        state_dict = get_state_dict(training_args, model)
        os.makedirs(f"{training_args.state_dir}", exist_ok=True)
        output_dir = os.path.join(training_args.state_dir, f"Downstream_{downstream_idx}_sc{training_args.scenario}.pth")

        if (training_args.local_rank == 0 or training_args.local_rank == -1):
            torch.save(state_dict, output_dir)
        logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
        
        trainer.deepspeed.empty_partition_cache()
        trainer.accelerator.free_memory()
        trainer.delete_accelerator()
        del trainer
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    logger.info("Downstream adaptation done\n")


# =============================================== DO NOT MODIFY THIS PART =========================================================#
def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_state_dict(model, local_state_dict_list, training_args):
    model_to_load = local_state_dict_list
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(model_to_load, model, strict=False)
        else:
            model.load_state_dict(model_to_load, strict=False)  

# ===================================================================================================================================#

if __name__ == "__main__":
    main()

from utils.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, load_deepspeed
import random

def get_state_dict(training_args, model):
    # Modify this function to return the trained parameters of the model
    # This state_dict will be used to initialize model for downstream tasks and evaluation
    # Try to include only the trained parameters that is modified from the LLaVA weight, 
    # so that the saved model parameters size can be minimized.
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        state_dict.update(non_lora_state_dict)
    else:
        state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
    return state_dict


def configure_online_datastream(train_datalists, num_iterations, training_args, memory, memory_size, total_batchsize):
    ### Memory Only Training ###
    # To simulate online datastream, we construct the datastream by iteratively sampling from the memory before the training.
    # It is because it is hard to implement memory sampling during training for tranformers trainer.
    # We disabled random shuffling of dataloader in trainer, so the training happens in the order of the datalists.
    # You can modify the code to implement your own online datastream sampling strategy in TODO sections.
    
    datalists = []
    iteration=0
    #=======================================================================================================#
    #TODO
    #=======================================================================================================#
    for i, sample in enumerate(train_datalists):
        if len(memory) == memory_size:
            memory.pop(random.randrange(memory_size))
        memory.append(sample)
        iteration += num_iterations
        if iteration >= 1:
            for _ in range(int(iteration)):
                
                #=======================================================================================================#
                #TODO
                batch = random.sample(memory, k=min(len(memory), total_batchsize))
                
                #=======================================================================================================#
                mul = (total_batchsize//len(batch)) + 1
                batch = (batch*mul)[:total_batchsize]
                datalists.extend(batch[:])
                iteration -= 1
    return datalists
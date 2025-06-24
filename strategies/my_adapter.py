import torch

def load_adapter(model, model_args, training_args, tokenizer, compute_dtype):
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # Example of using customized adapter
        # ======================================================#
        # from models.duallora.dualloramodel import DualLoraModel
        # from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
        # PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
        # lora_config.peft_type = 'DUALLORA'
        # ======================================================#
        # Please refers to models/duallora/dualloralyer.py and models/duallora/dualloramodel.py to implement the customized adapter
        
        model = get_peft_model(model, lora_config)
    
    return model

def set_trainable_parameters(model, model_args, training_args, compute_dtype):
    # Set trainable parameters as you want.
    
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    model.lm_head.requires_grad_(False)
    
    

#=========================================================#
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
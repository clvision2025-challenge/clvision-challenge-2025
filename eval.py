import logging.config
import os
import random
import numpy as np
import torch
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import StoppingCriteria, StoppingCriteriaList

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, repeat_len = 2):
      self.n = repeat_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        should_stop =False
        if input_ids.shape[1] > self.n*3:
            last_n_ids = input_ids[0][-self.n:]		# 마지막으로 생성한 n개의 토큰
            lastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            lastlastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            for i in range(self.n):
                if lastlastlast_n_ids[i] != lastlast_n_ids[i] or lastlast_n_ids[i] != last_n_ids[i]: # stop sequence와 비교
                    should_stop = False
                    break
                else :
                    should_stop = True
        return should_stop

    
def evaluate(dataset, model, tokenizer, device, model_args, training_args, logger, output_name):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=2, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    
    if 'llava' in model_args.model_name_or_path.lower():
        conv = conversation_lib_llava.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):

            inputs, imgs, prompts, img_files = batch['input_ids'], batch['images'], batch['prompt'], batch['image_file']
            attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    images=imgs,
                    # image_sizes=image_sizes,
                    do_sample=True,# if args.temperature > 0 else False,
                    temperature=training_args.eval_temp,#args.temperature,
                    top_p=None,#args.top_p,
                    num_beams=1,#args.num_beams,
                    max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria = stopping_criteria
                )
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            # breakpoint()
            for pred_sentence, prompt, img_file in zip(pred_sentences, prompts, img_files):
                pred_sentence = pred_sentence.strip()
                predictions.append({"image_file":img_file, "input":prompt, "sentence":pred_sentence})
    
    with open(f"./eval_results/{training_args.note}/{output_name}.json", 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()

def main():   
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

    os.makedirs(f"eval_results/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.note}/eval.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    
    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    start_time = time.time()

    # Upstream scenario evaluation
    state_dict = torch.load(f'./checkpoints_{training_args.note}/Upstream_{training_args.scenario}.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    if training_args.scenario == 1:
        test_datalists = json.load(open(f'dataset/Upstream_scenario1/test/dataset-0.json', 'r'))
        dataset = GenerationDataset(test_datalists, tokenizer, data_args)
        evaluate(dataset, model, tokenizer, device, model_args, training_args, logger, 'Upstream_scenario1-0')
        logger.info(f'elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))}')
    elif training_args.scenario == 2:
        test_datalists = json.load(open(f'dataset/Upstream_scenario2/test/dataset-0.json', 'r'))
        dataset = GenerationDataset(test_datalists, tokenizer, data_args)
        evaluate(dataset, model, tokenizer, device, model_args, training_args, logger, 'Upstream_scenario2-0')
        logger.info(f'elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))}')
    # Downstream scenario evaluation
    for downstream_idx in range(4):
        state_dict = torch.load(f'./checkpoints_{training_args.note}/Downstream_{downstream_idx}_sc{training_args.scenario}.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        test_datalists = json.load(open(f'dataset/Downstream/test/dataset-{downstream_idx}.json', 'r'))
        dataset = GenerationDataset(test_datalists, tokenizer, data_args)
        evaluate(dataset, model, tokenizer, device, model_args, training_args, logger, f'Downstream_scenario{training_args.scenario}-{downstream_idx}')
        logger.info(f'elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))}')
    logger.info(f'elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))}')
if __name__ == "__main__":
    main()
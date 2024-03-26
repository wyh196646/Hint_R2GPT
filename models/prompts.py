import random
import pdb
from collections import defaultdict

from transformers import AutoTokenizer

import constants


def generate_chexpert_class_prompts(n = None):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in constants.CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def process_class_prompts(cls_prompts):
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77
    cls_prompt_inputs = defaultdict()
    for k,v in cls_prompts.items():
        text_inputs = tokenizer(v, truncation=True, padding=True, return_tensors='pt')
        cls_prompt_inputs[k] = text_inputs
    return cls_prompt_inputs


def list_of_lists_to_string_list(input_list):
    result = [' '.join(sublist) for sublist in input_list]
    return result

# from typing import List, Tuple, Optional
# cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
#                                       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#                                       'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
#                                       'Pneumothorax', 'Support Devices']

# # ---- TEMPLATES ----- # 
# # Define set of templates | see Figure 1 for more details                        
# cxr_pair_template: Tuple[str] = ("There is {}", "no {}")
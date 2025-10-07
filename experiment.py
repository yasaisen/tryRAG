"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507301621
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from datetime import datetime

from tryRAG.framework import RAGFramework
from tryRAG.evaluate import Evaluator, batchTestRunner

CUDA_LAUNCH_BLOCKING=1

SAVE_PATH = os.path.join(os.getcwd(), "..", "results")

def get_lmModel():
    return AutoModelForCausalLM.from_pretrained(
        "../../gemma-3-4b-it",
        device_map="auto",
    )

def save_list2json(
    meta_list, 
    save_filename, 
):
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(os.path.join(SAVE_PATH, f"{save_filename}.json"), "w") as file:
        json.dump(meta_list, file, indent=4, default=convert)

def run_test(
    exp_cfg: dict, 
    lm_model, 
    evaluator: Evaluator,
):
    print(f'\nstart run_test: {exp_cfg["exp_name"]}\n')
    rag = RAGFramework.from_config(
        cfg=exp_cfg, 
        lm_model=lm_model,
    )
    testrunner = batchTestRunner(
        rag=rag, 
        dataset_type=exp_cfg['dataset_type'], 
    )

    result_dict = testrunner.test(
        top_k=exp_cfg['top_k'], 
        use_upper_text=exp_cfg['use_upper_text'], 
        use_pre_answer=exp_cfg['use_pre_answer'], 
    )
    eval_res = evaluator.evaluate(
        result_dict=result_dict,
    )
    # save_list2json(
    #     meta_list=eval_res, 
    #     save_filename=exp_cfg['exp_name'], 
    # )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return exp_cfg.update(eval_res)

def run_testingsets(
    setting_list: list
):
    print('\nstart run_testingsets...\n')
    lm_model = get_lmModel()
    evaluator = Evaluator()

    eval_list = []
    for sample in setting_list:
        eval_list += [run_test(
            exp_cfg=sample,
            lm_model=lm_model,
            evaluator=evaluator,
        )]
    print('\n...end run_testingsets\n')

    nowtime = datetime.now().strftime("%y%m%d%H%M")

    df = pd.DataFrame(eval_list)
    df.to_csv(os.path.join(SAVE_PATH, f"{nowtime}_results.csv"), index=False)

if __name__ == "__main__":
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    setting_list = []
    setting_list += [
        # {
        #     'exp_name': 'CwMdT02DfUfPf', 
        #     'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
        #     'top_k': 1, 
        #     'use_upper_text': False, 
        #     'use_pre_answer': False, 
        #     "mode": "dense", #@ "dense" / "sparse" / "hybrid"
        #     "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
        #     "more_info": False, 
        # },
        {
            'exp_name': 'CwMsT02DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 1, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CwMhT02DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 1, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CwMdT02DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 1, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CwMsT02DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 1, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CwMhT02DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 1, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "web_page", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
    ]

    setting_list += [
        {
            'exp_name': 'CppMdT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
        {
            'exp_name': 'CppMsT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
        {
            'exp_name': 'CppMhT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
        {
            'exp_name': 'CppMdT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
        {
            'exp_name': 'CppMsT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
        {
            'exp_name': 'CppMhT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": True, 
        },
    ]

    setting_list += [
        {
            'exp_name': 'CpMdT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CpMsT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CpMhT05DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CpMdT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CpMsT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CpMhT05DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 5, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "paragraph", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
    ]

    setting_list += [
        {
            'exp_name': 'CsMdT15DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CsMsT15DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CsMhT15DfUfPf', 
            'dataset_type': 'factoidQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CsMdT15DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "dense", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CsMsT15DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "sparse", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
        {
            'exp_name': 'CsMhT15DoUfPf', 
            'dataset_type': 'optionalQA', #@ "factoidQA" / "optionalQA"
            'top_k': 15, 
            'use_upper_text': False, 
            'use_pre_answer': False, 
            "mode": "hybrid", #@ "dense" / "sparse" / "hybrid"
            "chunk_level": "sentence", #@ "web_page" / "paragraph" / "sentence"
            "more_info": False, 
        },
    ]











    run_testingsets(
        setting_list=setting_list,
    )
























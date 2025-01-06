import re
import sys
sys.path.append("..")

def process(responses:str, aspect):
    #把所有中文冒号替换为英文冒号
    responses = responses.replace("：", ":")

    responses = responses.split("####")[1:]
    assert len(responses) == 4
    annotation = []
    try:
        if aspect in ["instruction_following", "honesty"]:
            pattern = r"Rationale\**:(.+?)Rating\**:(.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL|re.IGNORECASE)

                annotation.append({
                    "Rating": re.findall(r'\d+', matches.group(2))[0] if "N/A" not in matches.group(2) else "N/A",
                    "Rationale": matches.group(1)
                })
        elif aspect in ["truthfulness", "helpfulness"]:
            pattern = r"Rationale for type\**:(.+?)Type\**:(.+?)Rationale for rating\**:(.+?)Rating\**:(.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL|re.IGNORECASE)
                annotation.append({
                    "Rating": re.findall(r'\d+', matches.group(4))[0] if "N/A" not in matches.group(4) else "N/A",
                    "Type": re.findall(r'\d+', matches.group(2)) if "None" not in matches.group(2) else ["None"],
                    "Rationale for type": matches.group(1),
                    "Rationale for rating": matches.group(3),

                })
    # except ValueError as e:  # TODO: bug process when the response does not follow the format
    #     print(responses)
    #     raise ValueError(e)
    # except AttributeError as e:
    #     print(responses)
    #     raise AttributeError(e)
    except Exception as e:
        print(responses)
        raise e
    return tuple(annotation)

from prompt_template_zh import system_prompt, instruction_following_template, honesty_template

from prompt_template_zh import truthfulness_template_with_answer,truthfulness_template_without_answer
from prompt_template_zh import helpfulness_template_without_answer as helpfulness_template
from 自己生成微调数据.api回复 import model_answer
from copy import deepcopy

SHUFLLE_NUM = 1


def annotate(example):
    #example=example.data
    aspects = ["instruction_following", "honesty", "truthfulness", "helpfulness"]
    completions = [dict({"annotations": {aspect: [] for aspect in aspects}}, **completion)
                    for completion in deepcopy(example["completions"])]

    subset=example["source"]

    model_names=example["models"]

    if subset=="TrufulQA_zh":
        TEMPLATE = {
            "instruction_following": instruction_following_template,
            "honesty": honesty_template,
            "truthfulness": truthfulness_template_with_answer,
            "helpfulness": helpfulness_template,
        }

        world_knowledge = "\n".join(["正确的答案可以是: " + str(example["correct_answers"]),
                                     "错误答案可能是: " + str(example["incorrect_answers"])])

    else:
        TEMPLATE = {
            "instruction_following": instruction_following_template,
            "honesty": honesty_template,
            "truthfulness": truthfulness_template_without_answer,
            "helpfulness": helpfulness_template,
        }
        world_knowledge=None

    # for i in range(4):
    #     example["completions"][i]["annotations"] = {}

    for aspect in aspects:

        # generate several lists of a random order of 4 completions, no repetition

        format_input = {"instruction": example["instruction"]}
        format_input.update({f"text_{i + 1}": example["completions"][i]["response"] for i in range(4)})
        if aspect == "truthfulness":
            format_input.update({"world_knowledge": world_knowledge})

        #responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
        full_instruction = TEMPLATE[aspect].format(**format_input)
        responses=model_answer(full_instruction, "deepseek-chat", max_tokens=1000, system=system_prompt, temperature=0,)

        # for i in range(5):
        #     try:
        #         # gpt-4 format error
        try:
            annotations = process(responses, aspect)
        except Exception as e:
            print(e)
            annotations=()
                #break

            # except Exception as e:
            #     if i < 10:
            #         responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
            #     else:
            #         raise e
            #         break
        for i in range(4):
            completions[i]["annotations"][aspect].append(annotations[i])
    example["completions"]=completions
    return example


def incorporate_annotation_to_completions(example):
    pass


if __name__ == "__main__":
    from datasets import Dataset,concatenate_datasets
    import pandas as pd

    for i in range(10):
        df_path="/data/yyj/ultrafeedback_zh/ultrafeedback_zh/completions_{}.jsonl".format(i)
        ds = Dataset.from_json(df_path)

        #将ds分为10个shard，依次处理
        num_shards=10
        ds_shards=[ds.shard(3000,i, contiguous=True) for i in range(10)]
        ds_shards_list=[]
        for ds_shard in ds_shards:
            ds_shard = ds_shard.map(annotate, num_proc=1)
            ds_shards_list.append(ds_shard)

        ds=concatenate_datasets(ds_shards_list)

        df=ds.to_pandas()
        df.to_json(df_path.replace("completions", "annotations"), lines=True, orient="records")


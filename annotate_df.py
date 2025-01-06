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
                    "Type": re.findall(r'\d+', matches.group(2)) if "None" not in matches.group(2) else "None",
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

SHUFLLE_NUM = 1


def annotate(example):
    #example=example.data
    aspects = ["instruction_following", "honesty", "truthfulness", "helpfulness"]

    subset=example["source"]

    model_names=example["models"]

    for i in range(4):
        example["completions"][i]["annotations"] = {}

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

    #example["annotations"] = {}
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
            annotations=("","","","")

        for j in range(4):
            example["completions"][j]["annotations"][aspect]=annotations[j]

    return example["completions"]


def incorporate_annotation_to_completions(example):
    pass

def process_df(df):
    # #新增一列annotations，初始化为空字典
    # df["annotations"] = [{} for _ in range(len(df))]


    for i in tqdm(range(len(df)),desc="处理数据",mininterval=30):
        example=df.loc[i].to_dict()
        annotations_with_completions = annotate(example)
        df.at[i, "completions"] = annotations_with_completions

    #df['annotations'] = df.apply(annotate, axis=1)
    return df


if __name__ == "__main__":
    from datasets import Dataset,concatenate_datasets
    import pandas as pd
    from multiprocessing import Pool
    from tqdm import tqdm

    for i in range(10):
        df_path="/data/yyj/ultrafeedback_zh/ultrafeedback_zh/completions_{}.jsonl".format(i)
        df=pd.read_json(df_path,lines=True)
        df.reset_index(drop=True,inplace=True)

        #将ds分为n个shard，依次处理
        num_workers=50
        df_list = []
        for i in range(num_workers):
            df_ = df[i::num_workers].copy()
            df_.reset_index(drop=True, inplace=True)
            df_list.append(df_)

        # 多进程
        with Pool(num_workers) as p:
            df_list = list(p.map(process_df, df_list))

        df = pd.concat(df_list, ignore_index=True)
        print("保存到：",df_path.replace("completions", "annotations"))
        df.to_json(df_path.replace("completions", "annotations"), lines=True, orient="records")


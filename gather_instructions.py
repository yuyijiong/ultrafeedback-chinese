import pandas as pd
import os
import re

#读取dir下所有parquet文件并拼接
def read_parquet_dir(dir:str)->pd.DataFrame:
    #获取dir下所有jsonl.gz文件
    files = [f for f in os.listdir(dir) if f.endswith('.parquet')]
    files.sort(reverse=True)
    print("文件数量：",len(files))

    df_list=[pd.read_parquet(os.path.join(dir, file)) for file in files]

    return pd.concat(df_list,ignore_index=True)

def conversation2instruction(conversation:list[dict]):
    if conversation[0]['from']=="human":
        return conversation[0]['value']
    elif conversation[1]['from']=="human" and conversation[0]['from']=="system":
        return conversation[0]['value']+"\n\n"+conversation[1]['value']
    else:
        raise ValueError("conversation format error")


def contains_invalid_characters(text):
    # 定义允许的字符范围
    pattern = re.compile(
        r'[^'
        r'\u4e00-\u9fff'  # 中文
        r'a-zA-Z'  # 英文字母
        r'!@#$%^&*()\[\],.?"\'/:{}|<>`·\-_'  # 英文标点符号
        # 中文标点符号
        r'！@#￥%……&*（），。—+【】、；：‘’“”|《》？、'
        r'\u2200-\u22FF'  # 数学符号
        r'\s'  # 空格和换行符
        r'\d'  # 数字
        r']'
    )

    # 搜索是否有不允许的字符
    if pattern.search(text):
        return True
    else:
        return False

if __name__ == '__main__':

    df_wizard = pd.read_json("./源数据/wizard_700k.jsonl", lines=True)
    df_wizard.drop(columns=["instruction"], inplace=True)
    df_wizard.rename(columns={"instruction_zh": "instruction"}, inplace=True)
    df_wizard['source'] = 'wizard_zh'

    df_ch_dpo = pd.read_parquet("./源数据/chinese-dpo-paris.parquet")
    df_ch_dpo.rename(columns={"prompt": "instruction"}, inplace=True)
    df_ch_dpo['source'] = 'chinese-dpo'

    # df_dpo_zh = pd.read_json("./源数据/dpo_zh.json")
    # df_dpo_zh["instruction"] = df_dpo_zh["conversations"].apply(conversation2instruction)
    # df_dpo_zh['source'] = 'dpo_zh'

    df_truth = pd.read_csv("./源数据/TrufulQA_zh.csv")
    df_truth.rename(columns={"Question": "instruction"}, inplace=True)
    df_truth.rename(columns={"Correct Answers": "correct_answers"}, inplace=True)
    df_truth.rename(columns={"Incorrect Answers": "incorrect_answers"}, inplace=True)
    df_truth['source'] = 'TrufulQA_zh'

    df_sharegpt = pd.read_json("./源数据/common_zh_70k.jsonl", lines=True)
    df_sharegpt["instruction"] = df_sharegpt["conversation"].apply(lambda x: x[0]['human'])
    df_sharegpt['source'] = 'sharegpt_zh'

    df_smoltalk = read_parquet_dir("../自己生成微调数据/chinese_magpie")
    df_smoltalk["instruction"] = df_smoltalk["conversations"].apply(lambda x: x[0]['content'])
    df_smoltalk['source'] = 'smoltalk_zh'
    #删除system_prompt_key为translate或rewriting或role-playing的样本
    df_smoltalk = df_smoltalk[~df_smoltalk["system_prompt_key"].apply(lambda x: x in ["translate", "role-playing","math23k"])]
    # 删除system_prompt_key为document-qa的且索引为奇数的样本
    df_smoltalk = df_smoltalk[~((df_smoltalk["system_prompt_key"] == "document-qa") & (df_smoltalk.index % 2 == 1))]
    # 删除score列为nan或score小于4的样本
    df_smoltalk = df_smoltalk[~df_smoltalk["score"].isna()]
    df_smoltalk = df_smoltalk[df_smoltalk["score"] >= 5]


    df_list=[df_wizard, df_ch_dpo, df_truth, df_sharegpt, df_smoltalk]
    df = pd.concat(df_list, ignore_index=True)

    #删除instruction长度小于10的样本（不会删除truthful_qa的样本）
    df=df[df.apply(lambda x:len(x["instruction"])>=20 or x["source"]=="TrufulQA_zh",axis=1)]

    #删除instruction中包含�的样本
    df=df[~df["instruction"].str.contains("\uFFFD")]
    #筛选出source为TrufulQA_zh的样本，删除instruction中包含不允许的字符的样本
    df_truth_cons=df[df["source"]=="TrufulQA_zh"]
    df_truth_cons=df_truth_cons[df_truth_cons["instruction"].apply(lambda x:contains_invalid_characters(x)==True)]
    #删除instruction中包含不允许的字符的样本
    df=df[df["instruction"].apply(lambda x:contains_invalid_characters(x)==False)]
    #删除instruction中中文占比小于0.5的样本
    df=df[df["instruction"].apply(lambda x:sum([1 for c in x if re.match(r'[\u4e00-\u9fa5]', c)])>len(x)*0.5)]

    #只保留 instruction \ source \ correct_answers \ incorrect_answers 列
    df=df[["instruction","source","correct_answers","incorrect_answers"]]

    #按instruction strip后去重
    df["instruction"]=df["instruction"].str.strip()
    df.drop_duplicates(subset=["instruction"],inplace=True)


    print(df["source"].value_counts())

    sample_rate={"smoltalk_zh":0.2,"sharegpt_zh":0.3,"TrufulQA_zh":1,"chinese-dpo":1,"wizard_zh":0.25}

    #按source采样
    df_sample=pd.concat([df[df["source"]==source].sample(frac=sample_rate[source],random_state=42) for source in sample_rate.keys()])

    print(df_sample["source"].value_counts())
    print("样本数量：",len(df_sample))

    df_sample.to_json("./ultrafeedback_zh/instructions.jsonl",orient='records',lines=True,force_ascii=False)





import pandas as pd
import swifter
import random

df_list=[]
for i in range(10):
    df_path = "/data/yyj/ultrafeedback_zh/ultrafeedback_zh/annotations_{}.jsonl".format(i)
    df_ = pd.read_json(df_path, lines=True)
    df_list.append(df_)

df=pd.concat(df_list,ignore_index=True)

def converse_type(x):
    #检查每个annotations的每个aspect
    for i in range(4):
        for aspect in ["instruction_following", "honesty", "truthfulness", "helpfulness"]:
            #如果有Type属性，且为str，转化为list
            if "Type" in x[i]["annotations"][aspect] and isinstance(x[i]["annotations"][aspect]["Type"],str):
                x[i]["annotations"][aspect]["Type"]=[x[i]["annotations"][aspect]["Type"]]

    return x


#检查completion的annotation的某个aspect是否为空字符串
def check_empty_annotations(x):
    aspects = ["instruction_following", "honesty", "truthfulness", "helpfulness"]
    for i in range(4):
        for aspect in aspects:
            if x[i]["annotations"][aspect]=="":
                return True
    return False

def check_empty_response(x):
    for i in range(4):
        if not x[i]["response"]:
            return True
    return False

def weighted_sum(x):
    aspects = ["instruction_following",  "truthfulness", "helpfulness","honesty",]
    #首先检查是否有分数为N/A
    has_none=False
    for i in range(4):
        for aspect in aspects:
            if "N/A" in x[i]["annotations"][aspect]["Rating"]:
                has_none=True
                print("N/A in ",aspect)
                if aspect!="honesty":
                    return None
                break

    #如果有N/A，则不统计honesty
    if has_none:
        aspects = ["instruction_following", "truthfulness", "helpfulness"]


    for i in range(4):
        overall_score=0
        for aspect in aspects:
            score = int(x[i]["annotations"][aspect]["Rating"])
            if aspect=="helpfulness":
                overall_score+=score*0.4
            else:
                overall_score+=score*0.2
        x[i]["overall_score"]=overall_score
    return x

#在completions中，选取overall_score最高的response作为正样本，其余随机选取一个作为负样本
def binarize(x):
    over_all_score_list=[x[i]["overall_score"] for i in range(4)]
    #如果得分完全一样，返回空
    if len(set(over_all_score_list))==1:
        return None,None,None,None


    best_score=max(over_all_score_list)
    best_index=over_all_score_list.index(best_score)
    negative_index=random.choice([i for i in range(4) if over_all_score_list[i]<best_score])


    best_response=x[best_index]["response"]
    negative_response=x[negative_index]["response"]

    best_info={"overall_score":best_score,
        "instruction_following":x[best_index]["annotations"]["instruction_following"]['Rating'],
                "truthfulness":x[best_index]["annotations"]["truthfulness"]['Rating'],
                "helpfulness":x[best_index]["annotations"]["helpfulness"]['Rating'],
                "honesty":x[best_index]["annotations"]["honesty"]['Rating']
               }
    negative_info={"overall_score":over_all_score_list[negative_index],
        "instruction_following":x[negative_index]["annotations"]["instruction_following"]['Rating'],
                "truthfulness":x[negative_index]["annotations"]["truthfulness"]['Rating'],
                "helpfulness":x[negative_index]["annotations"]["helpfulness"]['Rating'],
                "honesty":x[negative_index]["annotations"]["honesty"]['Rating']
               }

    return best_response.strip(),negative_response.strip(),best_info,negative_info

def binarize_chat_format(x):
    over_all_score_list=[x[i]["overall_score"] for i in range(4)]
    #如果得分完全一样，返回空
    if len(set(over_all_score_list))==1:
        return None,None,None,None


    best_score=max(over_all_score_list)
    best_index=over_all_score_list.index(best_score)
    negative_index=random.choice([i for i in range(4) if over_all_score_list[i]<best_score])

    best_response=[{"role":"system","content":x[best_index]["custom_system_prompt"]},
                   {"role":"user","content":x[best_index]["response"]}]

    negative_response=[{"role":"system","content":x[negative_index]["custom_system_prompt"]},
                     {"role":"user","content":x[negative_index]["response"]}]

    best_info={"overall_score":best_score,
        "instruction_following":x[best_index]["annotations"]["instruction_following"]['Rating'],
                "truthfulness":x[best_index]["annotations"]["truthfulness"]['Rating'],
                "helpfulness":x[best_index]["annotations"]["helpfulness"]['Rating'],
                "honesty":x[best_index]["annotations"]["honesty"]['Rating']
               }
    negative_info={"overall_score":over_all_score_list[negative_index],
        "instruction_following":x[negative_index]["annotations"]["instruction_following"]['Rating'],
                "truthfulness":x[negative_index]["annotations"]["truthfulness"]['Rating'],
                "helpfulness":x[negative_index]["annotations"]["helpfulness"]['Rating'],
                "honesty":x[negative_index]["annotations"]["honesty"]['Rating']
               }

    return best_response,negative_response,best_info,negative_info


#如果检查出空字符串，删除这一行
df["empty"]=df["completions"].apply(lambda x:check_empty_annotations(x) or check_empty_response(x))
df_f=df[~df["empty"]].copy()

#转化Type属性
df_f["completions"]=df_f["completions"].apply(converse_type)


#计算加权总分
df_f["completions"]=df_f["completions"].apply(weighted_sum)
#删除completions为空的行
df_f=df_f.dropna(subset=["completions"])

#保存
df_f.to_json("/data/yyj/ultrafeedback_zh/ultrafeedback_zh/annotations_weighted_sum.jsonl",lines=True,orient="records")
df_f.to_parquet("/data/yyj/ultrafeedback_zh/ultrafeedback_zh/annotations_weighted_sum.parquet")
#二值化
df_f["chosen_response"],df_f["rejected_response"],df_f["chosen_rating"],df_f["rejected_rating"]=zip(*df_f["completions"].swifter.apply(binarize))
df_b=df_f.dropna(subset=["chosen_response"])
#df_b只保留instruction,content,chosen_response,rejected_response,chosen_rating,rejected_rating
df_b=df_b[["instruction","source","chosen_response","rejected_response","chosen_rating","rejected_rating"]]


df_b.to_json("/data/yyj/ultrafeedback_zh/ultrafeedback_zh/annotations_binarized.jsonl",lines=True,orient="records")
df_b.to_parquet("/data/yyj/ultrafeedback_zh/ultrafeedback_zh/annotations_binarized.parquet")
#统计source列元素分布
print(df_b["source"].value_counts())


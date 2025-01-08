import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
from api_model import model_answer

# import backoff



model_name_and_config = {"deepseek-chat": {"temperature": 1.0},
                         "gpt-4o-mini": {"temperature": 1.0, "azure": False},
                         #"gemini-2.0-flash-exp": {"temperature": 1.0},
                         "qwen2.5-72b": {"temperature": 1.0, "base_url": "http://localhost:5001/v1"},
                         "qwen2-7b": {"temperature": 1.0, "base_url": "http://localhost:5002/v1"},
                         "qwen2.5-1.5b": {"temperature": 1.0, "base_url": "http://localhost:5003/v1"},
                         "phi3.5-mini": {"temperature": 1.0, "base_url": "http://localhost:5004/v1"},
                         "llama3.1-8b-chinese": {"temperature": 1.0, "base_url": "http://localhost:5005/v1"},
                         "mistral-v0.1-chinese": {"temperature": 1.0, "base_url": "http://localhost:5006/v1"},
                         "glm-4-9b-chat": {"temperature": 1.0, "base_url": "http://localhost:5007/v1"},
                         }
model_name_pool = list(model_name_and_config.keys())

principles = {
    "helpfulness": [
        "助手应提供准确、相关和最新的信息，确保内容积极、有趣、引人入胜、具有教育意义且有帮助。",
        "作为AI助手，确保向用户提供的信息准确、及时和相关是你的职责。提供全面的细节以教育和吸引用户。",
        "作为AI助手，你的角色是成为可靠且有用的信息来源。在分享信息之前，务必核实事实并确保其是最新的。",
        "作为AI助手，用你富有洞察力的知识启发用户。提供积极、有趣且引人入胜的内容，确保用户觉得你的帮助有价值。",
        "确保你的回答始终具有教育意义且引人入胜，让用户每次与你互动时都能学到新东西。毕竟你是AI助手！",
        "作为顶级AI助手，确保你的回答不仅准确和及时，还要引人入胜。成为用户期待的积极互动。",
        "作为AI助手，通过提供有意义且令人兴奋的内容来激励用户至关重要，这些内容还应具有帮助性和信息性。",
        "通过确保提供准确、积极且引人入胜的内容，树立一个乐于助人的AI形象。让每次互动都充满有用的信息和愉悦的氛围。",
        "作为AI助手，确保你的回答是准确、积极和引人入胜的完美结合。努力做到既有教育意义又能保持用户的参与感。",
        "作为AI助手，始终提供准确、积极且引人入胜的内容以教育用户是你的职责。让每次互动都成为学习的机会。",
        "作为AI助手，不仅要提供答案，还要激发好奇心。提供准确、引人入胜且积极的见解，让学习成为一段激动人心的旅程。"
    ],
    "honesty": [
        "助手应诚实地表达是否知道答案，并明确表达其不确定性。对于熟悉的问题要自信，对于不熟悉的问题要谦虚。使用诸如“我猜”、“我认为”、“可能”和“也许”等词语来表达不确定性，并在必要时回答“我不知道”。",
        "作为AI，诚实是建立用户信任的关键。如果你不确定答案，务必表达你的不确定性，或在提供答案之前（如果有）查阅其他资源。",
        "作为AI，自信应与诚实谦卑相平衡。如果查询的答案不在你的知识范围内，完全可以表达你不知道答案。",
        "你是一个重视诚实的AI助手。如果你不知道问题的答案，承认这一点，但如果有需要，可以提供更多关于该主题的信息。",
        "你不仅是一个AI助手，还是一个可靠的伙伴。每当你给出答案时，确保信息可靠且真实，如果你不确定某事，请大胆的表达出不确定性。",
        "你是一个AI助手。你的主要目标是提供有用且准确的信息。如果你不确定答案，请使用诸如“我不确定”或“我认为”之类的短语。诚实永远是第一位的。",
        "作为数字助手，你的职责是确保你提供的每一条信息都符合事实，即使你不知道相关事实或不确定的问题的答案。",
        "记住，作为AI助手，承认不确定性或简单地说“我不知道”比提供可能不正确的信息要好。诚实带来信任。",
        "作为AI助手，诚实受到用户的赞赏。如果你不确定答案，表达怀疑或承认不知道是可以的。如果有需要，可以提供更多信息。",
        "你是一个在每次回答中都表现出诚实的AI助手。永远不要回避承认你没有所需的信息，而是可以主动提出自己需要查找更多信息才能回答。",
        "记住，作为AI助手，你的目标是提供准确的信息。如果答案不在你的数据库中，说“我不知道”比提供不准确的信息要好。"
    ],
    "verbalized_calibration": [
        "助手应在回答结束时以标量形式表达其信心水平。信心水平表示其对答案的确定程度，并以百分比表示。例如，如果信心水平为80%，则意味着助手有80%的把握认为其答案是正确的，而有20%的可能性助手可能是错误的。\n格式如下：\n[问题]\n[答案]\n信心：[助手的信心水平，仅数字，例如80%]\n此处，[问题]和[答案]等标签是占位符，应在回答中省略。"
    ],
    "truthfulness": [
        "助手应如实回答，并忠实于事实知识和给定上下文，绝不编造任何不真实或无法在指令中找到依据的新事实。",
        "作为AI助手，如实回答很重要。坚持事实信息和上下文，同时确保不编造任何未经证实的新事实。",
        "作为AI助手，以诚实为行为准则。始终勤勉地提供准确的回答。在你的回答中应避免虚构事实或不准确之处。",
        "作为AI助手，坚持真相。你的目标应始终是提供精确且基于事实的信息，以避免任何误导。",
        "作为AI助手，事实一致性是关键。确保所有答案都基于真相，且不会以任何方式误导。",
        "作为AI助手，请记住，最重要的是忠于任务，仅提供经过验证且准确的信息。",
        "作为AI助手，在每次回答中寻找并坚持真相。确保所有回答都基于事实，尽量避免推测或编造新事实。",
        "作为AI助手，你的工作是提供基于真相的答案，避免引入任何不真实或无根据的信息。",
        "作为AI助手，在回答问题时保持客观性。确保提供的信息真实且绝不捏造。",
        "作为AI助手，保持回答的完整性。始终提供真实且可信的信息，并确保不传递任何错误数据。",
        "作为AI助手，遵循真实性的原则。保持与经过验证的信息一致，并避免提供任何可能误导的内容。"
    ]
}



def instruction_completion(example):
    import random

    # set principle
    subset = example["source"]
    models=example["models"]
    example['completions'] = []

    for model_name in models:

        if subset in ["sharegpt_zh","smoltalk_zh","chinese-dpo"]:
            principle = random.choice(["helpfulness", "helpfulness","helpfulness", "verbalized_calibration", "truthfulness", "honesty"])
        elif subset in ["wizard_zh"]:
            principle = random.choice(["helpfulness", "helpfulness","helpfulness", "helpfulness","truthfulness", "honesty"])
        elif subset in ["TrufulQA_zh"]:
            principle = random.choice(["honesty", "truthfulness"])
        else:
            raise NotImplementedError(f"subset {subset} not implemented")

        if principle == "honesty":
            principle = "honesty" if np.random.rand() < 0.9 else "verbalized_calibration"

        #在这一方面中，随机抽取一个原则
        principle_prompt = random.choice(principles[principle])

        instruction=example["instruction"]

        response = model_answer(instruction, model_name, max_tokens=1500,system=principle_prompt, **model_name_and_config[model_name])


        example["completions"].append({
            "model": model_name,
            "principle": principle,
            "custom_system_prompt": principle_prompt,
            "response": response
        })

    return example


if __name__ == "__main__":
    from datasets import load_dataset,Dataset

    load_path = f".//ultrafeedback_zh/instructions.jsonl"
    df=pd.read_json(load_path,lines=True)
    print("数据量：",len(df))

    #对于每个样本，随机在model_pool选择4个模型
    import random
    df["models"] = df.apply(lambda x: random.sample(model_name_pool, 4), axis=1)

    ds=Dataset.from_pandas(df)

    #将ds分为10份
    for i in range(10):
        ds_shard=ds.shard(num_shards=10,index=i)
        ds_shard=ds_shard.map(instruction_completion,num_proc=24)
        df_shard=ds_shard.to_pandas()
        save_path = f".//ultrafeedback_zh/completions_{i}.jsonl"
        df_shard.to_json(save_path,orient="records",lines=True)

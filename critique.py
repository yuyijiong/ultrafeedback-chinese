import re

system_prompt = "你是一个能够全面评估答案质量并提供建设性反馈的AI助手， 你会提供有帮助、具体且简洁明了的回答。"

feedback_prompt = \
"""我会提供一段对某个指令的回答，你的任务是给这段回答提供具体且建设性的反馈，帮助我学习和提升表现。你应该找到最适合我学习的方式，并通过反馈帮助我改进。

你需要从多个方面评估我的回答，包括帮助性、真实性、诚实性，以及回答在多大程度上遵循了指令。
---

### 指令
{instruction}

### 回答
{completion}
---

请扮演一位老师，提供具体且建设性的反馈。除了指出回答的弱点外，你还应提供具体的建议，指导我如何改进。请注意，你的建议应帮助我更好地完成指令，但不应引入指令中未提及的新要求。你的反馈应专注于提升我的批判性思维和准确回应的能力。然而，不要明确提供参考答案，也不需要礼貌用语。只需以简洁的风格回应。
最后，你需要为回答的整体质量打分，范围为1到10，其中1为最差，10为最好。

*格式*
### 反馈
[你的反馈]
总体评分: [1-10]

---

现在，请开始提供反馈。
"""

from 自己生成微调数据.api回复 import model_answer

SHUFLLE_NUM = 1


def annotate(example):
    example['critiques'] = []

    for i, completion in enumerate(example["completions"]):
        custom_system_prompt = completion["custom_system_prompt"]
        full_prompt=feedback_prompt.format(
            instruction="\n".join([example["instruction"], "Note: " + custom_system_prompt]),
            completion=completion["response"])

        response = model_answer(full_prompt, "deepseek-chat", max_tokens=1000, system=system_prompt, temperature=0,)

        response = response.split("Overall Score: ")
        assert len(response) == 2

        score=int(re.findall(r"\d+", response[1])[0])
        critique=response[0].strip()

        example['critiques'].append({
            "critique": critique,
            "score": score
        })

    return example


def incorporate_annotation_to_completions(example):
    pass


if __name__ == "__main__":
    from datasets import Dataset
    import pandas as pd

    for i in range(10):
        df_path="/data/yyj/ultrafeedback_zh/ultrafeedback_zh/completions_{}.jsonl".format(i)
        ds = Dataset.from_json(df_path)

        ds = ds.map(annotate, num_proc=32)

        df=ds.to_pandas()
        df.to_json(df_path.replace("completions", "critiques"), lines=True, orient="records")


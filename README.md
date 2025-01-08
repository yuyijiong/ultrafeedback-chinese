## UltraFeedback-chinese

``gather_instructions.py``: 从开源数据集中采样大量的 instruction，整合为一个文件

``completion.py``: 对于每条 instruction， 随机抽取4种不同的模型，调用 api 或 vllm，生成4种不同的 response

``annotate.py`` or ``annotate_df.py``: 使用 deepseek-v3 对4种不同的 response 进行 4个维度的评价，给出打分和理由

``gather_annotations.py``: 删除格式不对的数据，将数据集转化为 binarized 版本

``train-dpo.py``: 使用 ultrafeedback-chinese-binarized 对模型进行DPO训练

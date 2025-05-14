from funasr import AutoModel


# 1. 初始化：同时启用 VAD、Diarization、ASR
model = AutoModel(
    model=r"C:\Users\septemberlemon\.cache\modelscope\hub\models\iic\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model=r"C:\Users\septemberlemon\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model=r"C:\Users\septemberlemon\.cache\modelscope\hub\models\iic\punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model=r"C:\Users\septemberlemon\.cache\modelscope\hub\models\iic\speech_campplus_sv_zh-cn_16k-common",
    vad_kwargs={"max_single_segment_time": 30000},
    spk_kwargs={
        "threshold": 0.1,  # 阈值越低，簇数越多
        "min_duration_on": 0.2,  # 最短说话人活动段 0.2s
        "min_duration_off": 0.1  # 最短静音段 0.1s
    },
    merge_vad=False  # 不合并 VAD 产生的短片段
)


def write_res(res, output_path=f"output/output.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in res:
            merged = []
            last_spk = None
            buffer = []
            for sentence in seg["sentence_info"]:
                spk = sentence["spk"]
                text = sentence["text"]
                if spk == last_spk:
                    buffer.append(text)
                else:
                    if buffer:
                        merged_text = "".join(buffer)
                        merged.append((last_spk, merged_text))
                    buffer = [text]
                    last_spk = spk
            # 加上最后一段
            if buffer:
                merged.append((last_spk, "".join(buffer)))

            # 写入文件
            for spk, text in merged:
                f.write(f"Speaker {spk}: {text}\n")


# 2. 推理：直接拿到带说话人标注的 ASR 结果
res = model.generate(input="src/会议录音.m4a", batch_size_s=300)
# 3. 将推理的结果写入文本文件
write_res(res)

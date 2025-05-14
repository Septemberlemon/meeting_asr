from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


model = AutoModel(
    model=f"C:\\Users\septemberlemon\.cache\modelscope\hub\models\iic\SenseVoiceSmall",
    vad_model=f"C:\\Users\septemberlemon\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"E:\Audio\HuTao\\1.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

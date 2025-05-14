from pyannote.audio import Pipeline
from pathlib import Path


def load_local_pipeline():
    # 获取配置文件的绝对路径
    config_path = Path(__file__).parent / "models" / "pyannote_diarization_config.yaml"

    # 显式加载本地配置文件
    return Pipeline.from_pretrained(config_path.resolve())  # 关键修改：传递Path对象


pipeline = load_local_pipeline()
diarization = pipeline(f"./2.wav")

# 打印结果
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"说话人 {speaker} 在 {turn.start:.1f}s - {turn.end:.1f}s 发言")

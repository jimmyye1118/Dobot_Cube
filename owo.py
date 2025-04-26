#C:\Users\a2316\AppData\Local\Programs\Python\Python39\python.exe owo.py
#pip install -U openai-whisper

import whisper
import speech_recognition as sr


# 載入 Whisper 模型
model = whisper.load_model("medium")  # 使用 medium 或更高版本的模型

# 中文顏色詞庫
color_keywords = [
    "紅色", "藍色", "綠色", "黃色", "黑色", "白色",
    "橙色", "紫色", "灰色", "金色", "銀色"
]

# 初始化辨識器
recognizer = sr.Recognizer()
mic = sr.Microphone()

def record_and_transcribe():
    with mic as source:
        print("說話中")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # 根據背景噪音調整
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # 設定最大錄音時間

    # 儲存音檔
    with open("voice.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Whisper 轉文字
    result = model.transcribe("voice.wav", language="zh")
    text = result["text"]
    print("Whisper 辨識結果：", text)

    # 顏色分析
    found_colors = [color for color in color_keywords if color in text]
    if found_colors:
        print("偵測到顏色：", ', '.join(found_colors))
    else:
        print("沒有偵測到顏色")

if __name__ == "__main__":
    while True:
        cmd = input("\n按 Enter 開始錄音（或輸入 q 離開）:")
        if cmd.lower() == "q":
            break
        record_and_transcribe()



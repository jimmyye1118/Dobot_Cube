#C:\Users\a2316\AppData\Local\Programs\Python\Python39\python.exe GUI.py
#pip install -U openai-whisper

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import whisper
import speech_recognition as sr

# 載入 Whisper 模型
model = whisper.load_model("medium")
recognizer = sr.Recognizer()
mic = sr.Microphone()

# 中文顏色詞庫
color_keywords = ["紅色", "綠色", "黃色", "損壞", "異物"]

# 建立主視窗
root = tk.Tk()
root.title("AI方塊分揀系統")
root.geometry("1000x520")
root.configure(bg="#f5f5f5")

# 目前辨識結果
current_target = tk.StringVar(value="尚未辨識")

# ===== 左側面板 =====
left_frame = tk.Frame(root, width=350, bg="#ffffff", padx=20, pady=20)
left_frame.pack(side="left", fill="y")

tk.Label(left_frame, text="AI輸送帶方塊檢測系統", font=("Helvetica", 16, "bold"), bg="#ffffff").pack(pady=10)

# 顯示辨識結果
tk.Label(left_frame, textvariable=current_target, font=("Helvetica", 14), fg="#333", bg="#ffffff").pack(pady=15)

# Whisper 語音辨識按鈕
def whisper_voice_recognition():
    def recognize():
        current_target.set("辨識中，請稍候...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        with open("voice.wav", "wb") as f:
            f.write(audio.get_wav_data())

        result = model.transcribe("voice.wav", language="zh")
        text = result["text"]
        print("Whisper 辨識結果：", text)

        found = next((word for word in color_keywords if word in text), None)
        if found:
            current_target.set(f"目標：{found}方塊")
        else:
            current_target.set("未辨識到目標")

    threading.Thread(target=recognize).start()

ttk.Button(left_frame, text="🎙️ Whisper語音辨識", command=whisper_voice_recognition).pack(pady=10)

# 手動選擇下拉選單
def manual_select(event):
    selected = dropdown_var.get()
    current_target.set(f"目標：{selected}方塊")

tk.Label(left_frame, text="或手動選擇：", bg="#ffffff").pack(pady=(10, 5))
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(left_frame, textvariable=dropdown_var, values=color_keywords, state="readonly")
dropdown.pack()
dropdown.bind("<<ComboboxSelected>>", manual_select)

# 資料統計
tk.Label(left_frame, text="即時分類統計", font=("Helvetica", 12, "bold"), bg="#ffffff").pack(pady=10)

stat_labels = {
    "紅色": tk.StringVar(value="紅色：0"),
    "綠色": tk.StringVar(value="綠色：0"),
    "黃色": tk.StringVar(value="黃色：0"),
    "損壞": tk.StringVar(value="損壞：0"),
    "異物": tk.StringVar(value="異物：0"),
}
for key in stat_labels:
    tk.Label(left_frame, textvariable=stat_labels[key], font=("Helvetica", 11), bg="#ffffff", anchor="w").pack(fill="x", pady=2)

# 夾取次數
tk.Label(left_frame, text="已夾取數量：", font=("Helvetica", 12), bg="#ffffff").pack(pady=(20, 0))
grab_count = tk.StringVar(value="0")
tk.Label(left_frame, textvariable=grab_count, font=("Helvetica", 14), fg="#007acc", bg="#ffffff").pack()

# ===== 右側畫面 =====
right_frame = tk.Frame(root, width=640, height=480, bg="black")
right_frame.pack(side="right", padx=10, pady=10)
video_label = tk.Label(right_frame)
video_label.pack()

# 視訊更新
cap = cv2.VideoCapture(0)

def update_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, update_video)

update_video()

# 啟動主程式
root.mainloop()

# 關閉攝影機
cap.release()
cv2.destroyAllWindows()

'''

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
import random

class SortingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI方塊分類系統")
        self.root.geometry("1100x650")
        self.root.configure(bg="#f9f9f9")

        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.target_category = "紅色方塊"

        self.class_counts = {
            "紅色方塊": 0,
            "藍色方塊": 0,
            "黃色方塊": 0,
            "綠色方塊": 0,
            "瑕疵方塊": 0,
            "異物": 0,
        }

        self.build_gui()
        self.update_image()

    def build_gui(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TButton", font=("Helvetica", 10), padding=6)
        style.configure("TCombobox", padding=4)

        main_frame = tk.Frame(self.root, bg="#f9f9f9")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左側資訊欄
        left_frame = tk.Frame(main_frame, width=350, bg="#f9f9f9")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        # 視訊畫面右側
        right_frame = tk.Frame(main_frame, bg="#f9f9f9")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.video_label = tk.Label(right_frame, bg="black")
        self.video_label.pack()

        # 分類統計
        stats_frame = tk.LabelFrame(left_frame, text="📊 分類統計", font=("Helvetica", 12, "bold"), padx=10, pady=10, bg="white", fg="#333")
        stats_frame.pack(pady=10, fill=tk.X)

        self.labels = {}
        class_colors = {
            "紅色方塊": "#ff4d4d",
            "藍色方塊": "#4d79ff",
            "黃色方塊": "#ffd24d",
            "綠色方塊": "#4dff88",
            "瑕疵方塊": "#888888",
            "異物": "#000000"
        }

        for idx, (name, color) in enumerate(class_colors.items()):
            lbl = tk.Label(stats_frame, text=f"{name}：0", font=("Helvetica", 11), fg=color, bg="white", anchor="w")
            lbl.grid(row=idx, column=0, sticky="w", pady=2)
            self.labels[name] = lbl

        self.total_label = tk.Label(left_frame, text="📦 總處理數量：0", font=("Helvetica", 12, "bold"), bg="#f9f9f9", anchor="w")
        self.total_label.pack(pady=10, fill=tk.X)

        # 下拉選單
        tk.Label(left_frame, text="🎯 請選擇要夾的目標：", font=("Helvetica", 12), bg="#f9f9f9").pack(pady=(10, 5), anchor="w")
        self.target_var = tk.StringVar(value="紅色方塊")
        options = list(self.class_counts.keys())
        self.target_dropdown = ttk.Combobox(left_frame, textvariable=self.target_var, values=options, state="readonly", font=("Helvetica", 11))
        self.target_dropdown.pack(pady=5, fill=tk.X)
        self.target_dropdown.bind("<<ComboboxSelected>>", self.update_target)

        # 語音提示
        self.voice_label = tk.Label(left_frame, text="🔈 語音提示：--", font=("Helvetica", 11), bg="#f9f9f9", wraplength=320, justify="left")
        self.voice_label.pack(pady=15, anchor="w")

        # 控制按鈕
        btn_frame = tk.Frame(left_frame, bg="#f9f9f9")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="▶ 開始篩選", command=self.start_sorting, width=12, font=("Helvetica", 10), bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="⏸ 暫停篩選", command=self.stop_sorting, width=12, font=("Helvetica", 10), bg="#f39c12", fg="white").grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="🔄 重置統計", command=self.reset_stats, width=12, font=("Helvetica", 10), bg="#e74c3c", fg="white").grid(row=0, column=2, padx=5)

        self.status_bar = tk.Label(self.root, text="狀態：等待中", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 10), bg="#eeeeee")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_image)

    def start_sorting(self):
        if not self.running:
            self.running = True
            self.status_bar.config(text=f"狀態：篩選中（目標：{self.target_category}）")
            threading.Thread(target=self.fake_sorting_loop).start()

    def stop_sorting(self):
        self.running = False
        self.status_bar.config(text="狀態：已暫停")

    def reset_stats(self):
        for key in self.class_counts:
            self.class_counts[key] = 0
        self.update_labels()
        self.voice_label.config(text="🔈 語音提示：--")

    def update_target(self, event=None):
        self.target_category = self.target_var.get()
        self.status_bar.config(text=f"狀態：等待中（目標：{self.target_category}）")

    def fake_sorting_loop(self):
        categories = list(self.class_counts.keys())
        while self.running:
            time.sleep(1)
            detected = random.choice(categories)
            self.class_counts[detected] += 1
            self.update_labels()
            if detected == self.target_category:
                self.voice_label.config(text=f"🔈 目標【{detected}】出現，執行夾取！")
            else:
                self.voice_label.config(text=f"🔈 偵測到【{detected}】，非目標類別")

    def update_labels(self):
        total = sum(self.class_counts.values())
        for name, count in self.class_counts.items():
            self.labels[name].config(text=f"{name}：{count}")
        self.total_label.config(text=f"📦 總處理數量：{total}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SortingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

'''
import os.path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import functools
import argparse
import threading
import time
from pathlib import Path

from ppvector.predict import PPVectorPredictor
from ppvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class VoiceContrastGUI:
    def __init__(self, master):
        self.master = master
        master.title("夜雨飘零声纹对比系统")
        master.geometry('700x670')
        master.resizable(True, True)
        master.configure(bg='#f0f0f0')

        # 使用ttk样式
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 配置样式
        self.style.configure('TButton', font=('微软雅黑', 10), padding=5)
        self.style.configure('TLabel', font=('微软雅黑', 10), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('微软雅黑', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Result.TLabel', font=('微软雅黑', 16, 'bold'), foreground='#007bff', background='#f0f0f0')
        self.style.configure("Green.Horizontal.TProgressbar", background='#4CAF50', troughcolor='#f0f0f0',
                             borderwidth=0, thickness=20)

        # 创建主框架
        self.main_frame = ttk.Frame(master, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建标题
        self.title_label = ttk.Label(self.main_frame, text="声纹对比系统", style='Header.TLabel')
        self.title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))

        # 音频选择区域
        self.audio_frame = ttk.LabelFrame(self.main_frame, text="音频选择", padding="10 10 10 10")
        self.audio_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 20))

        # 音频1选择
        self.label1 = ttk.Label(self.audio_frame, text="音频文件1:")
        self.label1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.audio1_path = tk.StringVar()
        self.entry_audio1 = ttk.Entry(self.audio_frame, width=50, textvariable=self.audio1_path)
        self.entry_audio1.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.btn_audio1 = ttk.Button(self.audio_frame, text="选择文件", command=self.select_audio1)
        self.btn_audio1.grid(row=0, column=2, padx=10, pady=10)

        # 音频2选择
        self.label2 = ttk.Label(self.audio_frame, text="音频文件2:")
        self.label2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.audio2_path = tk.StringVar()
        self.entry_audio2 = ttk.Entry(self.audio_frame, width=50, textvariable=self.audio2_path)
        self.entry_audio2.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.btn_audio2 = ttk.Button(self.audio_frame, text="选择文件", command=self.select_audio2)
        self.btn_audio2.grid(row=1, column=2, padx=10, pady=10)

        # 设置列的权重
        self.audio_frame.columnconfigure(1, weight=1)

        # 参数设置区域
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="10 10 10 10")
        self.settings_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 20))

        # 判断阈值
        self.label3 = ttk.Label(self.settings_frame, text="对比阈值:")
        self.label3.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.threshold = tk.StringVar(value="0.6")
        self.entry_threshold = ttk.Entry(self.settings_frame, width=10, textvariable=self.threshold)
        self.entry_threshold.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.threshold_info = ttk.Label(self.settings_frame, text="(取值范围0-1，越大表示要求越严格)")
        self.threshold_info.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # 操作按钮区域
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 20))

        self.btn_predict = ttk.Button(self.button_frame, text="开始对比", command=self.predict_thread)
        self.btn_predict.pack(side=tk.LEFT, padx=10)

        self.btn_clear = ttk.Button(self.button_frame, text="清空", command=self.clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_quit = ttk.Button(self.button_frame, text="退出", command=self.quit)
        self.btn_quit.pack(side=tk.RIGHT, padx=10)

        # 状态区域
        self.status_frame = ttk.LabelFrame(self.main_frame, text="状态", padding="10 10 10 10")
        self.status_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(0, 10))

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, orient="horizontal",
                                            mode="determinate", variable=self.progress_var,
                                            style="Green.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # 结果显示区域
        self.result_frame = ttk.LabelFrame(self.main_frame, text="对比结果", padding="10 10 10 10")
        self.result_frame.grid(row=5, column=0, columnspan=4, sticky="ew")

        self.result_label = ttk.Label(self.result_frame, text="请选择两个音频文件进行对比",
                                      style='Result.TLabel', anchor=tk.CENTER)
        self.result_label.pack(fill=tk.X, pady=10)

        # 结果详情
        self.detail_frame = ttk.Frame(self.result_frame)
        self.detail_frame.pack(fill=tk.X, pady=5)

        self.similarity_label = ttk.Label(self.detail_frame, text="相似度: ")
        self.similarity_label.pack(side=tk.LEFT, padx=10)

        self.similarity_value = ttk.Label(self.detail_frame, text="--", font=('微软雅黑', 12, 'bold'))
        self.similarity_value.pack(side=tk.LEFT)

        # 设置列的权重
        for i in range(4):
            self.main_frame.columnconfigure(i, weight=1)

        # 预测器
        self.predictor = PPVectorPredictor(configs=args.configs, model_path=args.model_path, use_gpu=args.use_gpu)
        self.is_predicting = False

    def select_audio1(self):
        filename = filedialog.askopenfilename(initialdir='./dataset',
                                              filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                                                         ("所有文件", "*.*")])
        if filename:
            self.audio1_path.set(filename)

    def select_audio2(self):
        filename = filedialog.askopenfilename(initialdir='./dataset',
                                              filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                                                         ("所有文件", "*.*")])
        if filename:
            self.audio2_path.set(filename)

    def predict_thread(self):
        """在线程中执行预测"""
        if self.is_predicting:
            messagebox.showinfo("提示", "正在处理中，请稍候...")
            return

        audio_path1 = self.audio1_path.get()
        audio_path2 = self.audio2_path.get()

        if not audio_path1 or not audio_path2:
            messagebox.showerror("错误", "请选择两个音频文件")
            return

        try:
            threshold = float(self.threshold.get())
            if threshold < 0 or threshold > 1:
                messagebox.showerror("错误", "阈值必须在0-1之间")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效的阈值")
            return

        self.is_predicting = True
        self.btn_predict.config(state=tk.DISABLED)
        self.result_label.config(text="正在处理...")
        self.similarity_value.config(text="--")
        self.update_progress_bar(0)

        # 启动线程进行预测
        threading.Thread(target=self._predict, args=(audio_path1, audio_path2, threshold)).start()

    def _predict(self, audio_path1, audio_path2, threshold):
        """执行预测"""
        try:
            # 模拟进度
            for i in range(1, 101):
                if i < 90:  # 预留最后10%用于实际计算结果
                    self.update_progress_bar(i)
                    time.sleep(0.02)  # 调整速度使进度条看起来更自然

            # 执行实际预测
            dist = self.predictor.contrast(audio_path1, audio_path2)

            # 完成进度
            self.update_progress_bar(100)

            # 更新UI显示结果
            self.similarity_value.config(text=f"{dist:.5f}")

            if dist > threshold:
                result_text = f"两段语音来自同一个人"
                self.result_label.config(text=result_text, foreground="#4CAF50")
            else:
                result_text = f"两段语音来自不同的人"
                self.result_label.config(text=result_text, foreground="#F44336")

        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")
            self.result_label.config(text="预测失败，请检查音频文件格式", foreground="#F44336")

        finally:
            self.btn_predict.config(state=tk.NORMAL)
            self.is_predicting = False

    def clear(self):
        """清空所有输入和结果"""
        self.audio1_path.set("")
        self.audio2_path.set("")
        self.threshold.set("0.6")
        self.result_label.config(text="请选择两个音频文件进行对比", foreground="#007bff")
        self.similarity_value.config(text="--")
        self.update_progress_bar(0)

    def update_progress_bar(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        self.master.update_idletasks()

    def quit(self):
        self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VoiceContrastGUI(root)
    root.mainloop()

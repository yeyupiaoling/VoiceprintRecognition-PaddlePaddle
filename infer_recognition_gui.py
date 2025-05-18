import argparse
import functools
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import time
import queue

import numpy as np
import soundcard as sc

from ppvector.predict import PPVectorPredictor
from ppvector.utils.record import RecordAudio
from ppvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'audio_db/',                '音频库的路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class VoiceRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("夜雨飘零声纹识别系统")
        master.geometry('600x500')
        master.resizable(True, True)
        master.configure(bg='#f0f0f0')

        # 使用ttk样式
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 使用clam主题，也可以尝试'alt', 'default', 'classic'

        # 配置样式
        self.style.configure('TButton', font=('微软雅黑', 10), padding=5)
        self.style.configure('TLabel', font=('微软雅黑', 10), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('微软雅黑', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Result.TLabel', font=('微软雅黑', 16, 'bold'), foreground='#007bff', background='#f0f0f0')
        # 设置绿色进度条
        self.style.configure("Green.Horizontal.TProgressbar", background='#4CAF50', troughcolor='#f0f0f0',
                             borderwidth=0, thickness=20)

        # 识别使用时间，单位秒
        self.infer_time = 2
        # 录音采样率
        self.samplerate = 16000
        # 录音块大小
        self.numframes = 1024
        # 模型输入长度
        self.infer_len = int(self.samplerate * self.infer_time / self.numframes)
        self.recognizing = False
        self.record_data = []
        self.record_audio = RecordAudio()

        # 创建主框架
        self.main_frame = ttk.Frame(master, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建标题
        self.title_label = ttk.Label(self.main_frame, text="声纹识别系统", style='Header.TLabel')
        self.title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))

        # 参数设置区域
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="10 10 10 10")
        self.settings_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 20))

        # 录音长度标签和输入框
        self.record_seconds_label = ttk.Label(self.settings_frame, text="录音长度(秒):")
        self.record_seconds_label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.record_seconds = tk.StringVar(value='3')
        self.record_seconds_entry = ttk.Entry(self.settings_frame, width=15, textvariable=self.record_seconds)
        self.record_seconds_entry.grid(row=0, column=1, sticky="w", pady=5)

        # 判断是否为同一个人的阈值标签和输入框
        self.threshold_label = ttk.Label(self.settings_frame, text="识别阈值:")
        self.threshold_label.grid(row=0, column=2, sticky="w", padx=(20, 10), pady=5)
        self.threshold = tk.StringVar(value='0.6')
        self.threshold_entry = ttk.Entry(self.settings_frame, width=15, textvariable=self.threshold)
        self.threshold_entry.grid(row=0, column=3, sticky="w", pady=5)

        # 功能按钮区域
        self.buttons_frame = ttk.LabelFrame(self.main_frame, text="功能选择", padding="10 10 10 10")
        self.buttons_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 20))

        # 创建按钮
        self.register_button = ttk.Button(self.buttons_frame, text="注册声纹", command=self.register)
        self.register_button.grid(row=0, column=0, padx=10, pady=10)

        self.recognize_button = ttk.Button(self.buttons_frame, text="识别声纹", command=self.recognize)
        self.recognize_button.grid(row=0, column=1, padx=10, pady=10)

        self.remove_user_button = ttk.Button(self.buttons_frame, text="删除用户", command=self.remove_user)
        self.remove_user_button.grid(row=0, column=2, padx=10, pady=10)

        self.recognize_real_button = ttk.Button(self.buttons_frame, text="实时识别", command=self.recognize_thread)
        self.recognize_real_button.grid(row=0, column=3, padx=10, pady=10)

        # 状态区域
        self.status_frame = ttk.LabelFrame(self.main_frame, text="状态", padding="10 10 10 10")
        self.status_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 10))

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, orient="horizontal", length=100,
                                            mode="determinate", variable=self.progress_var,
                                            style="Green.Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.status_frame.columnconfigure(0, weight=1)

        # 结果显示
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.grid(row=4, column=0, columnspan=4, sticky="ew")

        self.result_label = ttk.Label(self.result_frame, text="欢迎使用声纹识别系统", style='Result.TLabel',
                                      anchor=tk.CENTER)
        self.result_label.pack(fill=tk.X, pady=10)

        # 音量显示区域
        self.volume_canvas = tk.Canvas(self.main_frame, height=50, bg='#f0f0f0', highlightthickness=1,
                                       highlightbackground='#d0d0d0')
        self.volume_canvas.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(0, 10))

        # 设置列和行的权重，使界面能够随窗口大小调整
        for i in range(4):
            self.main_frame.columnconfigure(i, weight=1)

        # 识别器
        self.predictor = PPVectorPredictor(configs=args.configs,
                                           threshold=float(self.threshold.get()),
                                           audio_db_path=args.audio_db_path,
                                           model_path=args.model_path,
                                           use_gpu=args.use_gpu)

    # 注册
    def register(self):
        try:
            record_seconds = int(self.record_seconds.get())
            if record_seconds <= 0:
                messagebox.showerror("错误", "录音时长必须大于0秒")
                return

            # 开始录音
            self.result_label.config(text="正在录音...")
            self.update_progress_bar(0)

            # 创建录音队列和停止事件
            self.audio_queue = queue.Queue()
            self.stop_recording = threading.Event()

            # 启动线程进行录音和进度更新
            self.recording_thread = threading.Thread(target=self._record_audio, args=(record_seconds,))
            self.progress_thread = threading.Thread(target=self._update_recording_progress, args=(record_seconds,))

            self.recording_thread.start()
            self.progress_thread.start()

        except ValueError:
            messagebox.showerror("错误", "请输入有效的录音时长")

    def _record_audio(self, record_seconds):
        """在后台线程中进行录音"""
        try:
            # 使用自定义录音方法
            audio_data = self._record_realtime(record_seconds)

            # 将录音数据放入队列
            self.audio_queue.put((True, audio_data))
        except Exception as e:
            # 发生错误时，通知主线程
            self.audio_queue.put((False, str(e)))
            self.stop_recording.set()

    def _record_realtime(self, record_seconds):
        """实时录制音频数据"""
        chunks = []
        default_mic = sc.default_microphone()
        with default_mic.recorder(samplerate=self.samplerate, channels=1) as mic:
            # 录制指定秒数的音频
            start_time = time.time()
            while time.time() - start_time < record_seconds and not self.stop_recording.is_set():
                data = mic.record(numframes=self.numframes)
                chunks.append(data)

        # 合并所有音频数据
        if chunks:
            return np.concatenate(chunks)
        return np.array([])

    def _update_recording_progress(self, record_seconds):
        """更新录音进度条"""
        start_time = time.time()
        try:
            while time.time() - start_time < record_seconds and not self.stop_recording.is_set():
                elapsed = time.time() - start_time
                progress = min(100, (elapsed / record_seconds) * 100)
                self.update_progress_bar(progress)
                self.master.update_idletasks()
                time.sleep(0.1)

            # 确保进度条显示100%
            self.update_progress_bar(100)

            # 等待录音线程完成
            if self.recording_thread.is_alive():
                self.recording_thread.join()

            # 从队列获取录音结果
            success, result = self.audio_queue.get(timeout=5)

            if success:
                self.result_label.config(text="录音完成")
                # 请求用户输入名称
                self.master.after(100, self._ask_for_name, result)
            else:
                messagebox.showerror("错误", f"录音失败: {result}")
                self.result_label.config(text="录音失败")

        except Exception as e:
            messagebox.showerror("错误", f"录音过程出错: {str(e)}")
            self.result_label.config(text="录音失败")
            self.stop_recording.set()

    def _ask_for_name(self, audio_data):
        name = simpledialog.askstring(title="注册声纹", prompt="请输入注册人姓名")
        if name is not None and name.strip() != '':
            try:
                self.predictor.register(user_name=name, audio_data=audio_data,
                                        sample_rate=self.record_audio.sample_rate)
                messagebox.showinfo("成功", f"用户 {name} 注册成功")
                self.result_label.config(text=f"用户 {name} 注册成功")
            except Exception as e:
                messagebox.showerror("错误", f"注册失败: {str(e)}")
                self.result_label.config(text="注册失败")
        else:
            self.result_label.config(text="注册取消")

    # 识别
    def recognize(self):
        try:
            threshold = float(self.threshold.get())
            record_seconds = int(self.record_seconds.get())

            if record_seconds <= 0:
                messagebox.showerror("错误", "录音时长必须大于0秒")
                return

            if threshold < 0 or threshold > 1:
                messagebox.showerror("错误", "阈值必须在0-1之间")
                return

            # 开始录音
            self.result_label.config(text="正在录音...")
            self.update_progress_bar(0)

            # 创建录音队列和停止事件
            self.audio_queue = queue.Queue()
            self.stop_recording = threading.Event()

            # 启动线程进行录音和进度更新
            self.recording_thread = threading.Thread(target=self._record_audio, args=(record_seconds,))
            self.progress_thread = threading.Thread(target=self._update_recognition_progress,
                                                    args=(record_seconds, threshold))

            self.recording_thread.start()
            self.progress_thread.start()

        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数")

    def _update_recognition_progress(self, record_seconds, threshold):
        """更新识别进度条并处理识别结果"""
        start_time = time.time()
        try:
            while time.time() - start_time < record_seconds and not self.stop_recording.is_set():
                elapsed = time.time() - start_time
                progress = min(100, (elapsed / record_seconds) * 100)
                self.update_progress_bar(progress)
                self.master.update_idletasks()
                time.sleep(0.1)

            # 确保进度条显示100%
            self.update_progress_bar(100)

            # 等待录音线程完成
            if self.recording_thread.is_alive():
                self.recording_thread.join()

            # 从队列获取录音结果
            success, result = self.audio_queue.get(timeout=5)

            if success:
                self.result_label.config(text="正在识别...")

                # 进行识别
                name, score = self.predictor.recognition(result, threshold, sample_rate=self.record_audio.sample_rate)

                # 显示结果
                if name:
                    self.result_label.config(text=f"识别结果: {name}，匹配度: {score:.2f}")
                else:
                    self.result_label.config(text="未能识别，可能是未注册用户")
            else:
                messagebox.showerror("错误", f"录音失败: {result}")
                self.result_label.config(text="录音失败")

        except Exception as e:
            messagebox.showerror("错误", f"识别过程出错: {str(e)}")
            self.result_label.config(text="识别失败")
            self.stop_recording.set()

    def remove_user(self):
        name = simpledialog.askstring(title="删除用户", prompt="请输入要删除的用户名")
        if name is not None and name.strip() != '':
            result = self.predictor.remove_user(user_name=name)
            if result:
                messagebox.showinfo("成功", f"用户 {name} 已删除")
                self.result_label.config(text=f"用户 {name} 已删除")
            else:
                messagebox.showerror("错误", f"用户 {name} 不存在或删除失败")
                self.result_label.config(text="删除失败")

    def recognize_thread(self):
        if not self.recognizing:
            self.recognizing = True
            self.recognize_real_button.config(text="停止识别")
            self.result_label.config(text="实时识别中...")
            threading.Thread(target=self.recognize_real).start()
            threading.Thread(target=self.record_real).start()
            threading.Thread(target=self.update_volume_display).start()
        else:
            self.recognizing = False
            self.recognize_real_button.config(text="实时识别")
            self.result_label.config(text="实时识别已停止")

    # 实时识别
    def recognize_real(self):
        try:
            threshold = float(self.threshold.get())
            while self.recognizing:
                if len(self.record_data) < self.infer_len:
                    time.sleep(0.1)
                    continue
                # 截取最新的音频数据
                seg_data = self.record_data[-self.infer_len:]
                audio_data = np.concatenate(seg_data)
                # 删除旧的音频数据
                del self.record_data[:len(self.record_data) - self.infer_len]
                name, score = self.predictor.recognition(audio_data, threshold,
                                                         sample_rate=self.record_audio.sample_rate)
                if name:
                    self.result_label.config(text=f"识别到: {name} (匹配度: {score:.2f})")
                else:
                    self.result_label.config(text="听音中...")
        except Exception as e:
            self.recognizing = False
            self.recognize_real_button.config(text="实时识别")
            messagebox.showerror("错误", f"实时识别错误: {str(e)}")

    def record_real(self):
        self.record_data = []
        try:
            default_mic = sc.default_microphone()
            with default_mic.recorder(samplerate=self.samplerate, channels=1) as mic:
                while self.recognizing:
                    data = mic.record(numframes=self.numframes)
                    self.record_data.append(data)
        except Exception as e:
            self.recognizing = False
            self.recognize_real_button.config(text="实时识别")
            messagebox.showerror("错误", f"录音错误: {str(e)}")

    def update_volume_display(self):
        """更新音量显示"""
        try:
            while self.recognizing:
                if self.record_data:
                    # 获取最新的音频数据计算音量
                    latest_data = self.record_data[-1] if self.record_data else np.zeros((self.numframes, 1))
                    # 将放大系数从500增加到3000，使音量显示更加明显
                    volume = np.abs(latest_data).mean() * 3000

                    # 绘制音量条
                    self.volume_canvas.delete("all")
                    canvas_width = self.volume_canvas.winfo_width()
                    canvas_height = self.volume_canvas.winfo_height()

                    # 确保有最小显示值
                    min_bar_width = 20
                    # 计算音量条的宽度，最大为画布宽度的95%
                    bar_width = max(min_bar_width, min(volume, canvas_width * 0.95))

                    # 根据音量值决定颜色
                    if volume < canvas_width * 0.3:
                        color = "#4CAF50"  # 绿色
                    elif volume < canvas_width * 0.6:
                        color = "#FFC107"  # 黄色
                    else:
                        color = "#F44336"  # 红色

                    # 绘制音量条
                    self.volume_canvas.create_rectangle(
                        10, 10,
                        10 + bar_width,
                        canvas_height - 10,
                        fill=color, outline=""
                    )

                    # 绘制文本
                    self.volume_canvas.create_text(
                        canvas_width - 50,
                        canvas_height // 2,
                        text=f"音量: {int(volume)}",
                        fill="#333333",
                        font=('微软雅黑', 9)
                    )

                time.sleep(0.1)
        except Exception as e:
            print(f"音量显示错误: {str(e)}")

    def update_progress_bar(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        self.master.update_idletasks()


if __name__ == '__main__':
    root = tk.Tk()
    gui = VoiceRecognitionGUI(root)
    root.mainloop()

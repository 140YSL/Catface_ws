#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频时间裁剪 + 空间裁剪。依赖：opencv-python Pillow imageio-ffmpeg。
运行：python video_extra_sort.py [视频目录]
默认读取 ../../cat_video/raw，分类保存至 ../../cat_video/organized。
FFmpeg 优先取 PATH 中的版本，否则使用 imageio-ffmpeg 自带版本。
"""
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.mpg', '.mpeg', '.ts', '.wmv'}


VIDEO_ROOT = (Path(__file__).resolve().parent / '../../cat_video').resolve()
DEFAULT_FOLDER = VIDEO_ROOT / 'raw'
ORGANIZED_FOLDER = VIDEO_ROOT / 'organized'
CATEGORIES = (
    ('放松/满足/愉悦', 'enjoy'),
    ('好奇', 'curious'),
    ('兴奋', 'excited'),
    ('害怕/紧张', 'afraid'),
    ('生气', 'angry'),
    ('委屈/示弱', 'aggrieved'),
    ('中性脸/无视/冷漠', 'ignore'),
    ('撒娇/乞求', 'coquettishly'),
    ('困乏', 'drowsy'),
)


def next_output_path(folder):
    """从 1 开始查找空闲编号，扩展名比较忽略大小写。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    occupied = {path.name.casefold() for path in folder.iterdir()}
    number = 1
    while f'{number}.mp4' in occupied:
        number += 1
    return folder / f'{number}.mp4'


def timestamp(seconds):
    ms = max(0, round(seconds * 1000))
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{ms:03}'


def find_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError('找不到 FFmpeg。请运行 pip install imageio-ffmpeg，或把 ffmpeg 加入 PATH。') from exc


class Timeline(tk.Canvas):
    """橙色 S/E 手柄定义导出区间，蓝色轨道定义播放位置。"""
    def __init__(self, master, seek, changed):
        super().__init__(master, height=108, bg='#f4f5f7', highlightthickness=0)
        self.seek_callback = seek
        self.change_callback = changed
        self.duration = 1.0
        self.start = 0.0
        self.end = 1.0
        self.position = 0.0
        self.drag = None
        self.enabled = False
        self.bind('<Configure>', lambda event: self.draw())
        self.bind('<Button-1>', self.press)
        self.bind('<B1-Motion>', self.motion)
        self.bind('<ButtonRelease-1>', self.release)

    def x(self, seconds):
        return 28 + seconds / max(self.duration, 0.001) * max(self.winfo_width() - 56, 1)

    def seconds(self, x):
        return min(self.duration, max(0, (x - 28) / max(self.winfo_width() - 56, 1) * self.duration))

    def reset(self, duration):
        self.duration = duration
        self.start, self.end, self.position = 0.0, duration, 0.0
        self.enabled = True
        self.draw()

    def draw(self):
        self.delete('all')
        left, right = 28, max(29, self.winfo_width() - 28)
        self.create_line(left, 46, right, 46, fill='#c7cbd1', width=10)
        self.create_line(self.x(self.start), 46, self.x(self.end), 46, fill='#ffd59d', width=10)
        for label, value in [('S', self.start), ('E', self.end)]:
            x = self.x(value)
            self.create_line(x, 27, x, 55, fill='#c56b00', width=2)
            self.create_rectangle(x - 11, 5, x + 11, 27, fill='#f59e0b', outline='')
            self.create_text(x, 16, text=label, fill='white', font=('', 10, 'bold'))
        self.create_line(left, 76, right, 76, fill='#a6b8d1', width=4)
        p = self.x(self.position)
        self.create_oval(p - 7, 69, p + 7, 83, fill='#2563eb', outline='white')
        self.create_text(left, 99, text='00:00:00.000', anchor='w', fill='#555')
        self.create_text(right, 99, text=timestamp(self.duration), anchor='e', fill='#555')

    def press(self, event):
        if not self.enabled:
            return
        if event.y <= 57:
            candidates = [(abs(event.x - self.x(self.start)), 'S'), (abs(event.x - self.x(self.end)), 'E')]
            distance, handle = min(candidates)
            self.drag = handle if distance <= 20 else 'P'
        else:
            self.drag = 'P'
        self.motion(event)

    def motion(self, event):
        if self.drag is None:
            return
        value = self.seconds(event.x)
        gap = min(0.01, self.duration / 2)
        if self.drag == 'S':
            self.start = min(value, self.end - gap)
            self.change_callback()
        elif self.drag == 'E':
            self.end = max(value, self.start + gap)
            self.change_callback()
        else:
            self.position = value
            self.seek_callback(value)
        self.draw()

    def release(self, event):
        self.motion(event)
        self.drag = None


class VideoCropper:
    def __init__(self, root, initial_folder=None):
        self.root = root
        self.root.title('视频框选、时间裁剪与分类保存')
        self.root.geometry('1280x850')
        self.root.minsize(900, 650)
        self.cap = None
        self.video = None
        self.files = []
        self.fps = 25.0
        self.frames = 0
        self.duration = 0.0
        self.width = self.height = 0
        self.current_frame = 0
        self.frame_bgr = None
        self.photo = None
        self.display_box = None
        self.crop = None
        self.first_point = None
        self.playing = False
        self.seek_target = None
        self.seek_job = None
        self.redraw_job = None
        self.events = queue.Queue()
        self.exporting = False
        self.category_dialog = None
        self.cancel_event = threading.Event()
        self.folder_var = tk.StringVar(value='请选择视频文件夹')
        self.info_var = tk.StringVar(value='打开文件夹，然后选择左侧视频。')
        self.range_var = tk.StringVar()
        self.crop_var = tk.StringVar(value='裁剪范围：整幅画面')
        self.status_var = tk.StringVar(value='就绪')
        self.build_ui()
        self.bind_shortcuts()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll_events)
        self.load_folder(initial_folder if initial_folder is not None else DEFAULT_FOLDER)

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        ttk.Button(top, text='打开视频文件夹', command=self.choose_folder).pack(side='left')
        ttk.Label(top, textvariable=self.folder_var).pack(side='left', padx=12)
        body = ttk.Panedwindow(self.root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=8)
        sidebar = ttk.Frame(body, width=235)
        body.add(sidebar, weight=0)
        ttk.Label(sidebar, text='视频文件（单击切换）').pack(anchor='w', pady=4)
        self.listbox = tk.Listbox(sidebar, width=30, exportselection=False)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.listbox.pack(fill='both', expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.select_video)
        panel = ttk.Frame(body)
        body.add(panel, weight=1)
        self.canvas = tk.Canvas(panel, bg='#141414', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.resize_preview)
        self.canvas.bind('<Button-1>', self.crop_click)
        ttk.Label(panel, text='框选：在画面中依次单击矩形的两个对角点；再次单击可重新框选。').pack(anchor='w', pady=(5, 0))
        ttk.Label(panel, text='快捷键：空格 播放/暂停 | ←/→ 逐帧 | Ctrl+←/→ 跳转5秒 | Ctrl+S 分类保存 | Alt+S/E 起止点').pack(anchor='w')
        ttk.Label(panel, textvariable=self.info_var).pack(anchor='w')
        self.timeline = Timeline(panel, self.request_seek, self.update_range)
        self.timeline.pack(fill='x')
        controls = ttk.Frame(panel)
        controls.pack(fill='x', pady=4)
        self.play_button = ttk.Button(controls, text='▶ 播放', command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Button(controls, text='当前设为 S', command=lambda: self.set_marker('S')).pack(side='left', padx=4)
        ttk.Button(controls, text='当前设为 E', command=lambda: self.set_marker('E')).pack(side='left')
        ttk.Button(controls, text='重置框选', command=self.reset_crop).pack(side='left', padx=4)
        ttk.Label(panel, textvariable=self.range_var).pack(anchor='w')
        ttk.Label(panel, textvariable=self.crop_var).pack(anchor='w', pady=3)
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill='x')
        self.export_button = ttk.Button(bottom, text='分类保存选定视频…', command=self.export)
        self.export_button.pack(side='right')
        self.cancel_button = ttk.Button(bottom, text='取消导出', command=self.cancel_event.set, state='disabled')
        self.cancel_button.pack(side='right', padx=8)
        self.progress = ttk.Progressbar(bottom, length=180)
        self.progress.pack(side='left')
        ttk.Label(bottom, textvariable=self.status_var).pack(side='left', padx=10)

    def bind_shortcuts(self):
        # 绑定到窗口：焦点位于列表、按钮或预览时均可使用。
        # 返回 break，避免方向键同时切换文件、空格同时触发按钮。
        shortcuts = {
            '<space>': self.toggle_play,
            '<Left>': lambda: self.keyboard_seek(-1, by_frame=True),
            '<Right>': lambda: self.keyboard_seek(1, by_frame=True),
            '<Control-Left>': lambda: self.keyboard_seek(-5),
            '<Control-Right>': lambda: self.keyboard_seek(5),
            '<Control-s>': self.export,
            '<Control-S>': self.export,
            '<Alt-s>': lambda: self.keyboard_marker('S'),
            '<Alt-S>': lambda: self.keyboard_marker('S'),
            '<Alt-e>': lambda: self.keyboard_marker('E'),
            '<Alt-E>': lambda: self.keyboard_marker('E'),
        }
        for sequence, action in shortcuts.items():
            def handler(event, callback=action):
                callback()
                return 'break'
            self.root.bind(sequence, handler)
        # 让窗口快捷键优先于控件默认键盘行为。
        def prioritize(widget):
            tags = list(widget.bindtags())
            root_tag = str(self.root)
            if root_tag in tags and str(widget) != root_tag:
                tags.remove(root_tag)
                tags.insert(0, root_tag)
                widget.bindtags(tuple(tags))
            for child in widget.winfo_children():
                prioritize(child)
        prioritize(self.root)

    def keyboard_seek(self, amount, by_frame=False):
        if not self.cap:
            return
        # 连续按键累计到尚未解码的目标，避免快速按键被吞掉。
        base = self.seek_target if self.seek_target is not None else self.timeline.position
        if by_frame:
            index = round(base * self.fps) + amount
        else:
            index = round((base + amount) * self.fps)
        index = min(self.frames - 1, max(0, index))
        target = index / self.fps
        self.timeline.position = target
        self.timeline.draw()
        self.request_seek(target)

    def keyboard_marker(self, marker):
        if self.seek_job:
            self.root.after_cancel(self.seek_job)
            self.flush_seek()
        self.set_marker(marker)

    def choose_folder(self):
        folder = filedialog.askdirectory(title='选择包含视频的文件夹',
                                         initialdir=str(DEFAULT_FOLDER))
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        try:
            files = sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.casefold())
        except OSError as exc:
            messagebox.showerror('无法读取文件夹', str(exc))
            return
        self.folder_var.set(str(folder))
        self.files = files
        self.listbox.delete(0, 'end')
        for file in files:
            self.listbox.insert('end', file.name)
        if files:
            self.listbox.selection_set(0)
            self.open_video(files[0])
        else:
            self.status_var.set('该文件夹没有支持的视频文件（不递归子文件夹）。')

    def select_video(self, event=None):
        selection = self.listbox.curselection()
        if selection:
            path = self.files[selection[0]]
            if path != self.video:
                self.open_video(path)

    def open_video(self, path):
        self.pause()
        if self.seek_job:
            self.root.after_cancel(self.seek_job)
            self.seek_job = None
        self.seek_target = None
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not ok or not math.isfinite(count) or count < 1:
            cap.release()
            messagebox.showerror('无法打开视频', f'无法读取视频或时长：\n{path}')
            return
        if self.cap:
            self.cap.release()
        self.cap, self.video = cap, path
        self.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        self.frames = int(count)
        self.duration = self.frames / self.fps
        self.height, self.width = frame.shape[:2]
        self.current_frame = 0
        self.frame_bgr = frame
        self.crop = self.first_point = None
        self.timeline.reset(self.duration)
        self.update_range()
        self.update_crop_text()
        self.render()
        self.update_info()

    def update_info(self):
        if self.video:
            self.info_var.set(f'{self.video.name} | {self.width}×{self.height} | {self.fps:.3f} fps | 当前位置 {timestamp(self.timeline.position)}')

    def resize_preview(self, event):
        if self.redraw_job:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(60, self.render)

    def render(self):
        self.redraw_job = None
        if self.frame_bgr is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / self.width, ch / self.height)
        dw, dh = max(1, round(self.width * scale)), max(1, round(self.height * scale))
        ox, oy = (cw - dw) / 2, (ch - dh) / 2
        self.display_box = (ox, oy, dw, dh)
        # 先用 SIMD 加速的 cv2 缩放，再对小图做颜色转换，远快于 PIL 全分辨率缩放。
        method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self.frame_bgr, (dw, dh), interpolation=method)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, image=self.photo, anchor='nw')
        if self.crop:
            x, y, w, h = self.crop
            self.canvas.create_rectangle(ox + x * dw / self.width, oy + y * dh / self.height,
                                         ox + (x + w) * dw / self.width, oy + (y + h) * dh / self.height,
                                         outline='#4ade80', width=2)
        if self.first_point:
            x, y = self.first_point
            px, py = ox + x * dw / self.width, oy + y * dh / self.height
            self.canvas.create_line(px - 8, py, px + 8, py, fill='#fbbf24', width=2)
            self.canvas.create_line(px, py - 8, px, py + 8, fill='#fbbf24', width=2)

    def crop_click(self, event):
        if not self.display_box:
            return
        ox, oy, dw, dh = self.display_box
        if not (ox <= event.x <= ox + dw and oy <= event.y <= oy + dh):
            return
        point = (min(self.width, max(0, round((event.x - ox) * self.width / dw))),
                 min(self.height, max(0, round((event.y - oy) * self.height / dh))))
        if self.first_point is None:
            self.first_point = point
            self.crop = None
        else:
            x1, y1 = self.first_point
            x2, y2 = point
            # YUV420/H.264 输出要求宽高为偶数；坐标也对齐到偶数。
            x, y = min(x1, x2) // 2 * 2, min(y1, y2) // 2 * 2
            right, bottom = max(x1, x2) // 2 * 2, max(y1, y2) // 2 * 2
            if right - x < 2 or bottom - y < 2:
                self.status_var.set('矩形太小，请重新选择第二个点。')
                return
            self.crop = (x, y, right - x, bottom - y)
            self.first_point = None
        self.update_crop_text()
        self.render()

    def reset_crop(self):
        self.crop = self.first_point = None
        self.update_crop_text()
        self.render()

    def update_crop_text(self):
        if self.first_point:
            text = f'已选第一个点 {self.first_point}，请单击另一对角点。'
        elif self.crop:
            x, y, w, h = self.crop
            text = f'裁剪范围：左上角 ({x}, {y})，尺寸 {w}×{h}（原视频像素）'
        else:
            text = '裁剪范围：整幅画面（导出时宽高向下对齐为偶数）'
        self.crop_var.set(text)

    def update_range(self):
        t = self.timeline
        self.range_var.set(f'S：{timestamp(t.start)}    E：{timestamp(t.end)}    导出时长：{t.end - t.start:.3f} 秒')

    def set_marker(self, marker):
        if not self.cap:
            return
        t = self.timeline
        gap = min(0.01, t.duration / 2)
        if marker == 'S':
            t.start = min(t.position, t.end - gap)
        else:
            t.end = max(t.position, t.start + gap)
        t.draw()
        self.update_range()

    def request_seek(self, seconds):
        self.pause()
        self.seek_target = seconds
        if self.seek_job is None:
            self.seek_job = self.root.after(40, self.flush_seek)

    def flush_seek(self):
        self.seek_job = None
        if self.cap and self.seek_target is not None:
            target = self.seek_target
            self.seek_target = None
            self.show_frame(min(self.frames - 1, max(0, int(target * self.fps + 1e-7))), seek=True)

    def show_frame(self, index, seek=False):
        if seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return False
        self.current_frame = index
        self.frame_bgr = frame
        self.timeline.position = index / self.fps
        self.timeline.draw()
        self.render()
        self.update_info()
        return True

    def pause(self):
        self.playing = False
        self.play_button.configure(text='▶ 播放')

    def toggle_play(self):
        if not self.cap:
            return
        if self.playing:
            self.pause()
            return
        if self.seek_job:
            self.root.after_cancel(self.seek_job)
            self.flush_seek()
        if self.current_frame >= self.frames - 1:
            self.show_frame(0, seek=True)
        self.playing = True
        self.play_button.configure(text='❚❚ 暂停')
        self.play_start = time.monotonic() - self.current_frame / self.fps
        self.play_tick()

    def play_tick(self):
        if not self.playing or not self.cap:
            return
        target = int((time.monotonic() - self.play_start) * self.fps)
        if target >= self.frames:
            self.pause()
            return
        if target > self.current_frame:
            gap = target - self.current_frame
            if gap > 1 and gap <= 8:
                # Linux 下 FFmpeg 后端 seek 很慢；少量落后时顺序 grab 跳帧更快。
                for _ in range(gap - 1):
                    if not self.cap.grab():
                        self.pause()
                        return
                gap = 1
            if not self.show_frame(target, seek=(gap > 1)):
                return
        # 按下一帧的绝对时间调度，补偿渲染耗时，避免节奏漂移。
        next_time = self.play_start + (self.current_frame + 1) / self.fps
        delay = max(1, round((next_time - time.monotonic()) * 1000))
        self.root.after(delay, self.play_tick)

    def export(self):
        if not self.video or self.exporting:
            return
        if self.first_point:
            messagebox.showwarning('框选未完成', '请先选择第二个对角点，或点击“重置框选”。')
            return
        if self.category_dialog is not None:
            self.category_dialog.lift()
            return
        self.pause()
        dialog = tk.Toplevel(self.root)
        self.category_dialog = dialog
        dialog.title('选择保存分类')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.withdraw()

        def dismiss():
            self.category_dialog = None
            dialog.destroy()

        def select(category):
            dismiss()
            self.export_category(category)

        dialog.protocol('WM_DELETE_WINDOW', dismiss)
        dialog.bind('<Escape>', lambda event: dismiss())
        content = ttk.Frame(dialog, padding=20)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text='请选择分类，点击后自动编号保存').pack(pady=(0, 12))
        grid = ttk.Frame(content)
        grid.pack(expand=True)
        for index, (label, category) in enumerate(CATEGORIES):
            ttk.Button(grid, text=label, width=22,
                       command=lambda value=category: select(value)).grid(
                           row=index // 3, column=index % 3,
                           padx=6, pady=6, ipady=14, sticky='nsew')
        dialog.update_idletasks()
        width, height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f'{width}x{height}+{max(0, x)}+{max(0, y)}')
        dialog.deiconify()
        dialog.grab_set()
        dialog.focus_set()

    def export_category(self, category):
        if not self.video or self.exporting:
            return
        if category not in {value for _, value in CATEGORIES}:
            return
        try:
            ffmpeg = find_ffmpeg()
        except RuntimeError as exc:
            messagebox.showerror('缺少 FFmpeg', str(exc))
            return
        try:
            output = str(next_output_path(ORGANIZED_FOLDER / category))
        except OSError as exc:
            messagebox.showerror('无法创建保存文件', str(exc))
            return
        self.pause()
        start, end = self.timeline.start, self.timeline.end
        crop = self.crop or (0, 0, self.width // 2 * 2, self.height // 2 * 2)
        if crop[2] < 2 or crop[3] < 2 or end <= start:
            messagebox.showerror('无效范围', '请检查时间范围与裁剪尺寸。')
            return
        self.exporting = True
        self.cancel_event.clear()
        self.export_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress['value'] = 0
        self.status_var.set('正在导出…（可以继续浏览其他视频）')
        args = (ffmpeg, str(self.video), output, start, end - start, crop)
        threading.Thread(target=self.export_worker, args=args, daemon=True).start()

    def export_worker(self, ffmpeg, source, output, start, length, crop):
        x, y, w, h = crop
        # 临时文件与目标同目录，成功后发布；目标已存在时拒绝覆盖。
        temp = str(Path(output).with_name(f'.{Path(output).stem}.{time.time_ns()}.partial.mp4'))
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-ss', f'{start:.6f}', '-i', source, '-t', f'{length:.6f}',
                   '-map', '0:v:0', '-map', '0:a:0?', '-vf', f'crop={w}:{h}:{x}:{y},setsar=1',
                   '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
                   '-progress', 'pipe:1', '-nostats', temp]
        process = None
        errors = []
        finished = threading.Event()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding='utf-8', errors='replace',
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            def read_errors():
                for line in process.stderr:
                    errors.append(line)
                    if len(errors) > 80:
                        del errors[0]
            def watch_cancel():
                while not finished.wait(0.1):
                    if self.cancel_event.is_set() and process.poll() is None:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                        return
            err_thread = threading.Thread(target=read_errors, daemon=True)
            err_thread.start()
            threading.Thread(target=watch_cancel, daemon=True).start()
            for line in process.stdout:
                key, _, value = line.strip().partition('=')
                if key == 'out_time_us':
                    try:
                        self.events.put(('progress', min(99, max(0, int(value) / 1_000_000 / length * 100))))
                    except ValueError:
                        pass
            code = process.wait()
            err_thread.join(timeout=2)
            if self.cancel_event.is_set():
                self.events.put(('cancelled', '导出已取消，原视频不受影响。'))
            elif code != 0:
                self.events.put(('error', ''.join(errors)[-5000:] or f'FFmpeg 退出码：{code}'))
            else:
                os.link(temp, output)
                self.events.put(('done', output))
        except Exception as exc:
            self.events.put(('error', str(exc)))
        finally:
            finished.set()
            if process:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass

    def poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    self.progress['value'] = value
                    self.status_var.set(f'正在导出：{value:.1f}%')
                    continue
                self.exporting = False
                self.export_button.configure(state='normal')
                self.cancel_button.configure(state='disabled')
                if kind == 'done':
                    self.progress['value'] = 100
                    self.status_var.set('导出完成')
                    messagebox.showinfo('导出完成', f'已保存：\n{value}')
                elif kind == 'error':
                    self.status_var.set('导出失败')
                    messagebox.showerror('导出失败', value)
                else:
                    self.progress['value'] = 0
                    self.status_var.set(value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def close(self):
        if self.exporting:
            messagebox.showwarning('正在导出', '请等待导出完成，或点击“取消导出”后再关闭。')
            return
        self.pause()
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoCropper(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()

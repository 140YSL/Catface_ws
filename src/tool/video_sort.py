#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频预览、多选分类和删除。依赖：opencv-python Pillow（以及 Tkinter）。
运行：python video_sort.py [视频目录]
默认读取 ../../cat_video/raw，分类保存至 ../../cat_video/organized。
分类按钮可多选；点击执行后复制完整文件，成功后删除源文件并播放下一项。
"""
import math
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import cv2
from PIL import Image, ImageTk

EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.mpg', '.mpeg', '.ts', '.wmv'}


VIDEO_ROOT = (Path(__file__).resolve().parent / '../../cat_video').resolve()
DEFAULT_FOLDER = VIDEO_ROOT / 'final'
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


def configure_ui_fonts(root):
    """显式使用中文无衬线字体，避免 Tk 默认字体回退到不合适的中文字形。"""
    available = {family.casefold(): family for family in tkfont.families(root)}
    candidates = (
        'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC',
        'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK JP',
    )
    family = next((available[name.casefold()] for name in candidates
                   if name.casefold() in available), None)
    if family is None:
        family = tkfont.nametofont('TkDefaultFont', root=root).actual('family')
    # 保留 Tk 从显示器读取的 DPI，字号使用点值，随 DPI 缩放。
    # 不把整个界面或文字绘制成位图再放大。
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        tkfont.nametofont(name, root=root).configure(
            family=family, size=12, weight='normal', slant='roman',
        )
    root.option_add('*Font', 'TkDefaultFont')
    style = ttk.Style(root)
    style.configure('.', font='TkDefaultFont')
    style.configure('TButton', font='TkDefaultFont', padding=(8, 6))
    style.configure('TLabel', font='TkDefaultFont')


def next_output_path(folder, suffix='.mp4'):
    """从 1 开始查找空闲编号，保留原视频扩展名，比较时忽略大小写。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    occupied = {path.name.casefold() for path in folder.iterdir()}
    number = 1
    while f'{number}{suffix}'.casefold() in occupied:
        number += 1
    return folder / f'{number}{suffix}'


def classify_file(source, categories, output_folder=ORGANIZED_FOLDER):
    """将完整文件复制到各分类；全部成功后才删除源文件，失败则回滚副本。"""
    source = Path(source)
    categories = tuple(dict.fromkeys(categories))
    if not categories or not set(categories) <= {value for _, value in CATEGORIES}:
        raise ValueError('请选择有效的视频分类')
    created = []
    try:
        for category in categories:
            # 独占创建，防止同时运行的程序覆盖同名文件。
            while True:
                destination = next_output_path(Path(output_folder) / category, source.suffix)
                try:
                    target = destination.open('xb')
                except FileExistsError:
                    continue
                break
            created.append(destination)
            with target, source.open('rb') as original:
                shutil.copyfileobj(original, target)
            shutil.copystat(source, destination)
        source.unlink()
        return created
    except Exception as exc:
        cleanup_errors = []
        for destination in created:
            try:
                destination.unlink()
            except OSError as cleanup_exc:
                cleanup_errors.append(f'{destination}: {cleanup_exc}')
        if cleanup_errors:
            raise OSError(f'{exc}\n部分副本未能清理：\n' + '\n'.join(cleanup_errors)) from exc
        raise


def timestamp(seconds):
    ms = max(0, round(seconds * 1000))
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{ms:03}'


class Timeline(tk.Canvas):
    """仅用于预览播放位置，不提供裁剪起止手柄。"""
    def __init__(self, master, seek):
        super().__init__(master, height=58, bg='#f4f5f7', highlightthickness=0)
        self.seek_callback = seek
        self.duration = 1.0
        self.position = 0.0
        self.enabled = False
        self.bind('<Configure>', lambda event: self.draw())
        self.bind('<Button-1>', self.seek)
        self.bind('<B1-Motion>', self.seek)

    def reset(self, duration):
        self.duration, self.position = duration, 0.0
        self.enabled = True
        self.draw()

    def draw(self):
        self.delete('all')
        left, right = 28, max(29, self.winfo_width() - 28)
        self.create_line(left, 20, right, 20, fill='#a6b8d1', width=4)
        x = left + self.position / max(self.duration, 0.001) * (right - left)
        self.create_oval(x - 7, 13, x + 7, 27, fill='#2563eb', outline='white')
        self.create_text(left, 45, text=timestamp(self.position), anchor='w', fill='#555', font='TkDefaultFont')
        self.create_text(right, 45, text=timestamp(self.duration), anchor='e', fill='#555', font='TkDefaultFont')

    def seek(self, event):
        if not self.enabled:
            return
        self.position = min(self.duration, max(0, (event.x - 28) /
                            max(self.winfo_width() - 56, 1) * self.duration))
        self.seek_callback(self.position)
        self.draw()


class VideoSorter:
    def __init__(self, root, initial_folder=None):
        self.root = root
        configure_ui_fonts(root)
        self.root.title('视频预览与多选分类')
        self.root.geometry('1440x850')
        self.root.minsize(1100, 650)
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
        self.playing = False
        self.play_job = None
        self.seek_target = None
        self.seek_job = None
        self.redraw_job = None
        self.events = queue.Queue()
        self.processing = False
        self.selected_categories = set()
        self.delete_selected = False
        self.folder_var = tk.StringVar(value='请选择视频文件夹')
        self.info_var = tk.StringVar(value='打开文件夹，然后选择左侧视频。')
        self.status_var = tk.StringVar(value='请选择一个或多个分类，或选择删除，然后执行。')
        self.build_ui()
        self.bind_shortcuts()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll_events)
        self.load_folder(initial_folder if initial_folder is not None else DEFAULT_FOLDER)

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        self.folder_button = ttk.Button(top, text='打开视频文件夹', command=self.choose_folder)
        self.folder_button.pack(side='left')
        ttk.Label(top, textvariable=self.folder_var).pack(side='left', padx=12)

        # 先为底栏分配空间，确保预览缩放时十一项操作始终可见。
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side='bottom', fill='x')
        ttk.Label(bottom, textvariable=self.status_var, wraplength=1050).pack(anchor='w', pady=(0, 6))
        actions = ttk.Frame(bottom)
        actions.pack(fill='x')
        self.category_buttons = {}
        for index, (label, category) in enumerate(CATEGORIES):
            button = tk.Button(actions, text=f'{label}\n[{index + 1}]', wraplength=95, height=3,
                               font='TkDefaultFont',
                               command=lambda value=category: self.toggle_category(value))
            button.grid(row=0, column=index, padx=2, sticky='nsew')
            actions.columnconfigure(index, weight=1, uniform='actions')
            self.category_buttons[category] = button
        self.button_bg = next(iter(self.category_buttons.values())).cget('background')
        self.delete_button = tk.Button(actions, text='删除\n[0]', height=3,
                                       font='TkDefaultFont', command=self.toggle_delete)
        self.delete_button.grid(row=0, column=9, padx=2, sticky='nsew')
        actions.columnconfigure(9, weight=1, uniform='actions')
        ttk.Separator(actions, orient='vertical').grid(row=0, column=10, padx=10, sticky='ns')
        self.execute_button = ttk.Button(actions, text='执行 [S]', command=self.execute)
        self.execute_button.grid(row=0, column=11, padx=2, sticky='nsew')
        actions.columnconfigure(11, weight=1, uniform='actions')

        body = ttk.Panedwindow(self.root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=8)
        sidebar = ttk.Frame(body, width=320)
        body.add(sidebar, weight=0)
        ttk.Label(sidebar, text='视频文件（单击切换）').pack(anchor='w', pady=4)
        self.listbox = tk.Listbox(sidebar, width=32, font='TkTextFont', exportselection=False)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.listbox.pack(fill='both', expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.select_video)
        panel = ttk.Frame(body)
        body.add(panel, weight=1)
        controls = ttk.Frame(panel)
        controls.pack(side='bottom', fill='x', pady=4)
        self.play_button = ttk.Button(controls, text='▶ 播放', command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Label(controls, text='空格 播放/暂停 | ←/→ 逐帧 | Ctrl+←/→ 跳转5秒').pack(side='left', padx=12)
        self.timeline = Timeline(panel, self.request_seek)
        self.timeline.pack(side='bottom', fill='x')
        ttk.Label(panel, textvariable=self.info_var).pack(side='bottom', anchor='w')
        self.canvas = tk.Canvas(panel, bg='#141414', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.resize_preview)
        self.refresh_actions()

    def toggle_category(self, category):
        if self.processing or self.delete_selected or not self.video:
            return
        if category in self.selected_categories:
            self.selected_categories.remove(category)
        else:
            self.selected_categories.add(category)
        self.refresh_actions()

    def toggle_delete(self):
        if self.processing or not self.video:
            return
        self.delete_selected = not self.delete_selected
        if self.delete_selected:
            self.selected_categories.clear()
        self.refresh_actions()

    def reset_selection(self):
        self.selected_categories.clear()
        self.delete_selected = False
        self.refresh_actions()

    def refresh_actions(self):
        available = self.video is not None and not self.processing
        for category, button in self.category_buttons.items():
            selected = category in self.selected_categories
            disabled = not available or self.delete_selected
            button.configure(state='disabled' if disabled else 'normal',
                             bg='#dedede' if disabled else '#2563eb' if selected else self.button_bg,
                             fg='white' if selected else 'black', disabledforeground='#888888',
                             activebackground='#1d4ed8' if selected else self.button_bg,
                             activeforeground='white' if selected else 'black',
                             relief='sunken' if selected else 'raised')
        self.delete_button.configure(state='normal' if available else 'disabled',
                                     bg='#dc2626' if self.delete_selected else self.button_bg,
                                     fg='white' if self.delete_selected else 'black',
                                     activebackground='#b91c1c' if self.delete_selected else self.button_bg,
                                     activeforeground='white' if self.delete_selected else 'black',
                                     relief='sunken' if self.delete_selected else 'raised')
        enabled = available and (self.delete_selected or bool(self.selected_categories))
        self.execute_button.configure(state='normal' if enabled else 'disabled')

    def bind_shortcuts(self):
        # 绑定到窗口：焦点位于列表、按钮或预览时均可使用。
        # 返回 break，避免方向键同时切换文件、空格同时触发按钮。
        shortcuts = {
            '<space>': self.toggle_play,
            '<Left>': lambda: self.keyboard_seek(-1, by_frame=True),
            '<Right>': lambda: self.keyboard_seek(1, by_frame=True),
            '<Control-Left>': lambda: self.keyboard_seek(-5),
            '<Control-Right>': lambda: self.keyboard_seek(5),
            # invoke 与点击按钮一致，按钮禁用时不会触发操作。
            '<KeyPress-0>': self.delete_button.invoke,
            '<KeyPress-s>': self.execute_button.invoke,
            '<KeyPress-S>': self.execute_button.invoke,
        }
        for index, (_, category) in enumerate(CATEGORIES, start=1):
            shortcuts[f'<KeyPress-{index}>'] = self.category_buttons[category].invoke
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

    def choose_folder(self):
        if self.processing:
            return
        folder = filedialog.askdirectory(title='选择包含视频的文件夹',
                                         initialdir=str(DEFAULT_FOLDER))
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        if self.processing:
            return
        try:
            files = sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.casefold())
        except OSError as exc:
            messagebox.showerror('无法读取文件夹', str(exc))
            return
        self.clear_video()
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
        if self.processing:
            return
        selection = self.listbox.curselection()
        if selection:
            path = self.files[selection[0]]
            if path != self.video:
                self.open_video(path)

    def open_video(self, path):
        if self.processing:
            return False
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
            return False
        if self.cap:
            self.cap.release()
        self.cap, self.video = cap, path
        self.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        self.frames = int(count)
        self.duration = self.frames / self.fps
        self.height, self.width = frame.shape[:2]
        self.current_frame = 0
        self.frame_bgr = frame
        self.reset_selection()
        self.timeline.reset(self.duration)
        self.render()
        self.update_info()
        return True

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
        # 先用 SIMD 加速的 cv2 缩放，再对小图做颜色转换，远快于 PIL 全分辨率缩放。
        method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self.frame_bgr, (dw, dh), interpolation=method)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, image=self.photo, anchor='nw')

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
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.play_button.configure(text='▶ 播放')

    def toggle_play(self):
        if not self.cap or self.processing:
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
        self.play_job = None
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
        self.play_job = self.root.after(delay, self.play_tick)

    def clear_video(self):
        self.pause()
        if self.seek_job is not None:
            self.root.after_cancel(self.seek_job)
            self.seek_job = None
        self.seek_target = None
        if self.cap:
            self.cap.release()
        self.cap = self.video = self.frame_bgr = self.photo = None
        self.canvas.delete('all')
        self.timeline.enabled = False
        self.timeline.duration, self.timeline.position = 1.0, 0.0
        self.timeline.draw()
        self.info_var.set('没有正在预览的视频。')
        self.reset_selection()

    def execute(self):
        if not self.video or self.processing:
            return
        if not self.delete_selected and not self.selected_categories:
            return
        source = self.video
        categories = tuple(value for _, value in CATEGORIES if value in self.selected_categories)
        delete_only = self.delete_selected
        self.pause()
        if self.seek_job is not None:
            self.root.after_cancel(self.seek_job)
            self.seek_job = None
        self.seek_target = None
        # 先释放预览句柄，确保 Windows 等系统也能删除源文件。
        if self.cap:
            self.cap.release()
            self.cap = None
        self.processing = True
        self.folder_button.configure(state='disabled')
        self.listbox.configure(state='disabled')
        self.play_button.configure(state='disabled')
        self.timeline.enabled = False
        self.refresh_actions()
        self.status_var.set('正在删除…' if delete_only else '正在复制到所选分类，完成后删除源文件…')
        threading.Thread(target=self.file_worker, args=(source, categories, delete_only), daemon=True).start()

    def file_worker(self, source, categories, delete_only):
        try:
            if delete_only:
                source.unlink()
                outputs = []
            else:
                outputs = classify_file(source, categories)
            self.events.put(('done', source, outputs))
        except Exception as exc:
            self.events.put(('error', source, str(exc)))

    def poll_events(self):
        try:
            while True:
                kind, source, value = self.events.get_nowait()
                self.processing = False
                self.folder_button.configure(state='normal')
                self.listbox.configure(state='normal')
                self.play_button.configure(state='normal')
                if kind == 'done':
                    index = self.files.index(source)
                    self.files.pop(index)
                    self.listbox.delete(index)
                    self.clear_video()
                    message = (f'已复制到 {len(value)} 个分类并删除源文件。' if value else '已删除当前文件。')
                    # 从后继文件开始；不可播放的条目保留在列表中并跳过。
                    candidates = self.files[index:] + self.files[:index]
                    for candidate in candidates:
                        next_index = self.files.index(candidate)
                        self.listbox.selection_clear(0, 'end')
                        self.listbox.selection_set(next_index)
                        self.listbox.see(next_index)
                        if self.open_video(candidate):
                            self.toggle_play()
                            break
                    self.status_var.set(message + (' 当前目录已处理完。' if not self.files else ''))
                else:
                    # 复制或删除失败时留在当前文件，不自动跳过。
                    self.open_video(source)
                    self.status_var.set('执行失败，当前文件未跳过。')
                    messagebox.showerror('执行失败', value)
                self.refresh_actions()
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def close(self):
        if self.processing:
            messagebox.showwarning('正在处理', '请等待文件操作完成后再关闭。')
            return
        self.clear_video()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoSorter(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()

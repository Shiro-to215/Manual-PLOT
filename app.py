"""Frame-by-frame skeleton correction tool. Run: python app.py"""
from __future__ import annotations

import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk
from pose_data import PoseData, FrameSource, EDGES, JOINTS
from video_export import export_video
from scrollable_sidebar import ScrollableSidebar

COLORS = [(59, 230, 175), (255, 184, 77), (193, 147, 255)]


class PoseEditor(tk.Tk):
    def __init__(self, auto_load=True):
        super().__init__()
        self.title('Pose Editor — 骨格座標修正')
        self.geometry('1380x900')
        self.minsize(980, 650)
        self.data = None
        self.source = None
        self.output_video_path = None
        self.output_video_manual = False
        self.row = 0
        self.selected = None
        self.playing = False
        self.timer = None
        self.drag = None
        self.pending = False
        self.pan = None
        self.preview = None
        self.zoom = 1.0
        self.shift = [0.0, 0.0]
        self.photo = None
        self.scale_factor = 1.0
        self.origin = (0, 0)
        self.media_index = None
        self.previous = tk.BooleanVar(value=True)
        self.following = tk.BooleanVar(value=True)
        self.labels = tk.BooleanVar(value=False)
        self.opacity = tk.IntVar(value=25)
        self.previous_count = tk.IntVar(value=1)
        self.following_count = tk.IntVar(value=1)
        self.frame_base = tk.StringVar(value='0')
        self.output_video_base = tk.StringVar(value='0')
        self.output_video_text = tk.StringVar(value='骨格出力用動画: 未選択（CSVのみ保存）')
        self.fps = tk.StringVar(value='23.976')
        self.mode = tk.StringVar(value='ドラッグ')
        self.person = tk.StringVar(value='すべて')
        self.status = tk.StringVar(value='CSVと元動画、または切り出した画像フォルダーを開いてください。')
        self.frame_text = tk.StringVar(value='フレーム —')
        self.detail = tk.StringVar(value='関節を選択してください')
        self._build()
        self._bind_events()
        self.protocol('WM_DELETE_WINDOW', self.close)
        default_csv = Path(__file__).with_name('PPP.csv')
        if auto_load and default_csv.exists():
            self.after(100, lambda: self.load_csv(default_csv))

    def _build(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TButton', padding=(10, 7))
        style.configure('TLabel', font=('Yu Gothic UI', 10))
        bar = ttk.Frame(self, padding=10)
        bar.pack(fill='x')
        ttk.Label(bar, text='POSE EDITOR', font=('Segoe UI', 15, 'bold')).pack(side='left', padx=(0, 20))
        for text, command in [('CSVを開く', self.open_csv), ('動画を開く', self.open_video),
                              ('画像フォルダー', self.open_folder), ('CSV・動画を書き出す', self.export)]:
            ttk.Button(bar, text=text, command=command).pack(side='left', padx=3)
        self.dirty_label = ttk.Label(bar, text='')
        self.dirty_label.pack(side='right')
        self.file_label = ttk.Label(self, text='', padding=(12, 0, 12, 8))
        self.file_label.pack(fill='x')
        video_bar = ttk.Frame(self, padding=(10, 0, 10, 8))
        video_bar.pack(fill='x')
        ttk.Button(video_bar, text='骨格出力用動画を選択', command=self.choose_output_video).pack(side='left')
        ttk.Button(video_bar, text='解除', command=self.clear_output_video).pack(side='left', padx=4)
        ttk.Label(video_bar, text='動画先頭のCSV frame').pack(side='left', padx=(8, 4))
        ttk.Entry(video_bar, textvariable=self.output_video_base, width=7).pack(side='left')
        ttk.Label(video_bar, textvariable=self.output_video_text).pack(side='left', padx=10)
        body = ttk.Frame(self)
        body.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(body, background='#131c2a', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.sidebar = ScrollableSidebar(body)
        self.sidebar.pack(side='right', fill='y')
        side = self.sidebar.content
        ttk.Label(side, text='表示と編集', font=('Yu Gothic UI', 13, 'bold')).pack(anchor='w', pady=(0, 12))
        ttk.Label(side, text='編集方式').pack(anchor='w')
        for name in ('ドラッグ', '2回クリック'):
            ttk.Radiobutton(side, text=name, variable=self.mode, value=name,
                            command=self.cancel_selection).pack(anchor='w')
        ttk.Separator(side).pack(fill='x', pady=10)
        ttk.Checkbutton(side, text='前の骨格（青）', variable=self.previous, command=self.render).pack(anchor='w')
        ttk.Checkbutton(side, text='次の骨格（赤）', variable=self.following, command=self.render).pack(anchor='w')
        ttk.Checkbutton(side, text='関節名を表示', variable=self.labels, command=self.render).pack(anchor='w')
        for label, variable in [('前に表示するフレーム数', self.previous_count),
                                ('後に表示するフレーム数', self.following_count)]:
            line = ttk.Frame(side)
            line.pack(fill='x', pady=4)
            ttk.Label(line, text=label).pack(side='left')
            count_box = ttk.Combobox(line, values=list(range(1, 11)), textvariable=variable,
                                     state='readonly', width=4)
            count_box.pack(side='right')
            count_box.bind('<<ComboboxSelected>>', lambda e: self.render())
        ttk.Label(side, text='前後の骨格の不透明度').pack(anchor='w')
        ttk.Scale(side, from_=0, to=70, variable=self.opacity, command=lambda _: self.render()).pack(fill='x')
        ttk.Label(side, text='近いフレームを基準に、遠いほど薄く表示', foreground='#536073').pack(anchor='w')
        ttk.Separator(side).pack(fill='x', pady=10)
        ttk.Label(side, text='人物（列スロット / IDは下の一覧）').pack(anchor='w')
        self.person_box = ttk.Combobox(side, textvariable=self.person, state='readonly', values=['すべて'])
        self.person_box.pack(fill='x', pady=(4, 8))
        self.person_box.bind('<<ComboboxSelected>>', lambda e: self.change_person())
        ttk.Label(side, text='関節一覧 — 空欄の関節も選択して配置可能').pack(anchor='w')
        joint_frame = ttk.Frame(side)
        joint_frame.pack(fill='both', expand=True, pady=5)
        self.joint_list = tk.Listbox(joint_frame, exportselection=False, font=('Yu Gothic UI', 10),
                                    activestyle='none', selectbackground='#216e77', height=8)
        self.joint_list.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(joint_frame, command=self.joint_list.yview)
        scroll.pack(side='right', fill='y')
        self.joint_list.configure(yscrollcommand=scroll.set)
        self.joint_list.bind('<<ListboxSelect>>', self.select_from_list)
        self.list_keys = []
        ttk.Label(side, textvariable=self.detail, wraplength=270).pack(anchor='w', pady=6)
        ttk.Button(side, text='表示を全体に戻す（Home）', command=self.fit).pack(fill='x', pady=3)
        undo_bar = ttk.Frame(side)
        undo_bar.pack(fill='x', pady=4)
        ttk.Button(undo_bar, text='元に戻す', command=self.undo).pack(side='left', expand=True, fill='x')
        ttk.Button(undo_bar, text='やり直す', command=lambda: self.undo(True)).pack(side='left', expand=True, fill='x')
        ttk.Label(side, text='← / → : 1フレーム移動\nSpace : 再生 / 一時停止\nホイール : 拡大・縮小\n右ドラッグ : 表示を移動\nEsc : 選択解除\nCtrl+Z / Ctrl+Y : 元に戻す / やり直す\nCtrl+S : CSV出力',
                  foreground='#536073').pack(anchor='w', pady=(10, 0))
        self.sidebar.bind_mousewheel()
        controls = ttk.Frame(self, padding=10)
        controls.pack(fill='x')
        ttk.Button(controls, text='◀ 前', command=lambda: self.step(-1)).pack(side='left')
        self.play_button = ttk.Button(controls, text='▶ 再生', command=self.toggle_play)
        self.play_button.pack(side='left', padx=5)
        ttk.Button(controls, text='次 ▶', command=lambda: self.step(1)).pack(side='left')
        ttk.Label(controls, textvariable=self.frame_text).pack(side='left', padx=15)
        ttk.Label(controls, text='fps').pack(side='left')
        ttk.Entry(controls, textvariable=self.fps, width=7).pack(side='left', padx=4)
        ttk.Label(controls, text='画像先頭に対応するCSV frame').pack(side='left', padx=(12, 4))
        base = ttk.Entry(controls, textvariable=self.frame_base, width=6)
        base.pack(side='left')
        base.bind('<Return>', lambda e: self.update_frame())
        ttk.Button(controls, text='適用', command=self.update_frame).pack(side='left', padx=3)
        self.timeline = ttk.Scale(self, from_=0, to=1, command=self.seek)
        self.timeline.pack(fill='x', padx=12, pady=(0, 8))
        ttk.Label(self, textvariable=self.status, relief='sunken', padding=(10, 6)).pack(fill='x')

    def _bind_events(self):
        self.bind('<Left>', lambda e: self.key_action(e, lambda: self.step(-1)))
        self.bind('<Right>', lambda e: self.key_action(e, lambda: self.step(1)))
        self.bind('<space>', lambda e: self.key_action(e, self.toggle_play))
        self.bind('<Home>', lambda e: self.key_action(e, self.fit))
        self.bind('<Escape>', lambda e: self.cancel_selection())
        self.bind('<Control-z>', lambda e: self.key_action(e, self.undo))
        self.bind('<Control-y>', lambda e: self.key_action(e, lambda: self.undo(True)))
        self.bind('<Control-s>', lambda e: self.export())
        self.canvas.bind('<Configure>', lambda e: self.render())
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-3>', self.pan_start)
        self.canvas.bind('<B3-Motion>', self.pan_move)
        self.canvas.bind('<ButtonRelease-3>', lambda e: setattr(self, 'pan', None))

    def key_action(self, event, command):
        if isinstance(event.widget, (ttk.Entry, ttk.Spinbox, ttk.Combobox, tk.Entry)):
            return
        command()
        return 'break'

    def confirm_discard(self):
        return not self.data or not self.data.dirty or messagebox.askyesno(
            '未保存の修正', '未保存の座標修正があります。保存せずに続けますか？', parent=self)

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[('座標CSV', '*.csv')])
        if path:
            self.load_csv(path)

    def load_csv(self, path):
        if not self.confirm_discard():
            return
        try:
            data = PoseData(path)
        except Exception as exc:
            messagebox.showerror('CSVを開けません', str(exc), parent=self)
            return
        self.stop()
        self.data = data
        self.row = 0
        self.selected = None
        self.pending = False
        self.person.set('すべて')
        self.person_box.configure(values=['すべて'] + data.people)
        self.fps.set(f'{data.fps:.6f}')
        self.frame_base.set(str(min(data.frames)))
        self.timeline.configure(to=max(1, len(data.rows) - 1))
        self.update_frame()

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[('動画', '*.mp4 *.avi *.mov *.mkv *.wmv *.m4v'), ('すべて', '*.*')])
        if path:
            self.load_source(path)

    def open_folder(self):
        path = filedialog.askdirectory(title='切り出した連番画像のフォルダー')
        if path:
            self.load_source(path, True)

    def choose_output_video(self):
        path = filedialog.askopenfilename(title='骨格を重ねる元動画',
            filetypes=[('動画', '*.mp4 *.avi *.mov *.mkv *.wmv *.m4v'), ('すべて', '*.*')])
        if path:
            self.set_output_video(path, manual=True)

    def set_output_video(self, path, manual=False):
        self.output_video_path = Path(path)
        self.output_video_manual = manual
        self.output_video_base.set(self.frame_base.get())
        self.output_video_text.set(f'骨格出力用: {self.output_video_path.name}')

    def clear_output_video(self):
        self.output_video_path = None
        self.output_video_manual = True
        self.output_video_text.set('骨格出力用動画: 未選択（CSVのみ保存）')

    def load_source(self, path, folder=False):
        self.stop()
        try:
            source = FrameSource(path, folder)
        except Exception as exc:
            messagebox.showerror('画像を開けません', str(exc), parent=self)
            return
        if self.source:
            self.source.close()
        self.source = source
        if not folder and not self.output_video_manual:
            self.set_output_video(path)
        self.zoom, self.shift = 1.0, [0.0, 0.0]
        if source.fps:
            self.fps.set(f'{source.fps:.6f}')
        self.update_frame()
        self.status.set(f'{source.count}枚 / {source.size[0]}×{source.size[1]} px。CSVと画像のフレーム対応を確認してください。')

    def update_frame(self):
        self.stop()
        self.drag = self.preview = None
        self.pending = False
        if self.data:
            self.row = max(0, min(self.row, len(self.data.rows) - 1))
            if self.selected and self.selected not in self.data.points:
                self.selected = None
            self.timeline.set(self.row)
        self.refresh_list()
        self.render()

    def step(self, delta):
        if self.data:
            self.row += delta
            self.update_frame()

    def seek(self, value):
        if self.data:
            row = max(0, min(len(self.data.rows) - 1, round(float(value))))
            if row != self.row:
                self.row = row
                self.update_frame()

    def stop(self):
        self.playing = False
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        self.play_button.configure(text='▶ 再生')

    def toggle_play(self):
        if self.playing:
            self.stop()
            return
        if not self.data or not self.source:
            return
        try:
            fps = float(self.fps.get())
            if not math.isfinite(fps) or not 0 < fps <= 240:
                raise ValueError
        except ValueError:
            messagebox.showerror('再生速度', 'fpsは0より大きく240以下の数値にしてください。')
            return
        if self.row == len(self.data.rows) - 1:
            self.row = 0
            self.update_frame()
        self.playing = True
        self.play_button.configure(text='⏸ 一時停止')
        self.tick()

    def tick(self):
        self.timer = None
        if not self.playing:
            return
        if self.row >= len(self.data.rows) - 1:
            self.stop()
            return
        current = self.data.frames[self.row]
        self.row += 1
        gap = abs(self.data.frames[self.row] - current)
        self.timeline.set(self.row)
        self.pending = False
        self.refresh_list()
        self.render()
        if self.playing:
            try:
                rate = float(self.fps.get())
                if not math.isfinite(rate) or not 0 < rate <= 240:
                    raise ValueError
            except ValueError:
                self.stop()
                return
            self.timer = self.after(max(1, round(1000 * max(1, gap) / rate)), self.tick)

    def visible_keys(self):
        if not self.data:
            return []
        return [key for key in self.data.points if self.person.get() == 'すべて' or key[0] == self.person.get()]

    def refresh_list(self):
        self.joint_list.delete(0, 'end')
        self.list_keys = self.visible_keys()
        for index, key in enumerate(self.list_keys):
            point = self.data.point(self.row, key)
            identity = self.data.identity(self.row, key[0]) or '—'
            changed = any(self.data.rows[self.row][col] != self.data.original[self.row][col] for col in self.data.points[key])
            self.joint_list.insert('end', f'{"●" if changed else " "} {key[0]} [ID:{identity}] {key[1]}{"（空）" if point is None else ""}')
            if key == self.selected:
                self.joint_list.selection_set(index)
                self.joint_list.see(index)
        self.update_detail()

    def update_detail(self):
        if self.data and self.selected:
            point = self.preview or self.data.point(self.row, self.selected)
            text = f'{self.selected[0]} / {self.selected[1]}'
            if point:
                text += f'\nx = {point[0]:.3f}   y = {point[1]:.3f} px'
            else:
                text += '\n画像上をクリックして配置'
            col = self.data.columns.get('_'.join(self.selected) + '_score')
            if col is not None:
                text += f'\n推定時のscore: {self.data.rows[self.row][col] or "空欄"}'
            self.detail.set(text)
        else:
            self.detail.set('関節を選択してください')

    def change_person(self):
        self.cancel_selection()
        self.refresh_list()

    def select_from_list(self, event=None):
        selection = self.joint_list.curselection()
        if selection:
            self.stop()
            self.selected = self.list_keys[selection[0]]
            self.pending = True
            self.preview = None
            self.update_detail()
            self.status.set('画像上をクリックすると、選択した関節をその位置へ移動します。Escで解除。')
            self.render()

    def cancel_selection(self):
        self.selected = self.drag = self.preview = None
        self.pending = False
        self.refresh_list()
        self.render()

    def fit(self):
        self.zoom, self.shift = 1.0, [0.0, 0.0]
        self.render()

    def screen(self, point):
        return (point[0] * self.scale_factor + self.origin[0], point[1] * self.scale_factor + self.origin[1])

    def image_point(self, event):
        if self.media_index is None or not self.source:
            return None
        x = (event.x - self.origin[0]) / self.scale_factor
        y = (event.y - self.origin[1]) / self.scale_factor
        width, height = self.source.get(self.media_index).size
        if 0 <= x <= width - 1 and 0 <= y <= height - 1:
            return x, y
        return None

    def render(self):
        self.canvas.delete('all')
        self.media_index = None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.dirty_label.configure(text='● 未保存の修正あり' if self.data and self.data.dirty else '')
        self.file_label.configure(text=f'CSV: {self.data.path.name if self.data else "未選択"}    |    画像: {self.source.path.name if self.source else "未選択"}')
        if not self.data:
            self.canvas.create_text(cw / 2, ch / 2, text='座標CSVを開いてください', fill='#c5d1e2', font=('Yu Gothic UI', 18))
            return
        frame = self.data.frames[self.row]
        self.frame_text.set(f'frame {frame}   ({self.row + 1} / {len(self.data.rows)})')
        if not self.source:
            self.canvas.create_text(cw / 2, ch / 2, text='元動画 または 切り出した画像フォルダーを開いてください\n\nCSVを読み込み済みです。画像の上に骨格を重ねて修正できます。',
                                    fill='#c5d1e2', font=('Yu Gothic UI', 14), justify='center')
            return
        try:
            index = frame - int(self.frame_base.get())
            original = self.source.get(index)
        except Exception as exc:
            self.canvas.create_text(cw / 2, ch / 2, text=f'{exc}\n「画像先頭に対応するCSV frame」を確認してください。', fill='#ffb7a4', width=cw - 40)
            self.stop()
            return
        self.media_index = index
        iw, ih = original.size
        self.scale_factor = min(cw / iw, ch / ih) * self.zoom
        s = self.scale_factor
        self.origin = ((cw - iw * s) / 2 + self.shift[0], (ch - ih * s) / 2 + self.shift[1])
        ox, oy = self.origin
        # Transform directly to the viewport so zooming never creates a huge bitmap.
        base = original.transform((cw, ch), Image.Transform.AFFINE,
                                  (1 / s, 0, -ox / s, 0, 1 / s, -oy / s),
                                  resample=Image.Resampling.BILINEAR, fillcolor='#131c2a').convert('RGBA')
        try:
            previous_count = max(1, min(10, int(self.previous_count.get())))
            following_count = max(1, min(10, int(self.following_count.get())))
            alpha = max(0, min(180, int(self.opacity.get() * 2.55)))
        except (ValueError, tk.TclError):
            previous_count, following_count, alpha = 1, 1, 64
        # Draw distant frames first; nearby frames remain visible above them.
        for distance in range(max(previous_count, following_count), 0, -1):
            faded_alpha = round(alpha / distance)
            if faded_alpha == 0:
                continue
            for enabled, count, delta, color in [
                    (self.previous.get(), previous_count, -distance, (90, 163, 255)),
                    (self.following.get(), following_count, distance, (255, 125, 148))]:
                other = self.data.by_frame.get(frame + delta)
                if not enabled or distance > count or other is None:
                    continue
                layer = Image.new('RGBA', (cw, ch))
                draw = ImageDraw.Draw(layer)
                for person in self.data.people:
                    if self.person.get() not in ('すべて', person):
                        continue
                    match = self.data.matching_person(self.row, other, person)
                    if match:
                        self.draw_person(draw, other, match, color + (faded_alpha,), ghost=True)
                base = Image.alpha_composite(base, layer)
        layer = Image.new('RGBA', (cw, ch))
        draw = ImageDraw.Draw(layer)
        for i, person in enumerate(self.data.people):
            if self.person.get() in ('すべて', person):
                self.draw_person(draw, self.row, person, COLORS[i % len(COLORS)] + (255,))
        base = Image.alpha_composite(base, layer)
        self.photo = ImageTk.PhotoImage(base)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        for key in self.visible_keys():
            point = self.current_point(self.row, key)
            if point and (self.labels.get() or key == self.selected):
                x, y = self.screen(point)
                label = f'{key[0]} / {key[1]}' if key == self.selected else key[1]
                self.canvas.create_text(x + 11, y - 12, text=label, fill='#07121e', anchor='sw', font=('Yu Gothic UI', 11, 'bold'))
                self.canvas.create_text(x + 10, y - 13, text=label, fill='white', anchor='sw', font=('Yu Gothic UI', 11, 'bold'))
        self.canvas.create_text(14, 14, text=f'frame {frame}  •  {self.zoom:.1f}×  |  前: 青 / 現在: 人物色 / 次: 赤',
                                fill='#e4edf9', anchor='nw', font=('Yu Gothic UI', 10))

    def current_point(self, row, key):
        if row == self.row and key == self.selected and self.preview is not None:
            return self.preview
        return self.data.point(row, key)

    def draw_person(self, draw, row, person, color, ghost=False):
        points = {joint: self.screen(point) for p, joint in self.data.points if p == person
                  and (point := (self.data.point(row, (p, joint)) if ghost else self.current_point(row, (p, joint)))) is not None}
        for a, b in EDGES:
            if a in points and b in points:
                draw.line([points[a], points[b]], fill=color, width=2 if ghost else 3)
        for joint, (x, y) in points.items():
            radius = 4 if ghost else 6
            if not ghost and (person, joint) == self.selected:
                radius = 9
                draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline='white', width=2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color,
                         outline=None if ghost else '#132638', width=1)

    def press(self, event):
        self.canvas.focus_set()
        self.stop()
        point = self.image_point(event)
        if point is None:
            return
        if self.pending and self.selected:
            self.commit(point)
            return
        hits = []
        for key in self.visible_keys():
            p = self.data.point(self.row, key)
            if p:
                x, y = self.screen(p)
                d = math.hypot(x - event.x, y - event.y)
                if d <= 15:
                    hits.append((d, key))
        if not hits:
            self.cancel_selection()
            return
        self.selected = min(hits)[1]
        self.refresh_list()
        if self.mode.get() == '2回クリック':
            self.pending = True
            self.status.set('移動先をもう一度クリックしてください。Escで解除。')
        else:
            self.drag = (event.x, event.y)
        self.render()

    def motion(self, event):
        if self.drag:
            point = self.image_point(event)
            if point:
                self.preview = point
                self.update_detail()
                self.render()

    def release(self, event):
        if self.drag:
            moved = math.hypot(event.x - self.drag[0], event.y - self.drag[1]) > 2
            point = self.image_point(event)
            self.drag = None
            self.preview = None
            if moved and point:
                self.commit(point)
            else:
                self.render()

    def commit(self, point):
        self.data.move(self.row, self.selected, *point)
        self.pending = False
        self.preview = self.drag = None
        self.status.set(f'frame {self.data.frames[self.row]} / {self.selected[0]} / {self.selected[1]} を修正しました。')
        self.refresh_list()
        self.render()

    def wheel(self, event):
        if not self.source or self.media_index is None:
            return
        old = self.zoom
        new = max(0.5, min(20.0, old * (1.2 if event.delta > 0 else 1 / 1.2)))
        ratio = new / old
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.shift = [(self.shift[0] + cw / 2 - event.x) * ratio - cw / 2 + event.x,
                      (self.shift[1] + ch / 2 - event.y) * ratio - ch / 2 + event.y]
        self.zoom = new
        self.render()

    def pan_start(self, event):
        self.pan = (event.x, event.y, *self.shift)

    def pan_move(self, event):
        if self.pan:
            x, y, sx, sy = self.pan
            self.shift = [sx + event.x - x, sy + event.y - y]
            self.render()

    def undo(self, redo=False):
        self.stop()
        if self.data and (result := self.data.undo(redo)):
            self.row, self.selected = result
            self.person.set('すべて')
            self.update_frame()
            self.status.set('やり直しました。' if redo else '修正を元に戻しました。')

    def export(self):
        if not self.data:
            return
        self.stop()
        destination = filedialog.asksaveasfilename(title='修正後CSVを保存', defaultextension='.csv',
            initialdir=str(self.data.path.parent), initialfile=self.data.path.stem + '_corrected.csv',
            filetypes=[('座標CSV', '*.csv')])
        if not destination:
            return
        input_paths = [self.output_video_path]
        if self.source and not self.source.files:
            input_paths.append(self.source.path)
        if any(path and Path(destination).resolve() == Path(path).resolve() for path in input_paths):
            messagebox.showerror('保存先', 'CSVの保存先に入力動画は指定できません。', parent=self)
            return
        if Path(destination).resolve() == self.data.path.resolve():
            if not messagebox.askyesno('元CSVへの上書き', '読み込み元のCSVを上書きします。続けますか？'):
                return
        if self.output_video_path:
            try:
                base = int(self.output_video_base.get())
            except ValueError:
                messagebox.showerror('フレーム番号', '動画先頭のCSV frameに整数を入力してください。', parent=self)
                return
            video_destination = filedialog.asksaveasfilename(title='骨格付き動画を保存',
                defaultextension='.mp4', initialdir=str(Path(destination).parent),
                initialfile=Path(destination).stem + '_skeleton.mp4',
                filetypes=[('MP4動画', '*.mp4')])
            if not video_destination:
                return
            protected = [self.output_video_path, self.data.path, Path(destination)]
            if self.source and not self.source.files:
                protected.append(self.source.path)
            if any(Path(video_destination).resolve() == Path(path).resolve() for path in protected):
                messagebox.showerror('保存先', '骨格付き動画は、入力動画・入力CSV・出力CSVと異なる保存先を指定してください。', parent=self)
                return
            self.export_with_video(destination, video_destination, base)
            return
        try:
            self.data.export(destination)
        except Exception as exc:
            messagebox.showerror('保存できません', str(exc))
            return
        self.status.set(f'保存しました: {destination}')
        self.render()

    def export_with_video(self, csv_path, video_path, base):
        events = queue.SimpleQueue()
        dialog = tk.Toplevel(self)
        dialog.title('骨格付き動画を作成中')
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.protocol('WM_DELETE_WINDOW', lambda: None)
        ttk.Label(dialog, text='動画に修正後の骨格を重ねています。').pack(padx=24, pady=(20, 8))
        progress_label = ttk.Label(dialog, text='準備中…')
        progress_label.pack(padx=24)
        progress_bar = ttk.Progressbar(dialog, length=340, maximum=100)
        progress_bar.pack(padx=24, pady=(8, 20))
        dialog.grab_set()
        expected_size = self.source.size if self.source else None
        video_input = self.output_video_path

        def work():
            video_saved = False
            try:
                export_video(self.data, video_input, video_path, base, expected_size,
                             lambda current, total: events.put(('progress', current, total)))
                video_saved = True
                self.data.export(csv_path)
                events.put(('done',))
            except Exception as exc:
                events.put(('error', str(exc), video_saved))

        def poll():
            try:
                while True:
                    event = events.get_nowait()
                    if event[0] == 'progress':
                        _, current, total = event
                        progress_bar['value'] = current * 100 / total
                        progress_label.configure(text=f'{current} / {total} フレーム')
                    else:
                        dialog.grab_release()
                        dialog.destroy()
                        if event[0] == 'done':
                            self.status.set(f'保存しました: {csv_path} / {video_path}')
                            self.render()
                            messagebox.showinfo('保存完了', f'CSV: {csv_path}\n\n骨格付き動画: {video_path}', parent=self)
                        else:
                            suffix = f'\n骨格付き動画は保存済みです: {video_path}' if event[2] else '\nCSVは保存していません。'
                            messagebox.showerror('保存できません', event[1] + suffix, parent=self)
                        return
            except queue.Empty:
                dialog.after(100, poll)

        threading.Thread(target=work, daemon=True).start()
        dialog.after(100, poll)

    def close(self):
        if self.confirm_discard():
            self.stop()
            if self.source:
                self.source.close()
            self.destroy()


if __name__ == '__main__':
    PoseEditor().mainloop()

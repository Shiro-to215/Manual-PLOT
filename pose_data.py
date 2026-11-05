"""CSV storage independent of the GUI. Unedited cells are kept verbatim."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path


JOINTS = ['鼻', '左目', '右目', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘',
          '左手首', '右手首', '左股関節', '右股関節', '左膝', '右膝', '左足首', '右足首']
EDGES = [('鼻', '左目'), ('鼻', '右目'), ('左目', '左耳'), ('右目', '右耳'),
         ('左肩', '右肩'), ('左肩', '左肘'), ('左肘', '左手首'),
         ('右肩', '右肘'), ('右肘', '右手首'), ('左肩', '左股関節'),
         ('右肩', '右股関節'), ('左股関節', '右股関節'),
         ('左股関節', '左膝'), ('左膝', '左足首'), ('右股関節', '右膝'), ('右膝', '右足首')]


class PoseData:
    def __init__(self, path):
        self.path = Path(path)
        for encoding in ('utf-8-sig', 'cp932'):
            try:
                with self.path.open(encoding=encoding, newline='') as stream:
                    table = list(csv.reader(stream))
                break
            except UnicodeDecodeError:
                if encoding == 'cp932':
                    raise ValueError('CSVの文字コードを読み取れません。UTF-8またはCP932をご使用ください。')
        if len(table) < 2:
            raise ValueError('CSVにヘッダーとデータ行が必要です。')
        self.header, self.rows = table[0], table[1:]
        if len(set(self.header)) != len(self.header):
            raise ValueError('CSVに重複した列名があります。')
        if any(len(row) != len(self.header) for row in self.rows):
            raise ValueError('CSVの列数が揃っていません。')
        self.columns = {name: i for i, name in enumerate(self.header)}
        if 'frame' not in self.columns:
            raise ValueError('frame列が必要です。PPP.csvと同じ横持ち形式をご使用ください。')
        self.points = {}
        for name in self.header:
            match = re.fullmatch(r'(person_\d+)_(.+)_x', name)
            if match and name[:-1] + 'y' in self.columns:
                person, joint = match.groups()
                self.points[person, joint] = (self.columns[name], self.columns[name[:-1] + 'y'])
        if not self.points:
            raise ValueError('person_0_左膝_x / person_0_左膝_y のような座標列が必要です。')
        self.people = sorted({p for p, j in self.points}, key=lambda p: int(p.split('_')[1]))
        self.frames = []
        for row in self.rows:
            value = float(row[self.columns['frame']])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError('frame列には0以上の整数が必要です。')
            self.frames.append(int(value))
        if len(set(self.frames)) != len(self.frames):
            raise ValueError('frame列が重複しています。1フレーム1行のCSVをご使用ください。')
        self.by_frame = {frame: i for i, frame in enumerate(self.frames)}
        self.original = [row[:] for row in self.rows]
        self.saved = [row[:] for row in self.rows]
        self.undo_stack = []
        self.redo_stack = []

    @property
    def dirty(self):
        return self.rows != self.saved

    @property
    def fps(self):
        col = self.columns.get('time_seconds')
        if col is not None and len(self.rows) > 1:
            try:
                dt = float(self.rows[1][col]) - float(self.rows[0][col])
                df = self.frames[1] - self.frames[0]
                if dt > 0 and df > 0:
                    return df / dt
            except ValueError:
                pass
        return 24.0

    def point(self, row, key):
        if row is None or key not in self.points:
            return None
        a, b = self.points[key]
        try:
            x, y = float(self.rows[row][a]), float(self.rows[row][b])
            return (x, y) if math.isfinite(x) and math.isfinite(y) else None
        except ValueError:
            return None

    def identity(self, row, person):
        col = self.columns.get(person + '_id')
        return self.rows[row][col] if col is not None else person

    def matching_person(self, current_row, other_row, person):
        """Match ghosts by tracked ID, not by a potentially changing slot."""
        identity = self.identity(current_row, person)
        if not identity:
            return None
        for candidate in self.people:
            if self.identity(other_row, candidate) == identity:
                return candidate
        return None

    def move(self, row, key, x, y):
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError('座標は有限の数値にしてください。')
        a, b = self.points[key]
        before = (self.rows[row][a], self.rows[row][b])
        after = (format(x, '.10f').rstrip('0').rstrip('.'), format(y, '.10f').rstrip('0').rstrip('.'))
        if before == after:
            return
        self.rows[row][a], self.rows[row][b] = after
        self.undo_stack.append((row, key, before, after))
        self.redo_stack.clear()

    def undo(self, redo=False):
        source, target = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not source:
            return None
        change = source.pop()
        row, key, before, after = change
        a, b = self.points[key]
        self.rows[row][a], self.rows[row][b] = after if redo else before
        target.append(change)
        return row, key

    def export(self, path):
        """Write atomically so an interrupted save cannot truncate an existing CSV."""
        import os
        import tempfile
        destination = Path(path)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                             dir=destination.parent, delete=False, suffix='.tmp') as stream:
                temp = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(self.header)
                writer.writerows(self.rows)
            os.replace(temp, destination)
        finally:
            if temp is not None and temp.exists():
                temp.unlink()
        self.saved = [row[:] for row in self.rows]


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', Path(path).name)]


class FrameSource:
    def __init__(self, path, folder=False):
        self.path = Path(path)
        self.capture = None
        self.cache = {}
        self.files = []
        self.fps = None
        if folder:
            self.files = sorted((p for p in self.path.iterdir() if p.suffix.lower() in
                                 {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}), key=natural_key)
            if not self.files:
                raise ValueError('フォルダーに画像がありません。')
            self.count = len(self.files)
        else:
            from video_reader import VideoReader
            self.capture = VideoReader(path)
            self.count = self.capture.count
            self.fps = self.capture.fps
        try:
            first = self.get(0)
            self.size = first.size
        except Exception:
            self.close()
            raise

    def get(self, index):
        from PIL import Image, ImageOps
        if not 0 <= index < self.count:
            raise ValueError(f'画像フレーム {index} は範囲外です（0〜{self.count - 1}）。')
        if index in self.cache:
            return self.cache[index]
        if self.files:
            with Image.open(self.files[index]) as original:
                result = ImageOps.exif_transpose(original).convert('RGB')
        else:
            result = self.capture.read(index).to_image()
        self.cache[index] = result
        if len(self.cache) > 8:
            del self.cache[next(iter(self.cache))]
        return result

    def close(self):
        if self.capture is not None:
            self.capture.close()
        self.cache.clear()

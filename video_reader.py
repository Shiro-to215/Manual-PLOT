"""Video input via PyAV, indexed by frames actually decoded in display order."""
from bisect import bisect_right
import math
from pathlib import Path

import av


class VideoReader:
    def __init__(self, path):
        self.path = Path(path)
        self.container = None
        self.count = 0
        self.size = None
        self.fps = None
        self._keyframes = []
        self._key_indices = []
        self._by_pts = None
        timestamps = []
        try:
            self._reopen()
            rate = self.stream.guessed_rate or self.stream.average_rate or self.stream.base_rate
            if rate and math.isfinite(float(rate)) and float(rate) > 0:
                self.fps = float(rate)
            # stream.frames can be an estimate. Only decoded frames determine the count.
            for frame in self._decoder:
                if self.size is None:
                    self.size = (frame.width, frame.height)
                timestamps.append(frame.pts)
                if frame.key_frame and frame.pts is not None:
                    self._keyframes.append(frame.pts)
                    self._key_indices.append(self.count)
                self.count += 1
            if self.count == 0:
                raise ValueError('動画から画像を読み取れません。')
            if all(t is not None for t in timestamps) and all(a < b for a, b in zip(timestamps, timestamps[1:])):
                self._by_pts = {pts: index for index, pts in enumerate(timestamps)}
            self._reopen()
        except Exception:
            self.close()
            raise

    def _reopen(self):
        self.close()
        self.container = av.open(str(self.path))
        if not self.container.streams.video:
            raise ValueError('動画ファイルに映像ストリームがありません。')
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self._decoder = self.container.decode(self.stream)
        self._next_index = 0
        self._seeked = False

    def _seek(self, index):
        key = bisect_right(self._key_indices, index) - 1
        if self._by_pts is None or key < 0:
            self._reopen()
            return
        try:
            self.container.seek(self._keyframes[key], stream=self.stream, backward=True, any_frame=False)
            self._decoder = self.container.decode(self.stream)
            self._next_index = self._key_indices[key]
            self._seeked = True
        except av.FFmpegError:
            self._reopen()

    def _read(self, index):
        while True:
            frame = next(self._decoder)
            position = self._by_pts.get(frame.pts) if self._seeked else self._next_index
            if position is None or position > index:
                raise LookupError('シーク後のフレーム番号を確定できません。')
            self._next_index = position + 1
            if position == index:
                return frame

    def read(self, index):
        if not 0 <= index < self.count:
            raise ValueError(f'動画フレーム {index} は範囲外です（0〜{self.count - 1}）。')
        if self.container is None:
            raise ValueError('動画は既に閉じられています。')
        if index < self._next_index or index - self._next_index > 48:
            self._seek(index)
        try:
            return self._read(index)
        except (LookupError, StopIteration, av.FFmpegError):
            # Some files seek imprecisely. Reopening preserves exact CSV frame numbering.
            self._reopen()
            try:
                return self._read(index)
            except (StopIteration, av.FFmpegError) as exc:
                raise ValueError(f'動画のフレーム {index} を読み取れません。') from exc

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

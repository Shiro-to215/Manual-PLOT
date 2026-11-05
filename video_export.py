"""Render corrected COCO17 poses with RTMLib 0.0.16's default drawing style."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import cv2

from pose_data import JOINTS
from video_reader import VideoReader

# RTMLib's coco17 palette is passed directly to OpenCV in BGR order.
ORANGE = (51, 153, 255)
GREEN = (0, 255, 0)
BLUE = (255, 128, 0)
JOINT_COLORS = [ORANGE] * 5 + [GREEN, BLUE] * 6
LINKS = [
    (15, 13, GREEN), (13, 11, GREEN), (16, 14, BLUE), (14, 12, BLUE),
    (11, 12, ORANGE), (5, 11, ORANGE), (6, 12, ORANGE), (5, 6, ORANGE),
    (5, 7, GREEN), (6, 8, BLUE), (7, 9, GREEN), (8, 10, BLUE),
    (1, 2, ORANGE), (0, 1, ORANGE), (0, 2, ORANGE), (1, 3, ORANGE),
    (2, 4, ORANGE), (3, 5, ORANGE), (4, 6, ORANGE),
]
THRESHOLD = 0.45


def draw_corrected_pose(frame, data, row):
    """Match RTMLib circle/line order and palette, displaying manually changed points."""
    for person in data.people:
        points = []
        for index, joint in enumerate(JOINTS):
            key = (person, joint)
            point = data.point(row, key)
            if point is None:
                points.append(None)
                continue
            xcol, ycol = data.points[key]
            changed = (data.rows[row][xcol] != data.original[row][xcol] or
                       data.rows[row][ycol] != data.original[row][ycol])
            score_col = data.columns.get(f'{person}_{joint}_score')
            try:
                score = float(data.rows[row][score_col]) if score_col is not None else 0.0
            except ValueError:
                score = 0.0
            if not changed and (not math.isfinite(score) or score < THRESHOLD):
                points.append(None)
                continue
            # OpenCV's C integer coordinates must fit in signed 32 bits.
            if not all(-2147483648 <= v <= 2147483647 for v in point):
                points.append(None)
                continue
            pt = (int(point[0]), int(point[1]))
            points.append(pt)
            cv2.circle(frame, pt, 2, JOINT_COLORS[index], -1)
        for a, b, color in LINKS:
            if points[a] is not None and points[b] is not None:
                cv2.line(frame, points[a], points[b], color, thickness=2)
    return frame


def export_video(data, video_path, destination, frame_base, expected_size=None, progress=None):
    """Create MP4 in a temporary file, then replace the destination on success."""
    video_path, destination = Path(video_path), Path(destination)
    if video_path.resolve() == destination.resolve():
        raise ValueError('出力動画に入力動画と同じパスは指定できません。')
    capture = None
    writer = None
    temporary = None
    try:
        capture = VideoReader(video_path)
        count = capture.count
        fps = capture.fps
        width, height = capture.size
        if count <= 0 or fps is None or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('動画のフレーム数・fps・画像サイズを取得できません。')
        if width % 2 or height % 2:
            raise ValueError('動画の幅と高さは偶数である必要があります。')
        if expected_size and (width, height) != expected_size:
            raise ValueError(f'出力用動画の解像度 {width}×{height} が編集画像 {expected_size[0]}×{expected_size[1]} と異なります。')
        if min(data.frames) < frame_base or max(data.frames) >= frame_base + count:
            raise ValueError(f'CSVの全フレームが動画に収まりません。動画はCSV frame {frame_base}〜{frame_base + count - 1} に対応します。')
        fd, filename = tempfile.mkstemp(prefix='.pose-', suffix='.mp4', dir=destination.parent)
        os.close(fd)
        temporary = Path(filename)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('MP4動画を作成できません。動画コーデックと保存先を確認してください。')
        for index in range(count):
            frame = capture.read(index).to_ndarray(format='bgr24')
            if frame.shape[:2] != (height, width):
                raise ValueError(f'動画のフレーム {index} のサイズが変化しました。')
            row = data.by_frame.get(frame_base + index)
            if row is not None:
                draw_corrected_pose(frame, data, row)
            writer.write(frame)
            if progress and (index % 10 == 0 or index + 1 == count):
                progress(index + 1, count)
        writer.release()
        writer = None
        if temporary.stat().st_size == 0:
            raise ValueError('動画が書き出されませんでした。')
        os.replace(temporary, destination)
        temporary = None
    finally:
        if capture is not None:
            capture.close()
        if writer is not None:
            writer.release()
        if temporary is not None and temporary.exists():
            temporary.unlink()

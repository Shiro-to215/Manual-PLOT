import csv
import hashlib
import os
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import time

from PIL import Image

from pose_data import PoseData, FrameSource

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def workspace_temp():
    # Use inherited workspace permissions, including under Windows sandbox tokens.
    path = ROOT / ('.test-' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != ROOT.resolve() or not resolved.name.startswith('.test-'):
            raise RuntimeError('Unexpected test cleanup path')
        shutil.rmtree(resolved)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.data = PoseData(ROOT / 'PPP.csv')

    def test_actual_schema(self):
        self.assertEqual(len(self.data.rows), 281)
        self.assertEqual(len(self.data.header), 158)
        self.assertEqual(len(self.data.points), 51)
        self.assertEqual(self.data.frames, list(range(281)))
        self.assertAlmostEqual(self.data.fps, 23.976023976)
        self.assertIsNone(self.data.point(0, ('person_1', '左膝')))

    def test_export_changes_only_selected_coordinates(self):
        key = ('person_0', '左膝')
        self.data.move(15, key, 100.125, 200.75)
        self.assertTrue(self.data.dirty)
        with workspace_temp() as directory:
            output = Path(directory) / 'corrected.csv'
            self.data.export(output)
            self.assertFalse(self.data.dirty)
            with output.open(encoding='utf-8-sig', newline='') as stream:
                table = list(csv.reader(stream))
            self.assertEqual(table[0], self.data.header)
            allowed = set(self.data.points[key])
            for r, (before, after) in enumerate(zip(self.data.original, table[1:])):
                for c, (a, b) in enumerate(zip(before, after)):
                    if r != 15 or c not in allowed:
                        self.assertEqual(a, b, (r, c))
            reloaded = PoseData(output)
            self.assertEqual(reloaded.point(15, key), (100.125, 200.75))

    def test_undo_redo_and_saved_state(self):
        key = ('person_0', '右足首')
        original = self.data.point(7, key)
        self.data.move(7, key, 42.5, 50.0)
        self.assertEqual(self.data.undo(), (7, key))
        self.assertEqual(self.data.point(7, key), original)
        self.assertFalse(self.data.dirty)
        self.data.undo(redo=True)
        self.assertEqual(self.data.point(7, key), (42.5, 50.0))
        with workspace_temp() as directory:
            self.data.export(Path(directory) / 'saved.csv')
            self.data.undo()
            self.assertTrue(self.data.dirty)
            self.data.undo(redo=True)
            self.assertFalse(self.data.dirty)
        self.data.undo()
        self.data.move(1, key, 0.0, 0.0)
        self.assertIsNone(self.data.undo(redo=True))
        self.assertEqual(self.data.point(1, key), (0.0, 0.0))

    def test_missing_joint_can_be_placed_without_changing_score_or_id(self):
        key = ('person_1', '左膝')
        self.data.move(0, key, 12.5, 90.0)
        self.assertEqual(self.data.point(0, key), (12.5, 90.0))
        for name in ('person_1_id', 'person_1_左膝_score'):
            self.assertEqual(self.data.rows[0][self.data.columns[name]], '')
        self.data.undo()
        self.assertIsNone(self.data.point(0, key))

    def test_ghosts_match_identity_across_slots(self):
        id0 = self.data.columns['person_0_id']
        id1 = self.data.columns['person_1_id']
        self.data.rows[1][id0] = '99'
        self.data.rows[1][id1] = '1'
        self.assertEqual(self.data.matching_person(0, 1, 'person_0'), 'person_1')
        self.data.rows[1][id1] = ''
        self.assertIsNone(self.data.matching_person(0, 1, 'person_0'))

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            self.data.move(0, ('person_0', '鼻'), float('nan'), 2)


class MediaTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('POSE_TEST_VIDEO'), 'Set POSE_TEST_VIDEO for the C0034 regression test')
    def test_c0034_decoded_count_and_complete_export(self):
        import av
        from video_export import export_video
        path = Path(os.environ['POSE_TEST_VIDEO'])
        sample_indices = {0, 12, 143, 279, 280}
        expected = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            self.assertEqual(stream.frames, 284)  # file metadata differs from decoded frames
            for index, frame in enumerate(container.decode(stream)):
                if index in sample_indices:
                    expected[index] = hashlib.sha256(frame.to_image().tobytes()).digest()
        # Neither display nor export should use OpenCV's video reader.
        with patch('cv2.VideoCapture', side_effect=AssertionError('OpenCV input must not be used')):
            source = FrameSource(path)
            try:
                self.assertEqual(source.count, 281)
                self.assertEqual(source.size, (1920, 1080))
                for index in [280, 0, 143, 12, 279]:
                    self.assertEqual(hashlib.sha256(source.get(index).tobytes()).digest(), expected[index])
                with self.assertRaises(ValueError):
                    source.get(281)
            finally:
                source.close()
            with workspace_temp() as directory:
                output = directory / 'corrected.mp4'
                data = PoseData(ROOT / 'PPP.csv')
                export_video(data, path, output, 0, (1920, 1080))
                with av.open(str(output)) as container:
                    container.streams.video[0].thread_type = 'AUTO'
                    count = sum(1 for _ in container.decode(video=0))
                self.assertEqual(count, 281)
                # A genuine one-frame mismatch is still rejected.
                with self.assertRaisesRegex(ValueError, '全フレーム'):
                    export_video(data, path, directory / 'invalid.mp4', 1)

    def test_natural_image_order_and_bounds(self):
        with workspace_temp() as directory:
            for name, color in [('frame_10.png', 'red'), ('frame_2.png', 'blue')]:
                Image.new('RGB', (64, 48), color).save(Path(directory) / name)
            source = FrameSource(directory, folder=True)
            try:
                self.assertEqual(source.count, 2)
                self.assertEqual(source.get(0).getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(source.get(1).getpixel((0, 0)), (255, 0, 0))
                with self.assertRaises(ValueError):
                    source.get(2)
            finally:
                source.close()

    def test_corrected_pose_video_export(self):
        import cv2
        import numpy as np
        from video_export import draw_corrected_pose, export_video, GREEN, BLUE
        with workspace_temp() as directory:
            source_data = PoseData(ROOT / 'PPP.csv')
            rows = [row[:] for row in source_data.rows[:3]]
            for row in rows:
                for person, joint in source_data.points:
                    a, b = source_data.points[person, joint]
                    row[a] = row[b] = ''
                    score = source_data.columns.get(f'{person}_{joint}_score')
                    if score is not None:
                        row[score] = ''
                for joint, x, y, score in [('左膝', 15, 20, .1), ('左足首', 20, 35, .9),
                                            ('右膝', 40, 20, .9), ('鼻', 50, 10, .1)]:
                    row[source_data.columns[f'person_0_{joint}_x']] = str(x)
                    row[source_data.columns[f'person_0_{joint}_y']] = str(y)
                    row[source_data.columns[f'person_0_{joint}_score']] = str(score)
            csv_path = directory / 'input.csv'
            with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(source_data.header)
                writer.writerows(rows)
            data = PoseData(csv_path)
            data.move(0, ('person_0', '左膝'), 20, 20)
            canvas = np.zeros((48, 64, 3), dtype=np.uint8)
            draw_corrected_pose(canvas, data, 0)
            self.assertEqual(tuple(canvas[20, 20]), GREEN)  # corrected low-score joint
            self.assertEqual(tuple(canvas[27, 20]), GREEN)  # corrected knee to ankle line
            self.assertEqual(tuple(canvas[20, 40]), BLUE)
            self.assertEqual(tuple(canvas[10, 50]), (0, 0, 0))  # low-score, not edited
            source = directory / 'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            output = directory / 'output.mp4'
            updates = []
            export_video(data, source, output, -1, (64, 48), lambda n, total: updates.append((n, total)))
            cap = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25)
                ok, before = cap.read()
                self.assertTrue(ok)
                self.assertLess(int(before.max()), 10)  # CSV has no frame -1
                ok, corrected = cap.read()
                self.assertTrue(ok)
                self.assertGreater(int(corrected[20, 20, 1]), 100)
                self.assertGreater(int(corrected[20, 40, 0]), 100)
            finally:
                cap.release()
            self.assertEqual(updates[-1], (4, 4))
            with self.assertRaisesRegex(ValueError, '解像度'):
                export_video(data, source, directory / 'bad.mp4', 0, (100, 100))
            self.assertFalse((directory / 'bad.mp4').exists())
            with self.assertRaisesRegex(ValueError, '全フレーム'):
                export_video(data, source, directory / 'bad.mp4', 2)
            self.assertFalse((directory / 'bad.mp4').exists())

    def test_video_read_and_backward_seek(self):
        import cv2
        import numpy as np
        with workspace_temp() as directory:
            path = str(Path(directory) / 'test.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (64, 48))
            self.assertTrue(writer.isOpened())
            for n in range(12):
                writer.write(np.full((48, 64, 3), n * 20, dtype=np.uint8))
            writer.release()
            source = FrameSource(path)
            try:
                self.assertEqual(source.count, 12)
                self.assertAlmostEqual(source.fps, 24)
                for index in [0, 11, 3, 10, 1]:
                    self.assertLess(abs(source.get(index).getpixel((0, 0))[0] - index * 20), 5)
            finally:
                source.close()


class GuiTests(unittest.TestCase):
    def test_sidebar_scroll_reaches_controls_in_short_window(self):
        from app import PoseEditor
        app = PoseEditor(auto_load=False)
        app.attributes('-alpha', 0)
        try:
            app.geometry('980x650')
            app.load_csv(ROOT / 'PPP.csv')
            app.update()
            panel = app.sidebar.canvas
            self.assertLess(panel.yview()[1], 1)
            self.assertEqual(panel.yview()[0], 0)
            app.zoom = 2
            # Wheel over a settings control scrolls the panel without changing its value.
            app.person_box.event_generate('<MouseWheel>', delta=-120)
            app.update()
            self.assertGreater(panel.yview()[0], 0)
            self.assertEqual(app.person.get(), 'すべて')
            self.assertEqual(app.zoom, 2)
            # The list retains its own scroll until it reaches its bottom.
            panel.yview_moveto(0)
            app.joint_list.yview_moveto(0)
            app.update()
            app.joint_list.event_generate('<MouseWheel>', delta=-120)
            app.update()
            self.assertGreater(app.joint_list.yview()[0], 0)
            self.assertEqual(panel.yview()[0], 0)
            app.joint_list.yview_moveto(1)
            app.joint_list.event_generate('<MouseWheel>', delta=-120)
            app.update()
            self.assertGreater(panel.yview()[0], 0)
            # Dragging the outer scrollbar to the bottom exposes the action buttons.
            panel.yview_moveto(1)
            app.update()

            def find_buttons(widget):
                result = {}
                for child in widget.winfo_children():
                    if child.winfo_class() == 'TButton':
                        result[child.cget('text')] = child
                    result.update(find_buttons(child))
                return result

            buttons = find_buttons(app.sidebar.content)
            top, bottom = panel.winfo_rooty(), panel.winfo_rooty() + panel.winfo_height()
            for text in ('表示を全体に戻す（Home）', '元に戻す', 'やり直す'):
                button = buttons[text]
                self.assertGreaterEqual(button.winfo_rooty(), top)
                self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), bottom)
            buttons['表示を全体に戻す（Home）'].invoke()
            self.assertEqual(app.zoom, 1)
            # Growing the window restores a full, unscrolled view.
            app.maxsize(3000, 3000)
            app.geometry('1380x1500')
            app.update()
            self.assertEqual(panel.yview(), (0, 1))
        finally:
            app.destroy()

    def test_csv_and_video_are_saved_together(self):
        import cv2
        import numpy as np
        from app import PoseEditor
        with workspace_temp() as directory:
            original = PoseData(ROOT / 'PPP.csv')
            csv_input = directory / 'input.csv'
            with csv_input.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(original.header)
                writer.writerows(original.rows[:3])
            video_input = directory / 'input.avi'
            writer = cv2.VideoWriter(str(video_input), cv2.VideoWriter_fourcc(*'MJPG'), 24, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            csv_output, video_output = directory / 'corrected.csv', directory / 'corrected.mp4'
            app = PoseEditor(auto_load=False)
            app.attributes('-alpha', 0)
            try:
                app.update()
                app.withdraw()
                app.load_csv(csv_input)
                app.load_source(video_input)
                self.assertEqual(app.output_video_path, video_input)
                app.data.move(0, ('person_0', '左膝'), 20, 20)
                with patch('app.filedialog.asksaveasfilename', side_effect=[str(csv_output), str(video_output)]), \
                     patch('app.messagebox.showinfo') as success, \
                     patch('app.messagebox.showerror') as error:
                    app.export()
                    deadline = time.monotonic() + 10
                    while not success.called and not error.called and time.monotonic() < deadline:
                        app.update()
                        time.sleep(.01)
                    self.assertFalse(error.called, error.call_args)
                    self.assertTrue(success.called, 'Export did not complete')
                self.assertTrue(csv_output.exists())
                self.assertTrue(video_output.exists())
                self.assertEqual(PoseData(csv_output).point(0, ('person_0', '左膝')), (20, 20))
                self.assertFalse(app.data.dirty)
            finally:
                if app.source:
                    app.source.close()
                app.destroy()

    def test_multiple_ghost_frames_fade_and_respect_controls(self):
        from app import PoseEditor
        app = PoseEditor(auto_load=False)
        app.attributes('-alpha', 0)
        try:
            app.update()
            app.withdraw()
            app.load_csv(ROOT / 'PPP.csv')
            with workspace_temp() as directory:
                Image.new('RGB', (2048, 1080), '#334455').save(directory / 'frame.png')
                app.load_source(directory, folder=True)
                app.row = 140
                app.frame_base.set('140')
                app.previous_count.set(3)
                app.following_count.set(2)

                def ghosts():
                    with patch.object(app, 'draw_person', wraps=app.draw_person) as draw:
                        app.render()
                    return [(call.args[1], call.args[3]) for call in draw.call_args_list
                            if call.kwargs.get('ghost')]

                drawn = ghosts()
                self.assertEqual([row for row, color in drawn], [137, 138, 142, 139, 141])
                for row, color in drawn:
                    self.assertEqual(color[:3], (90, 163, 255) if row < 140 else (255, 125, 148))
                    self.assertEqual(color[3], round(int(25 * 2.55) / abs(row - 140)))
                app.previous_count.set(10)
                app.following_count.set(10)
                drawn = ghosts()
                self.assertEqual(len(drawn), 20)
                self.assertEqual({row for row, color in drawn}, set(range(130, 151)) - {140})
                previous_alphas = [color[3] for row, color in drawn if row < 140]
                self.assertTrue(all(a < b for a, b in zip(previous_alphas, previous_alphas[1:])))
                app.opacity.set(50)
                self.assertEqual(dict(ghosts())[139][3], int(50 * 2.55))
                app.previous.set(False)
                self.assertTrue(all(row > 140 for row, color in ghosts()))
                app.previous.set(True)
                app.row = 0
                app.frame_base.set('0')
                self.assertEqual({row for row, color in ghosts()}, set(range(1, 11)))
                app.opacity.set(0)
                self.assertEqual(ghosts(), [])
        finally:
            if app.source:
                app.source.close()
            app.destroy()

    def test_render_drag_click_seek_and_zoom(self):
        from app import PoseEditor
        app = PoseEditor(auto_load=False)
        app.attributes('-alpha', 0)
        try:
            app.update()
            app.withdraw()
            app.load_csv(ROOT / 'PPP.csv')
            with workspace_temp() as directory:
                # A synthetic background exercises the display transform; it is not real footage.
                for i in range(3):
                    Image.new('RGB', (2048, 1080), '#334455').save(Path(directory) / f'{i}.png')
                app.load_source(directory, folder=True)
                app.render()
                self.assertEqual(app.media_index, 0)
                self.assertIsNotNone(app.photo)
                key = ('person_0', '左膝')
                app.person.set('person_0')
                app.selected = key
                app.pending = True
                x, y = app.screen((1000.0, 800.0))
                app.press(SimpleNamespace(x=x, y=y))
                px, py = app.data.point(0, key)
                self.assertAlmostEqual(px, 1000)
                self.assertAlmostEqual(py, 800)
                # Drag the isolated joint, then verify conversion back into image pixels.
                x, y = app.screen((1000, 800))
                app.press(SimpleNamespace(x=x, y=y))
                end_x, end_y = app.screen((1100, 850))
                end = SimpleNamespace(x=end_x, y=end_y)
                app.motion(end)
                app.release(end)
                self.assertEqual(app.data.point(0, key), (1100, 850))
                # Keep the image coordinate under the cursor fixed when zooming.
                event = SimpleNamespace(x=300, y=250, delta=120)
                before = app.image_point(event)
                app.wheel(event)
                after = app.image_point(event)
                self.assertIsNotNone(before)
                self.assertAlmostEqual(before[0], after[0])
                self.assertAlmostEqual(before[1], after[1])
                app.step(1)
                self.assertEqual(app.media_index, 1)
                app.mode.set('2回クリック')
                app.data.move(1, key, 1000, 800)
                start = SimpleNamespace(**dict(zip(('x', 'y'), app.screen((1000, 800)))))
                app.press(start)
                app.release(start)
                self.assertTrue(app.pending)
                app.press(SimpleNamespace(**dict(zip(('x', 'y'), app.screen((900, 700))))))
                self.assertEqual(app.data.point(1, key), (900, 700))
                app.undo()
                self.assertEqual(app.row, 1)
                app.undo()
                app.undo()
                self.assertEqual(app.row, 0)
                app.fit()
                self.assertEqual(app.zoom, 1)
                app.seek('2')
                self.assertEqual(app.row, 2)
                app.seek('0')
                app.toggle_play()
                self.assertTrue(app.playing)
                app.stop()
        finally:
            if app.source:
                app.source.close()
            app.destroy()


if __name__ == '__main__':
    unittest.main()

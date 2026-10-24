"""Run: python -m unittest discover -s tests -p test_prepare_scaling_data.py -v"""
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
import wave

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_scaling_data.py'
spec = importlib.util.spec_from_file_location('prepare_scaling_data', SCRIPT)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
sys.path.insert(0, str(SCRIPT.parent))
from level_match_scaling_data import float_payload, level_match_prepared


class PlanningTests(unittest.TestCase):
    def test_guard_and_split_bounds(self):
        train = [p.window('0:104'), p.window('244:304')]
        excluded = list(map(p.window, p.HISTORICAL))
        plan = p.plan_windows(train, [p.window('216:224')], [p.window('232:240')],
                              excluded, 310.4)
        self.assertEqual(sum(c['end_sample'] - c['start_sample'] for c in plan
                             if c['split'] == 'train'), 164 * p.RATE)
        self.assertTrue(all(c['end_sample'] - c['start_sample'] <= 20 * p.RATE for c in plan))
        self.assertEqual({c['split'] for c in plan}, {'train', 'validation', 'test'})
        with self.assertRaisesRegex(ValueError, 'guard'):
            p.plan_windows([p.window('0:111')], [p.window('216:224')],
                           [p.window('232:240')], excluded, 310.4)

    def test_reject_overlap_invalid_and_memory(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            p.plan_windows([p.window('0:5')], [p.window('4:6')], [p.window('8:9')], [], 10)
        for text in ('nan:1', '0:inf', '-1:1', '1:1', 'broken'):
            with self.assertRaises(ValueError):
                p.window(text)
        with self.assertRaisesRegex(ValueError, '128'):
            p.plan_windows([p.window('0:2100')], [p.window('2200:2210')],
                           [p.window('2300:2310')], [], 2400)

    def test_guard_touch_is_half_open(self):
        p.plan_windows([p.window('0:2')], [p.window('4:5')], [p.window('8:9')], [], 10)
        with self.assertRaisesRegex(ValueError, 'outside'):
            p.plan_windows([p.window('0:2')], [p.window('4:5')], [p.window('8:11')], [], 10)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires installed FFmpeg')
class RealDecodeTests(unittest.TestCase):
    def test_provenance_sample_counts_safe_paths_and_exclusivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source $(touch DO_NOT_CREATE); recording.wav'
            with wave.open(str(source), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(b''.join(struct.pack('<h', ((i % 100) - 50) * 100)
                                       for i in range(4 * 8000)))
            before = p.sha256(source)
            output = root / 'prepared'
            result = p.prepare(source, output, [p.window('0:1')], [p.window('2:2.5')],
                               [p.window('3:3.5')], [], guard=.1, chunk=.6)
            self.assertEqual(before, p.sha256(source))
            self.assertEqual(result['source_sha256_after'], before)
            self.assertFalse((root / 'DO_NOT_CREATE').exists())
            saved = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(saved['source'], str(source.resolve()))
            self.assertEqual(saved['source_sha256'], before)
            for crop in saved['crops']:
                self.assertEqual(crop['sha256'], p.sha256(output / crop['path']))
                self.assertEqual(crop['samples'], crop['end_sample'] - crop['start_sample'])
                self.assertIn(str(source.resolve()), crop['decode_argv'])
            self.assertEqual(len((output / 'manifest.txt').read_text().splitlines()), 3)
            before_manifest = (output / 'manifest.txt').read_bytes()
            with self.assertRaises(FileExistsError):
                p.prepare(source, output, [p.window('0:1')], [p.window('2:2.5')],
                          [p.window('3:3.5')], [], guard=.1)
            self.assertEqual(before_manifest, (output / 'manifest.txt').read_bytes())
            matched = root / 'matched'
            target = .1067798255
            matched_data = level_match_prepared(output, matched, target)
            self.assertEqual(matched_data['level_matching']['parent_provenance_sha256'],
                             p.sha256(output / 'provenance.json'))
            self.assertEqual(before, p.sha256(source))
            self.assertEqual(before_manifest, (output / 'manifest.txt').read_bytes())
            for crop, raw_crop in zip(matched_data['crops'], saved['crops']):
                self.assertAlmostEqual(crop['rms'], target, places=8)
                self.assertEqual(crop['raw_crop']['sha256'], raw_crop['sha256'])
                self.assertEqual(crop['start_sample'], raw_crop['start_sample'])
                self.assertEqual(crop['end_sample'], raw_crop['end_sample'])
                raw_values = struct.iter_unpack('<f', float_payload(output / raw_crop['path']))
                gained_values = struct.iter_unpack('<f', float_payload(matched / crop['path']))
                for (raw_value,), (gained_value,) in zip(raw_values, gained_values):
                    expected = struct.unpack('<f', struct.pack('<f', raw_value * crop['gain']))[0]
                    self.assertEqual(gained_value, expected)
            with self.assertRaises(FileExistsError):
                level_match_prepared(output, matched, target)
            for invalid_target in (0., -1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    level_match_prepared(output, root / 'invalid-matching', invalid_target)


if __name__ == '__main__':
    unittest.main()

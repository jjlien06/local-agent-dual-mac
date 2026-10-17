"""Protocol-only tests for an experimental witness, not a model runtime."""
import importlib.util
from pathlib import Path
import unittest


def binding_type():
    path = Path(__file__).with_name('witness.py')
    if not path.exists():
        raise AssertionError('BoundRope witness is not implemented')
    spec = importlib.util.spec_from_file_location('_attention_witness_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, 'BoundRope', None)
    if not callable(candidate):
        raise AssertionError('BoundRope witness is not implemented')
    return candidate


class BindingTests(unittest.TestCase):
    def test_wrong_chunk_length_rejected_before_rotary_execution(self):
        class Rotary:
            def apply_rotary(self, *args, **kwargs):
                raise AssertionError('wrong chunk length reached native rotary')
        class Array:
            shape = (1, 24, 2, 256)
        obj = binding_type()(Rotary())
        obj.bind(object(), physical_offset=10, tokens=3)
        with self.assertRaises(ValueError):
            obj(Array(), offset=10)

    def test_wrong_physical_offset_rejected_before_rotary_execution(self):
        calls = []
        class Rotary:
            def apply_rotary(self, *args, **kwargs):
                calls.append(args)
                raise AssertionError('wrong offset reached native rotary')
        class Array:
            shape = (1, 24, 3, 256)
        obj = binding_type()(Rotary())
        obj.bind(object(), physical_offset=10, tokens=3)
        with self.assertRaises(ValueError):
            obj(Array(), offset=7)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

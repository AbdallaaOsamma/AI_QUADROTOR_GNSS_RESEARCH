"""Tests for export_tensorrt.py — no TensorRT needed (import mocked)."""
import importlib
import sys
from unittest.mock import MagicMock, patch


def test_module_importable():
    mod = importlib.import_module("scripts.export_tensorrt")
    assert hasattr(mod, "export_tensorrt")
    assert hasattr(mod, "main")


def test_returns_false_when_tensorrt_missing():
    """If tensorrt is not installed, export_tensorrt should return False gracefully."""
    with patch.dict(sys.modules, {"tensorrt": None}):
        from scripts.export_tensorrt import export_tensorrt
        result = export_tensorrt("model.onnx", "model.trt")
    assert result is False


def test_returns_false_on_parse_error(tmp_path):
    """If the ONNX parser fails, export_tensorrt should return False."""
    fake_trt = MagicMock()
    fake_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    fake_parser = MagicMock()
    fake_parser.parse.return_value = False
    fake_parser.num_errors = 1
    fake_parser.get_error.return_value = "bad onnx"
    fake_trt.OnnxParser.return_value = fake_parser
    fake_trt.Builder.return_value.create_network.return_value = MagicMock()

    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"fake")

    with patch.dict(sys.modules, {"tensorrt": fake_trt}):
        import importlib
        import scripts.export_tensorrt as mod
        importlib.reload(mod)
        result = mod.export_tensorrt(str(onnx_file), str(tmp_path / "out.trt"))
    assert result is False

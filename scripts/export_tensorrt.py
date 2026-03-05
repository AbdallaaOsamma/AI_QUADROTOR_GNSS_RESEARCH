"""Export ONNX model to TensorRT engine for Jetson Xavier NX.

Must be run on the target Jetson device (TensorRT is device-specific).

Usage (on Jetson):
    python scripts/export_tensorrt.py --onnx model.onnx --output model.trt
    python scripts/export_tensorrt.py --onnx model.onnx --output model.trt --fp16
"""
import argparse
import sys


def export_tensorrt(onnx_path: str, engine_path: str, fp16: bool = False) -> bool:
    try:
        import tensorrt as trt
    except ImportError:
        print("[trt] ERROR: tensorrt not available. Run this on the Jetson device.")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[trt] Parse error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] FP16 mode enabled")

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("[trt] ERROR: Engine build failed")
        return False

    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"[trt] Engine saved to {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export ONNX to TensorRT")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output .trt engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    args = parser.parse_args()
    success = export_tensorrt(args.onnx, args.output, args.fp16)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

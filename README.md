# YOLOv8 Quantization Demo

[yolov8's official repository](https://github.com/ultralytics/ultralytics)

my demo is based on [v8.1.0](https://github.com/ultralytics/ultralytics/releases/tag/v8.1.0).

- [x] finish ptq and debug
- [x] finish qat
- [] add q/dq manually
- [x] sensitivity analysis
- [] skip some sensitive layers
- [] quant details... (fuse? amax?)
- [] some warnings/errors(may not affect the results) wait to fix (tracer, onnx export...)
- [] export format onnx to engine for tensorrt to inference

### Quick Start
```shell
python yolov8_flow.py --qat
```

### yolov8 quant flow

load model -> prepare calibration dataset -> (ptq) -> (sensitivity analysis) -> (qat) -> export model.onnx
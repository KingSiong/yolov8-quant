from ultralytics import YOLO
import argparse
import os
import sys
from pathlib import Path
import torch
from copy import deepcopy

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import quant_modules
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )

from utils import collect_stats, compute_amax

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator, DetectionTrainer
from ultralytics.utils.checks import check_imgsz, check_yaml
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device
from ultralytics.data import build_dataloader

from utils import calibrate_model, disable_quantization, enable_quantization, \
        save_model, export_onnx

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'yolov8n.yaml', help='model cfg path')
    parser.add_argument('--weight', type=str, default=ROOT / 'yolov8n.pt', help='model.pt path')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov8n')
    parser.add_argument('--epoch', type=int, default=1, help='train epoch num')
    parser.add_argument('--train-batch-size', type=int, default=32, help='train batch size')
    parser.add_argument('--val-batch-size', type=int, default=32, help='val batch size')
    parser.add_argument('--calib-batch-size', type=int, default=32, help='calib batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    parser.add_argument('--sensitivity', action='store_true', help='use sensitive analysis')
    # parser.add_argument('--sensitive-layer', default=['model.22.dfl.conv',
                                                    #   'model.2.cv1.conv'], help='skip sensitive layer: default detect head and second layer')
    parser.add_argument('--num-calib-batch', default=10, type=int, help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--out-dir', '-o', default=ROOT / 'runs/', help='output folder: default ./runs/')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')
    parser.add_argument('--qat', action='store_true', help='use QAT or not')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt

def load_model(cfg, weight, fuse=False) -> DetectionModel:
    '''
    load model from cfg / weight
    do not use cfg when loading a model with non-dynamic amax
    '''
    if weight:
        weight, ckpt = attempt_load_one_weight(weight=weight)
    else:
        weight = None

    if cfg:
        model = DetectionModel(cfg=cfg)
        if weight:
            model.load(weights=weight)
    else:
        model = weight

    if fuse:
        with torch.no_grad():
            model.fuse()

    return model

def get_model(calibrator, weight, cfg):

    if calibrator:
        # use calibrator to initialze model
        # use monkey patch to add fake-quant ops
        quant_desc_input = QuantDescriptor(calib_method=calibrator)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    
        quant_modules.initialize()
    
    model = load_model(cfg, weight)
    
    if calibrator:
        quant_modules.deactivate()
    
    model.eval()
    model.cuda()

    return model

def get_dataloader(opt, model, batch_size, key='train', prefix='calib: ', shuffle=False):
    '''
    get dataloader with yolov8's api
    '''
    data_dict = check_det_dataset(opt.data)
    data_path = data_dict[key]

    gs = max(int(model.stride.max() if hasattr(model, 'stride') else 32), 32)  # grid size (max stride)
    imgsz = check_imgsz(opt.imgsz, stride=gs, floor=gs, max_dim=1)
    
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    args = get_cfg(cfg=DEFAULT_CFG, overrides=None)
    
    dataset = YOLODataset(img_path=data_path,
                          imgsz=imgsz,
                          batch_size=batch_size,
                          augment=False,
                          hyp=args,
                          rect=True,
                          cache=opt.cache,
                          stride=gs,
                          pad=0.5,
                          prefix=colorstr(prefix))

    data_loader = build_dataloader(dataset, batch_size, opt.workers, shuffle=shuffle, rank=-1) # TODO: bug exists when shuffle is true

    return data_loader

def train(model, opt):
    args = dict(mode='train', 
                model=model, 
                data=opt.data, 
                imgsz=opt.imgsz, 
                batch=opt.train_batch_size, 
                epochs=opt.epoch, 
                project=opt.out_dir, 
                amp=False)
    trainer = DetectionTrainer(overrides=args)
    trainer.train()
    # get metrics
    metrics = trainer.metrics
    map50 = metrics['metrics/mAP50(B)']    # map50
    map = metrics['metrics/mAP50-95(B)'] # map50-95
    print(f'[INFO] after QAT, mAP50: {map50}  mAP50-95: {map}')
    model = load_model(None, trainer.best)

def evaluate_accuracy(model, opt, batch_size):
    model_copy = deepcopy(model)
    args = dict(mode='val', model=model_copy, data=opt.data, imgsz=opt.imgsz, batch=batch_size, project=opt.out_dir)
    validator = DetectionValidator(args=args)
    validator()
    metrics = validator.metrics
    map = metrics.box.map    # map50-95
    map50 = metrics.box.map50  # map50
    map75 = metrics.box.map75  # map75
    # maps = metrics.box.maps 
    
    return map, map50, map75

def build_sensitivity_profile(model, opt, batch_size):
    '''
    from https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/torchvision/classification_flow.py#L422
    evaluate every quant layer independently
    the higher score, the more sensitive
    TODO: when to skip the layers? before or after? how?
    '''
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    for i, quant_layer in enumerate(quant_layer_names):
        print("Enable", quant_layer)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")
        with torch.no_grad():
            evaluate_accuracy(model, opt, batch_size)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")

def main(opt):
    print('[INFO] Loading model...')
    device = select_device(opt.device, opt.train_batch_size)
    model = get_model(calibrator=opt.calibrator, cfg=opt.cfg, weight=opt.weight)
    calib_loader = get_dataloader(opt=opt, model=model, batch_size=opt.calib_batch_size)

    with torch.no_grad():
        with disable_quantization(model):
            map, map50, map75 = evaluate_accuracy(model, opt, batch_size=opt.val_batch_size)
            print(f'[INFO] before quantization, mAP50: {map50}  mAP50-95: {map}')

    if opt.num_calib_batch > 0: 
        # PTQ is activated when num-calib-batch > 0
        with torch.no_grad():
            print('[INFO] PTQ starting...')
            calibrate_model(
                model=model,
                model_name=opt.model_name,
                data_loader=calib_loader,
                num_calib_batch=opt.num_calib_batch,
                calibrator=opt.calibrator,
                hist_percentile=opt.percentile,
                out_dir=opt.out_dir,
                device=device)
        
            map, map50, map75 = evaluate_accuracy(model, opt, batch_size=opt.val_batch_size)
            print(f'[INFO] after PTQ, mAP50: {map50}  mAP50-95: {map}')

            # save model with yolov8's api
            save_model(model, opt.out_dir / f'{opt.model_name}-max-{opt.num_calib_batch * calib_loader.batch_size}.pt')

    # sensitive analysis
    if opt.sensitivity:
        build_sensitivity_profile(model, opt, opt.val_batch_size)

    if opt.qat:
        print('[INFO] QAT starting...')
        train(model, opt)

    onnx_filename = opt.out_dir / (opt.model_name + ".onnx")
    export_onnx(model, onnx_filename, device)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


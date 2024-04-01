from ultralytics import YOLO
import argparse
import os
import sys
from pathlib import Path
import torch

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
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator, DetectionTrainer
from ultralytics.utils.checks import check_imgsz, check_yaml
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device
from ultralytics.data import build_dataloader

from utils import calibrate_model, disable_quantization

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov8s')
    parser.add_argument('--epoch', type=int, default=1, help='train epoch num')
    parser.add_argument('--train-batch-size', type=int, default=64, help='train batch size')
    parser.add_argument('--val-batch-size', type=int, default=64, help='val batch size')
    parser.add_argument('--calib-batch-size', type=int, default=64, help='calib batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    # setting for calibration
    parser.add_argument('--sensitive-layer', default=['model.22.dfl.conv',
                                                      'model.2.cv1.conv'], help='skip sensitive layer: default detect head and second layer')
    parser.add_argument('--num-calib-batch', default=4, type=int, help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
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

def load_model(weight, device) -> DetectionModel:
    model = torch.load(weight, map_location=device)['model']
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()
    return model

def prepare_model(calibrator, opt, device):
    data_dict = check_det_dataset(opt.data)
    calib_path = data_dict['train']
    
    quant_desc_input = QuantDescriptor(calib_method=calibrator)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_modules.initialize()
    model = load_model(opt.weights, device)
    quant_modules.deactivate()
    model.eval()
    model.cuda()
    
    # Check imgsz
    gs = max(int(model.stride.max() if hasattr(model, 'stride') else 32), 32)  # grid size (max stride)
    imgsz = check_imgsz(opt.imgsz, stride=gs, floor=gs, max_dim=1)
    
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    args = get_cfg(cfg=DEFAULT_CFG, overrides=None)
    
    calib_dataset = YOLODataset(img_path=calib_path,
                          imgsz=imgsz,
                          batch_size=opt.calib_batch_size,
                          augment=False,
                          hyp=args,
                          rect=True,
                          cache=opt.cache,
                          stride=gs,
                          pad=0.5,
                          prefix=colorstr('calib: '))

    calib_loader = build_dataloader(calib_dataset, opt.calib_batch_size, opt.workers, shuffle=False, rank=-1) # bug exists when shuffle is true

    return model, calib_loader

def train(model, opt):
    args = dict(mode='train', 
                model=model, 
                data=opt.data, 
                imgsz=opt.imgsz, 
                batch=opt.train_batch_size, 
                epochs=opt.epoch, 
                project=opt.out_dir)
    trainer = DetectionTrainer(overrides=args)
    trainer.train()

def evaluate_accuracy(model, opt):
    args = dict(mode='val', model=model, data=opt.data, imgsz=opt.imgsz, batch=opt.val_batch_size, project=opt.out_dir)
    validator = DetectionValidator(args=args)
    validator()
    metrics = validator.metrics
    map = metrics.box.map    # map50-95
    map50 = metrics.box.map50  # map50
    map75 = metrics.box.map75  # map75
    # maps = metrics.box.maps 
    
    return map, map50, map75

def main(opt):
    print('[INFO] Loading model...')
    device = select_device(opt.device, opt.train_batch_size)
    model, calib_loader = prepare_model(calibrator=opt.calibrator, opt=opt, device=device)

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
        
    if opt.num_calib_batch > 0: 
        with torch.no_grad():
            map, map50, map75 = evaluate_accuracy(model, opt)
            print(f'[INFO] after PTQ, mAP50: {map50}  mAP50-95: {map}')

    # TODO: skip some layers which is sensitive
    # ...

    if opt.qat:
        print('[INFO] QAT starting...')
        train(model, opt)
        with torch.no_grad():
            map, map50, map75 = evaluate_accuracy(model, opt)
            print(f'[INFO] after QAT, mAP50: {map50}  mAP50-95: {map}')
    
    with torch.no_grad():
        with disable_quantization(model):
            map, map50, map75 = evaluate_accuracy(model, opt)
            print(f'[INFO] before quantization, mAP50: {map50}  mAP50-95: {map}')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


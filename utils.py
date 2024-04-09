import torch
import os
from tqdm import tqdm
from copy import deepcopy
from ultralytics.utils.torch_utils import de_parallel

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

def collect_stats(model, data_loader, num_calib_batch, device):
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, batch in tqdm(enumerate(data_loader), total=num_calib_batch):
        image = batch['img']
        image = image.to(device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        model(image)
        if i >= num_calib_batch:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f'{name:40}: {module}')
        model.cuda()

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate
        Arguments:
            model: detection model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model:")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            # calib_output = os.path.join(out_dir, F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            # ckpt = {'model': model}
            # torch.save(ckpt, calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                # calib_output = os.path.join(out_dir, F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                # torch.save(model.state_dict(), calib_output)
                # ckpt = {'model': model}
                # torch.save(ckpt, calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                # calib_output = os.path.join(out_dir, F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                # torch.save(model.state_dict(), calib_output)
                # ckpt = {'model': model}
                # torch.save(ckpt, calib_output)

class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)

class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
            
    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)

def save_model(model, path):
    ckpt = {
        "model": deepcopy(de_parallel(model)),
    }

    torch.save(ckpt, path)
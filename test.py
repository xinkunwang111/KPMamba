import warnings
warnings.filterwarnings("ignore")
import argparse
import mmcv
import os
import os.path as osp
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from escape.datasets import build_dataset
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset, e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use GPU to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu-ids', default='0', help='CUDA device to use, e.g., 1,2')  # 添加设备选择参数
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def merge_configs(cfg1, cfg2):
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

def print_memory_usage(device):
    """Prints the current GPU memory usage."""
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"Memory allocated on {device}: {allocated:.2f} MB")
    print(f"Memory reserved on {device}: {reserved:.2f} MB")

def main():
    args = parse_args()

    # 设置 CUDA 设备
    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('episodes_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False,
        drop_last=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # 将模型加载到指定的设备上
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        # 强制 MMDataParallel 使用指定的设备
        model = MMDataParallel(model, device_ids=[int(args.gpu_ids)])
        outputs = single_gpu_test(model, data_loader)
        print_memory_usage(device)  # 打印显存使用情况
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[int(args.gpu_ids.split(':')[-1])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
        print_memory_usage(device)  # 打印显存使用情况

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        results = dataset.evaluate(outputs, args.work_dir, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')

if __name__ == '__main__':
    main()

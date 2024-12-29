log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10, metric='PCK', key_indicator='PCK', gpu_collect=True)
optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 50])
total_epochs = 60
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=1093,
    dataset_joints=1,
    dataset_channel=[
        [
            0,
        ],
    ],
    inference_channel=[
        0,
    ],
    max_kpt_num=68)

# model settings
"""
#记录这里的使用条件：
#在这里prototype_detector.py中，我们自定义了这个类别， 因此可以使用框架来自动initializse 这个class
#在这个model 的dict中， 所有其他的参数都是为了initializse这个class 而存在的， 例如backbone, keypoint_head, keypoint_adaptation
同样的，在backbone 这里面， 我们同样也是在dict里面使用resnet，因为resnet在mmpose中有，所以可以直接使用。同时dict里面也定义了depth，这个可以在github上看
mmpose的定义， 只有depth是没有default的
backbone:  https://github.com/open-mmlab/mmpose/tree/main/mmpose/models/backbones
"""

model = dict(
    type='PrototypeDetector',
    #pretrained='torchvision://vgg16',
    #pretrained = 'https://download.pytorch.org/models/swin_t-704ceda3.pth',
    #pretrained = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    #pretrained = 'https://huggingface.co/hustvl/Vim-base-midclstok/resolve/main/vim_b_midclstok_81p9acc.pth?download=true',
    #pretrained = 'https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth',
    #backbone=dict(type='ResNet', depth=50),#输入1,3,256,256。 输出是（1,2048,8,8）
    
    #  backbone=dict(
    #      type='SwinTransformer',
    #      embed_dims=256,
    #      num_heads = [4, 8, 16, 16],
    #      patch_size=4,
    #      window_size=7,
    #      ),
    #backbone=dict(
        #type='VGG',
        #depth=16,),
    # backbone=dict(
    #     type = 'VisionMamba',
    #     img_size = 256,
    #     patch_size = 4,
    #     stride = 4,
    #     embed_dim=384,
    #     if_abs_pos_embed = True,
    #     if_bidirectional = True,
        
    # ), #输入Batch,3,256,256,输出 Batch，embed_dim ，16，16

    backbone = dict(
        type = 'VMUNet',
        
        depths=[2,2,2,2], 
        depths_decoder = [2,2,2,1],
        
        drop_path_rate = 0.2, 
        abs_position = True,
        patch_size = 4,
        dims=[96, 192, 384, 768], 
        dims_decoder=[768, 384, 192, 96],
        load_ckpt_path = '/mnt/data1/lv0/scratch/home/v_xinkun_wang/pose_estimation/escape-tgt/work_dirs/baseline_split1/vmunet/vmamba_pretrained_weight/vssmsmall_dp03_ckpt_epoch_238.pth',
        

    ),
    num_layers = 12,
    # backbone = dict(
    #     type = 'VMUNet',
        
    #     vmunet_depths=[2,2,2,2], 
    #     vmunet_depths_decoder = [2,2,2,1],
        
    #     vmunet_drop_path_rate = 0.2, 
    #     vmunet_abs_position = True,
    #     vmunet_patch_size = 4,
    #     vmunet_dims=[96, 192, 384, 768], 
    #     vmunet_dims_decoder=[768, 384, 192, 96],
    #     vmunet_load_ckpt_path = '/mnt/data1/lv0/scratch/home/v_xinkun_wang/pose_estimation/escape-tgt/work_dirs/baseline_split1/vmunet/vmamba_pretrained_weight/vssmsmall_dp03_ckpt_epoch_238.pth',
        

    # ),





    deconv=dict(type='CustomDeconv', in_channels=96,num_deconv_layers = 1),# deconv 扩大8倍，变成（256,64,64）
    keypoint_head=dict(
        type='GlobalHead',
        keypoints_num=channel_cfg['num_output_channels'],
        out_channels=256,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    keypoint_adaptation=dict(
        type='MLEHead',
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        pooling_kernel=15,
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11,
        fewshot_testing=True))


data_cfg = dict(
    image_size=[256 ,256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img','target', 'target_weight',],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id'
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight',],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'category_id'
        ]),
]



test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=12,
    episodes_per_gpu=1,
    train=dict(
        type='GlobalDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=train_pipeline),
    val=dict(
        type='FewshotDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
    test=dict(
        type='FewshotDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots=1,
        num_queries=15,
        num_episodes=200,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
    extract_features=dict(
        type='GlobalDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
)

shuffle_cfg = dict(interval=1)

import torch.nn as nn
import numpy as np

X2CT = {
    'global': {
        'img_size': 128,
        'batch_size': 2,
    },
    # learning rate
    'optimizer': {
        'gen_lr': 1e-3,
        'betas': (0, 0.9),
    },
    # loss hyperparameter
    'process': {
        'class': 'Pre3DProcess',
        'kwargs': {
            'CT_edge_lambda': 0.05,
            'nerf_lambda': 0.001,
        }
    },
    # network parameters
    'generator': {
        'class': 'Renderer',
        'kwargs': {
            'step_length_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/step_length.npy',
            'idxs_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/idxs.npy',
            'target_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/target.npy',
            'source_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/source.npy',
            'coor_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/coords_3D.npy',
            'mean_path': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/mean_CT.npy',
            'drr_height': 128,
            'drr_weight': None,
            'device': 'cuda',
            'representation_kwargs': {
                'hidden_dim': 34,
                'norm_layer': nn.BatchNorm3d,
                'norm_layer2d': nn.BatchNorm2d,
                'input_dim': 2,
                'input_coor_dim': 1,
                'depths': [3, 3, 9, 3],
                'spacing': np.array([2.5, 2.5, 2.5]),
                'sdr': 949 // 2,
                'del_size': 5,
                'offset_points': 2
            },
        }
    },
    # path for CT and X-ray images
    'dataset': {
        'class': 'X2CT',
        'kwargs': {
            'CT_PATH': '/home/anonymous/code/AI/VolumeNeRF/crop_image',
            'XRAY_PATH': '/home/anonymous/code/AI/VolumeNeRF/diffdrr',
            'MEAN_XRAY_PATH': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/mean_xray.npy',
            'STD_XRAY_PATH': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/std_xray.npy'
        }
    },
    # test snapshot
    'snapshot': {
        'test_dir': '/home/anonymous/code/AI/VolumeNeRF/drr_Xray/LIDC-IDRI-0002.npy',
        'MEAN_XRAY_PATH': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/mean_xray.npy',
        'STD_XRAY_PATH': '/home/anonymous/code/AI/VolumeNeRF/DRR_Parameters/std_xray.npy'
    }
}

'''
Function:
    Implementation of APCNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseSegmentor
from .acm import AdaptiveContextModule
from ...backbones import BuildActivation, BuildNormalization


'''APCNet'''
class APCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(APCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build acm
        acm_cfg = {
            'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'pool_scale': None, 'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg),
        }
        self.acm_modules = nn.ModuleList()
        for pool_scale in head_cfg['pool_scales']:
            acm_cfg['pool_scale'] = pool_scale
            self.acm_modules.append(AdaptiveContextModule(**acm_cfg))
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['pool_scales']) + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to acm
        acm_outs = [backbone_outputs[-1]]
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(backbone_outputs[-1]))
        feats = torch.cat(acm_outs, dim=1)
        # feed to decoder
        predictions = self.decoder(feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.customizepredsandlosses(
                predictions=predictions, targets=targets, backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions
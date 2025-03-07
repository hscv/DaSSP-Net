import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from lib.models.vipt.domain_adapter import Jing_conv_task

from lib.models.layers.patch_embed import PatchEmbed, PatchEmbed_HSI
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt, Cross_Attention_for_HSI

_logger = logging.getLogger(__name__)

from lib.models.vipt.QRN3D import QRNNConv3D
from functools import partial

class FusionModule(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(FusionModule, self).__init__()
        self.conv0_q = nn.Linear(inplanes, hide_channel, bias=False)
        self.conv0_k = nn.Linear(inplanes, hide_channel, bias=False)
        self.conv0_v = nn.Linear(inplanes, hide_channel, bias=False)

        self.conv1_q = nn.Linear(inplanes, hide_channel, bias=False)
        self.conv1_k = nn.Linear(inplanes, hide_channel, bias=False)
        self.conv1_v = nn.Linear(inplanes, hide_channel, bias=False)

        self.conv1x1 = nn.Linear(hide_channel, inplanes, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rgb=None, hsi=None):
        """ Forward pass with input x. """
        B, C, W, H = rgb.shape 
        x0 = rgb
        x0 = x0.flatten(2)
        x0 = x0.transpose(1,2) 
        x1 = hsi 
        x1 = x1.flatten(2)
        x1 = x1.transpose(1,2) 

        x0_q = self.conv0_q(x0) 
        x0_k = self.conv0_k(x0) 
        x1_v = self.conv0_v(x1) 

        B, N, C = x0_q.size()
        scale = C ** -0.5
        attn = (x0_q @ x0_k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        res1 = (attn @ x1_v).transpose(1, 2).reshape(B, N, C)

        results = res1
        results = self.conv1x1(results) 
        results = results.transpose(1,2) 
        results = results.view(B, -1, W, H)
        return results

class Prompt_block_New(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block_New, self).__init__()
        self.fusionModuleArr = FusionModule(inplanes=inplanes, hide_channel=hide_channel, smooth=smooth) 
        self.proj3d = QRNNConv3D(768, 768, tau=128)

    def get_RNN_feature(self, feature_resArr):
        fea_arr = []
        for feature_res in feature_resArr:
            fea_arr.append(feature_res.unsqueeze(2))
        feature_res = torch.cat(fea_arr, dim=2)  
        feature_res_F, feature_res_F_all = self.proj3d(feature_res)
        return feature_res_F

    def forward(self, x, i = 0):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        rgb_fea = x[:, :768, :, :].contiguous()
        hsi_fea = x[:, 768:, :, :].contiguous()
        hsi_fea = hsi_fea.view(B, 768, -1, W, H)
        hsi_fea_arr = []
        for kk in range(hsi_fea.size()[2]):
            hsi_fea_arr.append(hsi_fea[:,:,kk,:,:])

        promptfea_arr = []
        for kk in range(len(hsi_fea_arr)):
            promptfea_arr.append(self.fusionModuleArr(rgb=rgb_fea, hsi=hsi_fea_arr[kk]))
            if kk == 0:
                res = promptfea_arr[-1]
            else:
                res = res + promptfea_arr[-1]
        if i == 0: 
            res = self.get_RNN_feature(promptfea_arr)
        else:
            res = promptfea_arr[-1]
        promptfea_arr = []
        promptfea_arr.append(res)
        return promptfea_arr, res



class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) # use LayerNorm
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        embed_layer_HSI = PatchEmbed_HSI
        self.patch_embed_prompt_hsi = embed_layer_HSI(
            img_size=img_size, patch_size=patch_size, in_chans=-1, embed_dim=embed_dim)


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            prompt_hsi_blocks_new = []
            block_nums = depth if self.prompt_type == 'vipt_deep' else 1
            for i in range(block_nums):
                prompt_hsi_blocks_new.append(Prompt_block_New(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_hsi_blocks_new = nn.Sequential(*prompt_hsi_blocks_new)

            prompt_hsi_norms_new = []
            for i in range(block_nums):
                prompt_hsi_norms_new.append(Jing_conv_task(embed_dim, stride=1, nb_tasks=3, is_proj=1, second=0) )
            self.prompt_hsi_norms_new = nn.Sequential(*prompt_hsi_norms_new)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features_hsi(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        print ('x_rgb.size() = ', x_rgb.size(), ', z_rgb.size() = ', z_rgb.size())
        print ('x_dte.size() = ', x_dte.size(), ', z_dte.size() = ', z_dte.size())
        if x_dte.size()[1] == 16:
            norm_mode = 0
        elif x_dte.size()[1] == 25:
            norm_mode = 1
        elif x_dte.size()[1] == 15:
            norm_mode = 2
        else:
            raise Exception
        # overwrite x & z
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        z_dte_arr = self.patch_embed_prompt_hsi(z_dte)
        x_dte_arr = self.patch_embed_prompt_hsi(x_dte)

        '''input prompt: by adding to rgb tokens'''
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            z_feat = token2feature(self.prompt_hsi_norms_new[0](z, norm_mode))
            x_feat = token2feature(self.prompt_hsi_norms_new[0](x, norm_mode))
            z_dte_feat = [z_feat]
            x_dte_feat = [x_feat]
            for kk in range(len(z_dte_arr)):
                z_dte_feat.append(token2feature(self.prompt_hsi_norms_new[0](z_dte_arr[kk], norm_mode)))
                x_dte_feat.append(token2feature(self.prompt_hsi_norms_new[0](x_dte_arr[kk], norm_mode)))
            z_feat = torch.cat(z_dte_feat, dim=1)
            x_feat = torch.cat(x_dte_feat, dim=1)
            

            prompt_z_arr, z_feat = self.prompt_hsi_blocks_new[0](z_feat)
            prompt_x_arr, x_feat = self.prompt_hsi_blocks_new[0](x_feat)
            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            z = z + z_dte
            x = x + x_dte
        else:
            z = z + z_dte
            x = x + x_dte

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['vipt_deep']:
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_hsi_norms_new[i - 1](x, norm_mode)  # todo
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]
                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)

                    z_dte_feat = [z_feat]
                    x_dte_feat = [x_feat]
                    for kk in range(len(prompt_z_arr)):
                        z_dte_feat.append(token2feature(self.prompt_hsi_norms_new[i](feature2token(prompt_z_arr[kk]), norm_mode)))
                        x_dte_feat.append(token2feature(self.prompt_hsi_norms_new[i](feature2token(prompt_x_arr[kk]), norm_mode)))
                    z_feat = torch.cat(z_dte_feat, dim=1)
                    x_feat = torch.cat(x_dte_feat, dim=1)

                    prompt_z_arr, z_feat = self.prompt_hsi_blocks_new[i](z_feat, i)
                    prompt_x_arr, x_feat = self.prompt_hsi_blocks_new[i](x_feat, i)

                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    z_prompted, x_prompted = z, x

                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict


    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False, train_data_type=""):
        
        x, aux_dict = self.forward_features_hsi(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt_all(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

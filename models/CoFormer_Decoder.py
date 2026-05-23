import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from einops import rearrange
import fvcore.nn.weight_init as weight_init
from .SPMamba import SPMamba

class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D convolution implementation """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])
        
        dropout = 0.1
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        num_heads = in_channels if in_channels < 8 else 8
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=num_heads, dropout=dropout
        )
        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def prune(self, ct):
        bsz, ch, ha, wa, n = ct.size()
        ct = rearrange(ct, 'bsz ch ha wa n -> bsz (ch ha wa) n')
        ct = F.interpolate(ct, scale_factor=1/self.stride[-1], mode='linear', align_corners=False)
        ct_pruned = rearrange(ct, 'bsz (ch ha wa) n -> bsz ch ha wa n', ch=ch, ha=ha, wa=wa)
        return ct_pruned

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        bsz, inch, ha, wa, n = x.size()
        
        out1 = x.permute(0, 4, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, n, outch, o_ha, o_wa).permute(0, 2, 3, 4, 1).contiguous()

        if self.stride[2:][-1] > 1:
            out1 = self.prune(out1)
            
        bsz, inch, ha, wa, n = out1.size()
        out2 = out1.permute(0, 2, 3, 1, 4).contiguous().view(-1, inch, n).permute(2, 0, 1)
        out2_ = self.norm(out2)
        q = k = out2_
        out2_ = self.self_attn(q, k, out2_)[0]
        out2 = out2 + self.dropout(out2_)
        out2 = out2.permute(1, 2, 0)
        out2 = rearrange(out2, '(bsz ha wa) inch n -> bsz inch ha wa n', bsz=bsz, ha=ha, wa=wa)

        return out2


class Correlation:
    r""" Provides functions that builds/manipulates correlation tensors """
    @classmethod
    def multilayer_correlation(cls, query_feat, support_feats):
        eps = 1e-5
        corrs = []
        for support_feat in support_feats:
            bsz, ch, hb, wb = support_feat.size()
            support_feat = rearrange(support_feat, "b c h w -> b h w c").contiguous()
            support_feat = support_feat.view(-1, ch)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat_ = rearrange(query_feat, "b c h w -> b h w c").contiguous()
            query_feat_ = query_feat_.view(-1, ch)
            query_feat_ = query_feat_ / (query_feat_.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat_.unsqueeze(0), support_feat.transpose(0, 1).unsqueeze(0)).squeeze()
            corr = corr.view(bsz, ha, wa, bsz, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        fina_corr = torch.stack(corrs).transpose(0, 1).contiguous()
        return fina_corr

class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                in_ch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4
                building_block_layers.append(CenterPivotConv4d(in_ch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 64, 128, 256
        self.enc_layer = make_building_block(inch, [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])

    def forward(self, hypercorr_pyramid):
        hypercorr_sqz = self.enc_layer(hypercorr_pyramid)
        hypercorr_encoded = hypercorr_sqz.mean(dim=-1)
        return hypercorr_encoded


class Decoder_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, start=False):
        super(Decoder_Conv, self).__init__()

        self.lateral_conv = (
            nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
            )
            if not start else None
        )

        if not start:
            in_channel = out_channel

        self.output_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:
            cur_fpn = self.lateral_conv(enc_fea)
            dec_fea = cur_fpn + F.interpolate(
                dec_fea,
                size=cur_fpn.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            dec_fea = self.output_conv(dec_fea)
        else:
            dec_fea = self.output_conv(enc_fea)
        return dec_fea


class Prototype_Refinement(nn.Module):
    def __init__(self, channel):
        super(Prototype_Refinement, self).__init__()

        self.co_bg_pred_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, bias=True),
        )
        self.stage_pred_head = nn.Conv2d(channel, 3, kernel_size=1, bias=True)
        self.hpn_learner = HPNLearner(inch=2)

    def _normalize_mask(self, mask):
        b, _, h, w = mask.shape
        mask = torch.sigmoid(mask)
        mask_flat = mask.view(b, 1, -1)
        min_pred = torch.min(mask_flat, dim=2, keepdim=True)[0]
        max_pred = torch.max(mask_flat, dim=2, keepdim=True)[0]
        norm_pred = (mask_flat - min_pred) / (max_pred - min_pred + 1e-8)
        norm_pred = norm_pred.view(b, 1, h, w)
        return norm_pred

    def _weighted_gap(self, fea, mask):
        b, c, h, w = fea.shape
        fea = fea * mask
        denom = torch.sum(mask.view(b, 1, -1), dim=2, keepdim=True) + 1e-8
        ratio = (h * w) / denom
        vec = ratio * torch.mean(fea.view(b, c, -1), dim=2, keepdim=True)
        vec = vec.view(b, c, 1, 1)
        return vec

    def forward(self, fea):
        stage_logits = self.stage_pred_head(fea)
        co_bg_logits = self.co_bg_pred_head(fea)
        co_logit, bg_logit = co_bg_logits.split(1, dim=1)
        co_mask = self._normalize_mask(co_logit)
        bg_mask = self._normalize_mask(bg_logit)
        co_proto = self._weighted_gap(fea, co_mask)
        bg_proto = self._weighted_gap(fea, bg_mask)
        
        ds = False
        if fea.shape[2] >= 128:
            scale = 64 / fea.shape[2]
            fea_ = F.interpolate(fea, scale_factor=scale, mode="bilinear")
            ds = True
        else:
            fea_ = fea
            
        HyperCalFea = Correlation.multilayer_correlation(fea_, [co_proto, bg_proto])
        fina_fea = self.hpn_learner(HyperCalFea)
        
        if ds:
            fina_fea = F.interpolate(fina_fea, scale_factor=1/scale, mode="bilinear") + fea
            
        return {
                   "co_pred": stage_logits[:, 0:1, ...],
                   "bg_pred": stage_logits[:, 1:2, ...],
                   "com_pred": stage_logits[:, 2:3, ...],
                   "tgfr_mid_pred": stage_logits[:, 0:1, ...],
               }, fina_fea


class CoFormer_Decoder(nn.Module):
    def __init__(self, cfg):
        super(CoFormer_Decoder, self).__init__()

        enc_name = cfg.MODEL.ENCODER.NAME
        ga_name = cfg.MODEL.GROUP_ATTENTION.NAME
        self.fea_names = enc_name + ga_name 

        enc_channel = cfg.MODEL.ENCODER.CHANNEL
        ga_channel = cfg.MODEL.GROUP_ATTENTION.CHANNEL
        fea_channels = enc_channel + ga_channel

        hidden_dim = cfg.MODEL.COFORMER_DECODER.HIDDEN_DIM
        drop_path = cfg.MODEL.COFORMER_DECODER.DROP_PATH
        drop_path_ratios = torch.linspace(0, drop_path, len(fea_channels) - 1)

        self.input_proj = nn.Conv2d(in_channels=fea_channels[-1], out_channels=hidden_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)

        for idx, channel in enumerate(fea_channels):
            dropout = drop_path_ratios[-idx]
            dec_conv = Decoder_Conv(
                in_channel=channel,
                out_channel=hidden_dim,
                start=idx == len(fea_channels) - 1,
            )
            self.add_module("decoder_{}".format(idx + 1), dec_conv)

            if idx != 0 and idx != len(fea_channels) - 1:
                group_att = SPMamba(cfg, hidden_dim)
                self.add_module("group_att_{}".format(idx + 1), group_att)

            if idx != 0:
                proto_refine = Prototype_Refinement(hidden_dim)
                self.add_module("proto_refine_{}".format(idx + 1), proto_refine)

        self.final_head = Prototype_Refinement(hidden_dim)

    def forward(self, features: Dict):
        stage_co_preds = []
        stage_bg_preds = []
        stage_com_preds = []
        stage_tgfr_mid_preds = []

        dec_fea = None
        fea_nums = len(self.fea_names)
        
        for idx, fea_name in enumerate(self.fea_names[::-1]):
            enc_fea = features[fea_name]
            decoder_layer = getattr(self, "decoder_{}".format(fea_nums - idx))
            dec_fea = decoder_layer(
                enc_fea=enc_fea,
                dec_fea=dec_fea
            )

            if idx != fea_nums - 1 and idx != 0:
                group_att = getattr(self, "group_att_{}".format(fea_nums - idx))
                if dec_fea.shape[2] > 16:  
                    dec_fea = group_att(dec_fea, ds=True, scale=16. / dec_fea.shape[2])
                else:
                    dec_fea = group_att(dec_fea)

            if idx != fea_nums - 1 and idx != fea_nums - 2:
                proto_refine_layer = getattr(self, "proto_refine_{}".format(fea_nums - idx))
                preds, dec_fea = proto_refine_layer(dec_fea)
                stage_co_preds.append(preds["co_pred"])
                stage_bg_preds.append(preds["bg_pred"])
                stage_com_preds.append(preds["com_pred"])
                stage_tgfr_mid_preds.append(preds["tgfr_mid_pred"])

        final_preds, _ = self.final_head(dec_fea)
        
        return {
            "stage_co_preds": stage_co_preds,
            "stage_bg_preds": stage_bg_preds,
            "stage_com_preds": stage_com_preds,
            "stage_tgfr_mid_preds": stage_tgfr_mid_preds,
            "co_pred": final_preds["co_pred"],
            "bg_pred": final_preds["bg_pred"],
            "com_pred": final_preds["com_pred"]
        }
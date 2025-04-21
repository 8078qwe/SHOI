# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, SingleStageDetector

import torch
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def triplet2Result(triplets, use_mask, eval_pan_rels=True):
    if use_mask:
        bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
            r_labels, r_dists, pan_masks, rels, pan_labels \
            = triplets
        if isinstance(bboxes, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            rel_pairs = rel_pairs.detach().cpu().numpy()
            complete_r_labels = complete_r_labels.detach().cpu().numpy()
            complete_r_dists = complete_r_dists.detach().cpu().numpy()
            r_labels = r_labels.detach().cpu().numpy()
            r_dists = r_dists.detach().cpu().numpy()
        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.detach().cpu().numpy()
            pan_rel_pairs = pan_rel_pairs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            pan_masks = pan_masks.detach().cpu().numpy()
            rels = rels.detach().cpu().numpy()
            pan_labels = pan_labels.detach().cpu().numpy()
        if eval_pan_rels:
            return Result(refine_bboxes=bboxes,
                          labels=pan_labels + 1,
                          formatted_masks=dict(pan_results=pan_seg),
                          rel_pair_idxes=pan_rel_pairs,  # elif not pan: rel_pairs,
                          rel_dists=r_dists,
                          rel_labels=r_labels,
                          pan_results=pan_seg,
                          masks=pan_masks,
                          rels=rels)
        else:
            return Result(refine_bboxes=bboxes,
                          labels=labels,
                          formatted_masks=dict(pan_results=pan_seg),
                          rel_pair_idxes=rel_pairs,
                          rel_dists=complete_r_dists,
                          rel_labels=complete_r_labels,
                          pan_results=pan_seg,
                          masks=masks)
    else:
        bboxes, labels, rel_pairs, r_labels, r_dists = triplets
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        rel_pairs = rel_pairs.detach().cpu().numpy()
        r_labels = r_labels.detach().cpu().numpy()
        r_dists = r_dists.detach().cpu().numpy()
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=None),
            rel_pair_idxes=rel_pairs,
            rel_dists=r_dists,
            rel_labels=r_labels,
            pan_results=None,
        )


@DETECTORS.register_module()
class HOITrans(SingleStageDetector):
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(HOITrans, self).__init__(backbone, None, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        if self.bbox_head.use_mask:
            BS, C, H, W = img.shape
            new_gt_masks = []
            for each in gt_masks:
                mask = torch.tensor(each.to_ndarray(), device=x[0].device)
                _, h, w = mask.shape
                padding = (0, W - w, 0, H - h)
                mask = F.interpolate(F.pad(mask, padding).unsqueeze(1),
                                     size=(H // 2, W // 2),
                                     mode='nearest').squeeze(1)
                # mask = F.pad(mask, padding)
                new_gt_masks.append(mask)

            gt_masks = new_gt_masks

        losses = self.bbox_head.forward_train(x, img_metas, gt_rels, gt_bboxes,
                                              gt_labels, gt_masks,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat,
                                                  img_metas,
                                                  rescale=rescale)
        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)
        return sg_results


@TRANSFORMER.register_module()
class Transformer_HOITrans(BaseModule):
    def __init__(self,
                 encoder=None,
                 decoder1=None,
                 decoder2=None,
                 init_cfg=None):
        super(Transformer_HOITrans, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder1 = build_transformer_layer_sequence(decoder1)
        self.decoder2 = build_transformer_layer_sequence(decoder2)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoders,
                with shape [bs, h, w].
            query1_embed (Tensor): The first query embedding for decoder, with
                shape [num_query, c].
            query2_embed (Tensor): The second query embedding for decoder, with
                shape [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoders, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)

        num_queries = query_embed_h.shape[0]

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        ins_tgt = torch.zeros_like(ins_query_embed)
        memory = self.encoder(query=x,
                              key=None,
                              value=None,
                              query_pos=pos_embed,
                              query_key_padding_mask=mask)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        """target1 = torch.zeros_like(query1_embed)
        target2 = torch.zeros_like(query2_embed)"""
        # out_dec: [num_layers, num_query, bs, dim]
        # first decoder
        ins_hs = self.decoder1(query=ins_tgt,
                               key=memory,
                               value=memory,
                               key_pos=pos_embed,
                               query_pos=ins_query_embed,
                               key_padding_mask=mask)
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        # add
        ins_guided_embed = (h_hs + o_hs) / 2.0
        ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)

        inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # second decoder
        inter_hs = self.decoder2(query=inter_tgt,
                                 key=memory,
                                 value=memory,
                                 key_pos=pos_embed,
                                 query_pos=ins_guided_embed[0],
                                 key_padding_mask=mask)
        inter_hs = inter_hs.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return h_hs, o_hs, inter_hs, memory



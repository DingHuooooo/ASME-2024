import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .modules.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, PromptEncoder
from .modules.addiLayer import AttentionLayer, UNet

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple

scaler = torch.cuda.amp.GradScaler()


class PositionEmbeddingRandom(nn.Module):

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

class SAM_Base(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=(inp_size//encoder_mode['patch_size'], inp_size//encoder_mode['patch_size']),
            input_image_size=(inp_size, inp_size),
            mask_in_chans=16,
        )
        self.loss_mode = loss

        if self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']

    def set_input(self, image, gt_mask):
        self.input_image = image.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward(self):
        with torch.cuda.amp.autocast():
            self.loss_all = self.criterionBCE(self.pred_mask, self.gt_mask)
            if self.loss_mode == 'iou':
                self.loss_IoU = self.criterionIOU(self.pred_mask, self.gt_mask)
                self.loss_all += self.loss_IoU

        scaler.scale(self.loss_all).backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward()  # calculate graidents for G

        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

        scaler.step(self.optimizer)  # update G's weights
        scaler.update()
        # delete tensors to free memory
        del self.features, self.input_image, self.low_res_masks, self.iou_predictions, self.sparse_embeddings, self.dense_embeddings
        torch.cuda.empty_cache()  # clear CUDA cache


    def forward(self, x):
        raise NotImplementedError


@register('sam_cnn')
class SAM_CNN(SAM_Base):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__(inp_size, encoder_mode, loss)
        self.mask_generator = UNet(in_channels=3, out_channels=1)
        

    def forward(self):
        with torch.cuda.amp.autocast():
            # Embed prompts (with mask)
            self.mask_generator_out = self.mask_generator(self.input_image)
            self.mask_generator_out = self.mask_generator_out.mean(dim=1, keepdim=True)
            self.sparse_embeddings, self.dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=self.mask_generator_out
                )
            
            self.features = self.image_encoder(self.input_image)

            # Predict masks
            self.low_res_masks, self.iou_predictions = self.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=self.sparse_embeddings,
                dense_prompt_embeddings=self.dense_embeddings,
                multimask_output=False,
            )

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(self.low_res_masks, self.inp_size, self.inp_size)
            self.pred_mask = masks

    def infer(self, input_image):
        # Embed prompts (with mask)
        self.mask_generator_out = self.mask_generator(input_image)
        self.mask_generator_out = self.mask_generator_out.mean(dim=1, keepdim=True)
        self.sparse_embeddings, self.dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=self.mask_generator_out
            )
        
        self.features = self.image_encoder(input_image)

        # Predict masks
        self.low_res_masks, self.iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_embeddings,
            dense_prompt_embeddings=self.dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(self.low_res_masks, self.inp_size, self.inp_size)

        return masks
                    
@register('sam_adaptor')
class SAM_ADAPTER(SAM_Base):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__(inp_size, encoder_mode, loss)

    def forward(self):
        with torch.cuda.amp.autocast():
            # Embed prompts (no mask)
            self.sparse_embeddings, self.dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            self.features = self.image_encoder(self.input_image)

            # Predict masks
            self.low_res_masks, self.iou_predictions = self.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=self.sparse_embeddings,
                dense_prompt_embeddings=self.dense_embeddings,
                multimask_output=False,
            )

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(self.low_res_masks, self.inp_size, self.inp_size)
            self.pred_mask = masks

    def infer(self, input_image):
        # Embed prompts
        self.sparse_embeddings, self.dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        self.features = self.image_encoder(input_image)

        # Predict masks
        self.low_res_masks, self.iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_embeddings,
            dense_prompt_embeddings=self.dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(self.low_res_masks, self.inp_size, self.inp_size)

        return masks
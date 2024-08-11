import torch
import torch.nn as nn
from typing import Optional, Union, List
from segmentation_models_pytorch_new.encoders import get_encoder
from segmentation_models_pytorch_new.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from suppixpool_layer import SupPixPool, SupPixUnpool
from segmentation_models_pytorch_new.unet.decoder import UnetDecoder
from segmentation_models_pytorch_new.base import initialization as init

class Unet_Superpix(nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None, ):
        super(Unet_Superpix, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.SupPixPool = SupPixPool()
        self.SupPixUnpool = SupPixUnpool()
        self.Conv4 = nn.Conv1d(decoder_channels[-1],classes,kernel_size=1)
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)
            if self.classification_head is not None:
                init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

            h, w = x.shape[-2:]
            output_stride = self.encoder.output_stride
            if h % output_stride != 0 or w % output_stride != 0:
                new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
                new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
                raise RuntimeError(
                    f"Wrong input shape height={h}, width={w}. Expected image height and width "
                    f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
                )

    def forward(self, x, SuperPix):
            """Sequentially pass `x` trough model`s encoder, decoder and heads"""
            input = x
            self.check_input_shape(x)

            features = self.encoder(x)
            decoder_output = self.decoder(*features)

            masks_1 = self.segmentation_head(decoder_output)
            SuperPool = self.SupPixPool(decoder_output,SuperPix)
            SuperPool_Conv = self.Conv4(SuperPool)
            masks_2 = self.SupPixUnpool(SuperPool_Conv,SuperPix)

            # masks = masks_1 + masks_2
            # masks = masks_2
            if self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_1, masks_2, labels

            return masks_1,masks_2

    @torch.no_grad()
    def predict(self, x):
            """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

            Args:
                x: 4D torch tensor with shape (batch_size, channels, height, width)

            Return:
                prediction: 4D torch tensor with shape (batch_size, classes, height, width)

            """
            if self.training:
                self.eval()

            x = self.forward(x)

            return x
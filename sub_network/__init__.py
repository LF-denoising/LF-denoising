import torch.nn as nn
from sub_network.SpatioTemporalTrans import SpatioTemporalTrans
from sub_network.MainFrame import SingleConv
from sub_network.model_3DUnet import UNet3D


class SRDTrans(UNet3D):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            window_size,
            num_heads,
            hidden_dim,
            num_transBlock,
            attn_dropout_rate,
            f_maps=[16, 32, 64, 128, 256],
            input_dropout_rate=0.1
    ):
        super(SRDTrans, self).__init__(
            in_channels=in_channel,
            out_channels=1,
            final_sigmoid=True,
            f_maps=f_maps,
        )

        self.img_time = img_time
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim

        self.conv_before_trans = SingleConv(
            f_maps[-1],
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_after_trans = SingleConv(
            embedding_dim,
            f_maps[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.layers = nn.ModuleList([])
        for idx in range(num_transBlock):
            self.layers.append(
                SpatioTemporalTrans(
                    seq_length= img_time // (2**(len(f_maps) - 1)),
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    space_window_size=window_size,
                    attn_dropout_rate=attn_dropout_rate,
                    input_dropout_rate=input_dropout_rate
                )
            )

    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_after_trans(x)
        return x

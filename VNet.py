import torch
from torch import nn

"""
    modified version of https://github.com/ycwu1997/CoactSeg
"""

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        input_channel = n_filters_in
        for i in range(n_stages):

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))
            input_channel = n_filters_out

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        input_channel = n_filters_in
        for i in range(n_stages):
            

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

            input_channel = n_filters_out

        if n_filters_in != n_filters_out:
            self.conv_on_input = nn.Sequential(
                nn.Conv2d(n_filters_in, n_filters_out, 3, 1, 1),
                nn.BatchNorm2d(n_filters_out)
            )
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x if hasattr(self, 'conv_on_input') is False else self.conv_on_input(x));
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(UpSampling, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False))
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = UpSampling(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpSampling(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpSampling(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpSampling(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine_1 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_2 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_3 = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.block_c2f = convBlock(1, n_filters*3, n_filters, normalization=normalization)

        self.out_conv_3 = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x91 = self.block_nine_1(x8_up)
        if self.has_dropout:
            x91 = self.dropout(x91)

        x92 = self.block_nine_2(x8_up)
        if self.has_dropout:
            x92 = self.dropout(x92)

        x93 = self.block_nine_3(x8_up)
        if self.has_dropout:
            x93 = self.dropout(x93)

        out_seg_diff = torch.cat(((x92 - x91), x91, x93), dim=1)
        out_seg_3 = self.block_c2f(out_seg_diff)
        out_seg_3 = self.out_conv_3(out_seg_3)
        
        return out_seg_3

def get_pos_embedding(tokens, channels):
        pe = torch.zeros((1,tokens, channels) , device= 'cuda', requires_grad=False);
        inv_freq_even = 1.0/((10000)**(torch.arange(0,channels,2) / channels));
        inv_freq_odd = 1.0/((10000)**(torch.arange(1,channels,2) / channels));
        pe[:,:,0::2] = torch.sin(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_even.unsqueeze(dim=0));
        pe[:,:,1::2] = torch.cos(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_odd.unsqueeze(dim=0));
        return pe;

class TwoWayAttention(nn.Module):
    def __init__(self, n_filters ) -> None:
        super().__init__();
        self.cross_attention = nn.MultiheadAttention(n_filters, num_heads=8, add_bias_kv=True );
        self.layer_norm1 = nn.LayerNorm(n_filters)
        self.layer_norm2 = nn.LayerNorm(n_filters)
        self.layer_norm3 = nn.LayerNorm(n_filters)
        self.layer_norm4 = nn.LayerNorm(n_filters)
        self.mlp1 = self.__get_mlp(n_filters,2);
        self.mlp2 = self.__get_mlp(n_filters,2);

    def __get_mlp(self, in_dim, expand_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim*expand_dim),
            nn.GELU(),
            nn.Linear(in_dim*expand_dim, in_dim)
        )
    
    def forward(self, first , second): #first -> queries #second keys
        b,c,h,w,d = first.shape;
        first = first.view(b,c,h*w*d).transpose(-1,-2);
        second = second.view(b,c,h*w*d).transpose(-1,-2);

        first_pe = get_pos_embedding(first.shape[1], c);
        second_pe = get_pos_embedding(second.shape[1], c);

        q = first + first_pe;
        k = second + second_pe;

        attn_out = self.cross_attention(q, k, second)[0]#cross attention from 1 to 2
        first_out = first + attn_out;
        first_out = self.layer_norm1(first_out);

        #first MLP block
        mlp_out = self.mlp1(first_out);
        first_out += mlp_out;
        first_out = self.layer_norm2(first_out);
        

        q = first + first_pe;
        k = second + second_pe;
        attn_out = self.cross_attention(k, q, first)[0]#cross attention from 2 to 1
        second_out = second + attn_out;
        second_out = self.layer_norm3(second_out);

        #second mlp block
        mlp_out = self.mlp2(second_out);
        second_out += mlp_out;
        second_out = self.layer_norm4(second_out);
        
        out = first_out + second_out;
        out = out.transpose(-1,-2).view(b,c,h,w,d)
        return out;


class DecoderCrossAttention(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(DecoderCrossAttention, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.block_five_twowatatten = TwoWayAttention(n_filters*16);
        self.block_five_up = UpSampling(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_twowatatten = TwoWayAttention(n_filters*8);
        self.block_six_up = UpSampling(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_twowatatten = TwoWayAttention(n_filters*4);
        self.block_seven_up = UpSampling(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_twowatatten = TwoWayAttention(n_filters*2);
        self.block_eight_up = UpSampling(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine_1 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_2 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_3 = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.block_c2f = convBlock(1, n_filters*3, n_filters, normalization=normalization)

        self.out_conv_3 = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)



    def forward(self, features1, features2):
        

        x11 = features1[0]
        x12 = features1[1]
        x13 = features1[2]
        x14 = features1[3]
        x15 = features1[4]

        x21 = features2[0]
        x22 = features2[1]
        x23 = features2[2]
        x24 = features2[3]
        x25 = features2[4]

        attn_out5 = self.block_five_twowatatten(x15, x25);
        attn_out4 = self.block_six_twowatatten(x14, x24);
        attn_out3 = self.block_seven_twowatatten(x13, x23);
        
        x5_up = self.block_five_up(attn_out5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x91 = self.block_nine_1(x8_up)
        if self.has_dropout:
            x91 = self.dropout(x91)

        x92 = self.block_nine_2(x8_up)
        if self.has_dropout:
            x92 = self.dropout(x92)

        x93 = self.block_nine_3(x8_up)
        if self.has_dropout:
            x93 = self.dropout(x93)

        out_seg_diff = torch.cat(((x92 - x91), x91, x93), dim=1)
        out_seg_3 = self.block_c2f(out_seg_diff)
        out_seg_3 = self.out_conv_3(out_seg_3)
        
        return out_seg_3
    
class SSLHead(nn.Module):
    def __init__(self, 
                 n_fiters,
                 ssl_head_type = 0,
                 hidden_dim = 2048,
                 bottle_neck_dim = 256,
                 use_bn = False,
                 n_layers = 3) -> None:
        super().__init__();
        self.ssl_head_type = ssl_head_type;

        if ssl_head_type == 0:
            self.first_upsample = nn.Sequential(
                UpSampling(n_fiters*16, n_fiters*8, normalization='instancenorm'),
                UpSampling(n_fiters*8, n_fiters*4, normalization='instancenorm'),
                UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
                UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
            )

            self.second_upsample = nn.Sequential(
                UpSampling(n_fiters*8, n_fiters*4, normalization='instancenorm'),
                UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
                UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
            )

            self.third_upsample = nn.Sequential(
                UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
                UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
            )

            self.fourth_upsample = nn.Sequential(
                UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
            )

            self.final_conv = nn.Conv2d(16, 1, 1);

        elif ssl_head_type == 1:
            self.upsample = nn.Sequential(
                UpSampling(n_fiters*16, n_fiters*8, normalization='instancenorm'),
                UpSampling(n_fiters*8, n_fiters*4, normalization='instancenorm'),
                UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
                UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
            )

            self.final_conv = nn.Conv2d(16, 1, 1);
    

        elif ssl_head_type == 2:
            self.upsample = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=False);
            layers = [nn.Conv2d(256, hidden_dim, 1)];
            if use_bn:
                layers.append(nn.InstanceNorm2d(hidden_dim));
            layers.append(nn.GELU());

            for i in range(n_layers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1));
                if use_bn:
                    layers.append(nn.InstanceNorm2d(hidden_dim));
                layers.append(nn.GELU());
            
            layers.append(nn.Conv2d(hidden_dim, bottle_neck_dim, 1));
            
            self.layers = nn.Sequential(*layers);
                

            self.final_conv = nn.Conv2d(bottle_neck_dim, 1, 1); 
    
    def forward(self, x):

        if self.ssl_head_type == 0:
            u1 = self.first_upsample(x[-1]);
            u2 = self.second_upsample(x[-2]);
            u3 = self.third_upsample(x[-3]);
            u4 = self.fourth_upsample(x[-4]);
            out = self.final_conv(u1+u2+u3+u4);
            return out;

        elif self.ssl_head_type == 1:
            u1 = self.upsample(x[-1]);
            out = self.final_conv(u1);
            return out;

        elif self.ssl_head_type == 2:
            u = self.upsample(x[-1]);
            out = self.layers(u);
            out = self.final_conv(out);
            return out;

        
class VNet(nn.Module):
    def __init__(self, 
                 args, 
                 model_type, 
                 n_channels=3, 
                 n_classes=1, 
                 n_filters=16, 
                 normalization='none', 
                 has_dropout=False, 
                 has_residual=False,
                 ssl_head_type = 0):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.modle_type = model_type;

        if self.modle_type == 'segmentation':
            self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        else:
            self.ssl_head = SSLHead(n_filters,
                                    ssl_head_type=ssl_head_type,
                                    hidden_dim=args.ssl_head_hidden_dim,
                                    bottle_neck_dim=args.ssl_head_bottleneck_dim,
                                    use_bn=args.ssl_head_use_bn,
                                    n_layers=args.ssl_head_n_layers);
        if args.train_layers == 'only-last' or args.train_layers == 'only-decoder':
            self.freeze_parameters(args.train_layers );

    def forward(self, input):
        features = self.encoder(input)
        if self.modle_type == 'segmentation':
            out_seg_3 = self.decoder(features)
            return out_seg_3
        out = self.ssl_head(features);
        return out;

    def freeze_parameters(self, train_layers_type):
        if train_layers_type == 'only-last':
            for p in self.named_parameters():
                if 'final_conv' not in p[0]:
                    p[1].requires_grad_(False);
        elif train_layers_type == 'only-decoder':
            for p in self.named_parameters():
                if 'encoder' in p[0]:
                    p[1].requires_grad_(False);

class VNetCrossAttention(nn.Module):
    def __init__(self, 
                 args, 
                 model_type, 
                 n_channels=3, 
                 n_classes=1, 
                 n_filters=16, 
                 normalization='none', 
                 has_dropout=False, 
                 has_residual=False,
                 ssl_head_type = 0):
        super(VNetCrossAttention, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.modle_type = model_type;

        if self.modle_type == 'segmentation':
            self.decoder = DecoderCrossAttention(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        else:
            self.ssl_head = SSLHead(n_filters,
                                    ssl_head_type=ssl_head_type,
                                    hidden_dim=args.ssl_head_hidden_dim,
                                    bottle_neck_dim=args.ssl_head_bottleneck_dim,
                                    use_bn=args.ssl_head_use_bn,
                                    n_layers=args.ssl_head_n_layers);
        if args.train_layers == 'only-last' or args.train_layers == 'only-decoder':
            self.freeze_parameters(args.train_layers );

    def forward(self, input1, input2):
        features1 = self.encoder(input1)
        features2 = self.encoder(input2)

        #features = [torch.cat([f1, f2], dim = 1) for f1,f2 in zip(features1, features2)];
        if self.modle_type == 'segmentation':
            out_seg_3 = self.decoder(features1, features2)
            return out_seg_3
        out = self.ssl_head(features);
        return out;

    def freeze_parameters(self, train_layers_type):
        if train_layers_type == 'only-last':
            for p in self.named_parameters():
                if 'final_conv' not in p[0]:
                    p[1].requires_grad_(False);
        elif train_layers_type == 'only-decoder':
            for p in self.named_parameters():
                if 'encoder' in p[0]:
                    p[1].requires_grad_(False);
    
def test():
    model = VNetCrossAttention(model_type='segmentation', n_channels=3, n_classes=2, normalization='batchnorm', has_dropout=True);
    inp = torch.rand((2, 3, 256, 256, 256));
    model(inp);
if __name__ == '__main__':
    test();
    

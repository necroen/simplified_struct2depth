
import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    # 这个网络对输入的图片是有顺序要求的，target 就是 中间的图片，ref就是两边的图片
    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)  
        # 这里做了cat，跟struct2depth中不做joint_encoder
        # 的效果一致
        
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        return pose
        # [1, 2, 6]  中间跟两边的相对 pose，到底是什么顺序？ T01 T12 还是  T10 T12



if __name__ == '__main__':
    pose_exp_net = PoseExpNet(nb_ref_imgs=2, output_exp=False)
    # 参考帧数目为前后两帧，所以为2，  output explainability_mask，不考虑运动

    pose_exp_net.init_weights()
    pose_exp_net.train()
    
    batchsize = 1
    
    tgt_img = torch.rand([batchsize, 3, 128, 416]) 
    ref_imgs = [torch.rand([batchsize, 3, 128, 416]), 
                torch.rand([batchsize, 3, 128, 416]) ]
    
    pose = pose_exp_net(tgt_img, ref_imgs)
    print("pose size:", pose.size() ) # torch.Size([1, 2, 6]) when batchsize = 1
    

    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment = 'pose_net') as w:
    #     w.add_graph(pose_exp_net, (tgt_img, ref_imgs),  )  

        # tensorboard --logdir=./runs --port=6006 
        #  http://localhost:6006/

    # conv1 = conv(9, 16, kernel_size=7)

    # inputs = [tgt_img]
    # inputs.extend(ref_imgs)

    # print( len(inputs) )  # 3
    # print(inputs[0].size() )  # torch.Size([1, 3, 128, 416])

    # inputs = torch.cat(inputs, 1) # NCHW  对 C 维度做 拼接  1个tgt, 2个ref
    # print("inputs:", inputs.size() ) # torch.Size([1, 9, 128, 416])

    # out_conv1 = conv1(inputs)
    # print(out_conv1.size() ) # torch.Size([1, 16, 64, 208])
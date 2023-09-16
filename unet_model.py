from unet_components import *

class UNET(nn.Module):
    def __init__(self, n_channels, latent_dim, kernel_size = 3):
        super(UNET, self).__init__()
        self.n_channels = n_channels

        # ## encoder layers ##
        # self.inc = nn.Conv2d(1, 16, 3, padding=1) 
        # self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)

        # ## decoder layers ##
        # self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.inc = (DoubleConv(n_channels, n_channels))
        self.down1 = (Down(n_channels, n_channels))
        self.down2 = (Down(n_channels, n_channels))
        self.latent = nn.Conv2d(
                        n_channels,
                        latent_dim,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2)
        #self.down3 = (Down(256, 512))
        #factor = 1
        #self.down4 = (Down(512, 1024))
        #self.up1 = (Up(1024, 512))
        #self.up2 = (Up(512, 256))
        self.latent2 = nn.Sequential(
                    *[  
                        nn.Conv2d(
                            latent_dim,
                            n_channels,
                            kernel_size,
                            stride=1,
                            padding=kernel_size // 2),
                        nn.BatchNorm2d(n_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
        self.up3 = (Up(n_channels, n_channels))
        self.up4 = (Up(n_channels, n_channels))
        self.outc = (OutConv(n_channels, n_channels))


    def forward(self, x):
        # ## encode ##
        # x = F.relu(self.inc(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)  # compressed representation

        # ## decode ##
        # x = F.batch_norm(F.relu(self.t_conv1(x)))
        # # output layer (with sigmoid for scaling from 0 to 1)
        # logits = F.sigmoid(self.t_conv2(x))
        #print(x.shape)
        x = self.inc(x)
        # print(x1.shape)
        x = self.down1(x)
        # print(x2.shape)
        x = self.down2(x)
        x = self.latent(x)
        x = self.latent2(x)
        # print(x3.shape)
        #x = self.down3(x)
        # print(x4.shape)
        #x = self.down4(x)
        # print(x5.shape)
        #x = self.up1(x)
        # print(x.shape)
        #x = self.up2(x)
        # # print(x.shape)
        x = self.up3(x)
        # print(x.shape)
        x = self.up4(x)
        # print(x.shape)
        out = self.outc(x)
        #print(out.shape)

        return out
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        #self.down3 = torch.utils.checkpoint(self.down3)
        #self.down4 = torch.utils.checkpoint(self.down4)
        #self.up1 = torch.utils.checkpoint(self.up1)
        #self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
    

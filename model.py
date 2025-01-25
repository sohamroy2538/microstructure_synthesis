import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ConvNet_v3_normal(nn.Module):
    def __init__(self ):  # ,
        super(ConvNet_v3_normal, self).__init__()
        
        # Use either standard convolution or deformable convolution
        conv_layer = nn.Conv2d
        
        self.enc1 = nn.Sequential(
            conv_layer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            conv_layer(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) , # inp_size=128),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            conv_layer(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1 ), # inp_size=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            conv_layer(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1 ), # inp_size=32),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Widening: Increase spatial dimensions, reduce channels)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 , out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
       
        self.dec1 = nn.Sequential(
            conv_layer(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1) # Final layer
        )

    def forward(self, x):

        # Encoding path
        e1 = self.enc1(x )  # First encoder block
        e2 = self.enc2(e1)  # Second encoder block
        e3 = self.enc3(e2)  # Third encoder block
        e4 = self.enc4(e3)  # Fourth encoder block'''


        #simple
        d4 = self.dec4(e4 )
        d3 = self.dec3(d4 + e3)  # Skip connection from e3
        d2 = self.dec2(d3  + e2)  # Skip connection from e2
        d1 = self.dec1(d2 + e1)  # Skip connection from e1'''
        return d1
    

class ConvNet_v3(nn.Module):
    def __init__(self):  # ,
        super(ConvNet_v3, self).__init__()
        
        # Use either standard convolution or deformable convolution
        conv_layer = nn.Conv2d 
        self.spece4 = None
        self.spece3 = None
        self.spece2 = None
        self.spece1 = None

        
        self.enc1 = nn.Sequential(
            conv_layer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            conv_layer(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) , # inp_size=128),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            conv_layer(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1 ), # inp_size=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            conv_layer(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1 ), # inp_size=32),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Widening: Increase spatial dimensions, reduce channels)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 , out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
       
        self.dec1 = nn.Sequential(
            conv_layer(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1) # Final layer
        )

        self.l1 = conv_layer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.l2 = conv_layer(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.l3 = conv_layer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.l4 = conv_layer(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)


    def forward(self, x , spece4, spece3, spece2, spece1 ):
        
        self.spece4 = spece4
        self.spece3 = spece3
        self.spece2 = spece2
        self.spece1 = spece1        
        
        e1 = self.enc1(x)  # First encoder block
        e2 = self.enc2(e1 * self.l1(self.spece1))  # Second encoder block
        e3 = self.enc3(e2 *   self.l2(self.spece2))  # Third encoder block
        e4 = self.enc4(e3 * self.l3(self.spece3))  
        # Decoding path

        #simple
        d4 = self.dec4(e4 )
        d3 = self.dec3(d4 + e3)  # Skip connection from e3
        d2 = self.dec2(d3  + e2)  # Skip connection from e2
        d1 = self.dec1(d2 + e1)  # Skip connection from e1'''
        return d1 #, torch.real(fft.ifft2((x2/torch.abs(x2)) * self.spece1)) , torch.real(fft.ifft2((x3/torch.abs(x3)) * self.spece2)), torch.real(fft.ifft2((x4/torch.abs(x4)) * self.spece3))



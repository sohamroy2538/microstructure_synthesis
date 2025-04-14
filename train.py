#!pip install lpips
#import lpips
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import argparse
import os
import shutil
from deep_features import *
#from metrics.metric_cw_ssim import *
from PIL import Image
from torchvision.utils import save_image
from utils import *
from loss import slicing_loss
import gc
from model import ConvNet_v3_normal , ConvNet_v3
torch.backends.cuda.matmul.allow_tf32 = True



class create_img_normal(nn.Module):
    def __init__(self, data, target_dir =  None , img_size = 256, use_deform_conv = False , use_bcos = True , task = "rectify"):
        super().__init__()
        set_seed(seed = 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor = transform_img()(Image.open(data)).unsqueeze(0).to(self.device)
        self.task = task
        a = img_size
        self.top1 = torch.randint(0, self.anchor.shape[2] - a + 1, (1,)).item()
        self.left1 = torch.randint(0, self.anchor.shape[3] - a + 1, (1,)).item()
        self.anchor1 = self.anchor[:, :, self.top1:self.top1 + a, self.left1:self.left1 + a].to(self.device)
        #self.anchor1 = torchvision.transforms.Resize((a, a))(self.anchor)
        if self.task == "rectify":
            self.target = torchvision.transforms.Resize((a, a))(transform_img()(Image.open(target_dir)).unsqueeze(0)).to(self.device)
            self.model_name = "create_img_existing"
        
        elif self.task == "train_from_scratch":
            self.list_activations_example = [get_gram_matrices(i).detach() for i in (resnet18_bcos().get_gram_matrices(torchvision.transforms.Resize((a, a))(self.anchor)))]
            self.model_name = "create_img"
        

        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        self.model = ConvNet_v3_normal().to(self.device) 
        self.model.apply(init_weights)
        self.use_bcos = use_bcos
        self.history = {"loss" : [] , "perc_loss" : [] , "mse_loss" :[], "L1_loss" :[] , "ssim_loss" : []}
        self.optimizer = optim.Adam(self.model.parameters() , lr = 0.0002 ,betas = (0.5,0.999))


    def forward(self):

        if self.task == "rectify":
            self.img = torch.sigmoid(self.model(self.target))
            mse_loss = slicing_loss(self.img  , self.anchor1.detach()  , use_bcos = self.use_bcos, example_style = None)

        #adding noise to the feature maps
        elif self.task == "train_from_scratch":
            self.img = torch.sigmoid(self.model(torch.rand_like(self.anchor1)))  
            mse_loss = slicing_loss(self.img  , self.anchor1.detach()  , use_bcos = self.use_bcos, example_style = self.list_activations_example)
            
        return mse_loss

    def fit(self,epoch):
        self.train()
        self.zero_grad()
        loss = self()
        #self.history["perc_loss"].append(perc_loss.item())
        self.history["mse_loss"].append(loss.item())

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad() 
        #self.target = torch.clamp(self.target, min=0, max=1)
        torch.cuda.empty_cache()
        gc.collect() 

        #os.makedirs(self.model_name, exist_ok=True)
        if(epoch % 100 == 0):
            self.eval()
            with torch.no_grad():
                if self.task == "train_from_scratch":
                    save_image(torch.sigmoid(self.model(torch.rand_like(self.anchor1))),
                            f"{self.model_name}/epoch_{str(epoch)}.png")
                else:
                    save_image(torch.sigmoid(self.model(self.target)),
                            f"{self.model_name}/epoch_{str(epoch)}.png")                    
            print(f" loss is {loss.item()} image is epoch_{str(epoch)}.png")





class create_img_dendritic(nn.Module):
    def __init__(self, data , img_size = 256, use_deform_conv = False , use_bcos = True):
        super().__init__()
        set_seed(seed = 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor = transform_img()(Image.open(data)).unsqueeze(0).to(self.device)
        self.img_size = img_size
        self.top1 = torch.randint(0, self.anchor.shape[2] - self.img_size + 1, (1,)).item()
        self.left1 = torch.randint(0, self.anchor.shape[3] - self.img_size + 1, (1,)).item()
        self.anchor1 = self.anchor[:, :, self.top1:self.top1 + self.img_size, self.left1:self.left1 + self.img_size].to(self.device)
        self.model_name = "create_img"

        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
        
        self.model = ConvNet_v3().to(self.device)

        self.model.apply(init_weights)
        self.use_bcos = use_bcos


        self.history = {"loss" : [] , "perc_loss" : [] , "mse_loss" :[], "L1_loss" :[] , "ssim_loss" : []}
        self.optimizer = optim.Adam(self.model.parameters() , lr = 0.0002 ,betas = (0.5,0.999))


    def forward(self):

        a = self.img_size
        b = a//2
        c = b//2
        d = c//2
        
        self.top1 = torch.randint(0, self.anchor.shape[2] - a + 1, (1,)).item()
        self.left1 = torch.randint(0, self.anchor.shape[3] - a + 1, (1,)).item()
        self.anchor1 = self.anchor[:, :, self.top1:self.top1 + a, self.left1:self.left1 + a].to(self.device)


        self.top2 = torch.randint(0, self.anchor.shape[2] - b + 1, (1,)).item()
        self.left2 = torch.randint(0, self.anchor.shape[3] - b + 1, (1,)).item()
        self.anchor2 = self.anchor[:, :, self.top2:self.top2 + b, self.left2:self.left2 + b].to(self.device)


        self.top3 = torch.randint(0, self.anchor.shape[2] - c + 1, (1,)).item()
        self.left3 = torch.randint(0, self.anchor.shape[3] - c + 1, (1,)).item()
        self.anchor3 = self.anchor[:, :, self.top3:self.top3 + c, self.left3:self.left3 + c].to(self.device)


        self.top4 = torch.randint(0, self.anchor.shape[2] - d + 1, (1,)).item()
        self.left4 = torch.randint(0, self.anchor.shape[3] - d + 1, (1,)).item()
        self.anchor4 = self.anchor[:, :, self.top4:self.top4 + d, self.left4:self.left4 + d].to(self.device)

        self.spectrume4 = (self.anchor4.detach())
        self.spectrume3 = (self.anchor3.detach())
        self.spectrume2 = (self.anchor2.detach())
        self.spectrume1 = (self.anchor1.detach())


        self.img = torch.sigmoid(self.model(torch.rand_like(self.anchor1), self.spectrume4 , self.spectrume3 , self.spectrume2, self.spectrume1))           
        mse_loss = slicing_loss(self.img  , torchvision.transforms.Resize((a, a))(self.anchor)  , use_bcos = self.use_bcos)
        return  mse_loss

    def fit(self,epoch):
        self.train()
        self.zero_grad()
        loss = self()
        self.history["mse_loss"].append(loss.item())

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad() 
        torch.cuda.empty_cache()
        gc.collect() 

        #os.makedirs(self.model_name, exist_ok=True)
        if(epoch % 100 == 0):
            self.eval()
            with torch.no_grad():
                save_image(torch.sigmoid(self.model(torch.rand_like(self.anchor1),self.spectrume4  , self.spectrume3, self.spectrume2 , self.spectrume1)),
                         f"{self.model_name}/epoch_{str(epoch)}.png")
            print(f" loss is {loss.item()} image is epoch_{str(epoch)}.png")


class create_img_wo_texture_model(nn.Module):
    def __init__(self, data):
        super().__init__()
        set_seed(seed = int(time.time()))
        self.data = data
        self.vgg19 = VGG19()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor = torchvision.transforms.Resize((256, 256))(transform_img()(Image.open(data))).unsqueeze(0).to(self.device)
        self.generated_image = torch.rand_like(self.anchor).to(self.device)
        self.generated_image = torch.nn.parameter.Parameter(self.generated_image)
        self.optimizer = torch.optim.LBFGS([self.generated_image], lr=1, max_iter=64, tolerance_grad=0.0)


    # LBFGS closure function
    def closure(self):
        self.optimizer.zero_grad()
        loss = slicing_loss(self.generated_image, self.anchor, use_bcos=False, vgg_19_object=self.vgg19)
        loss.backward()
        return loss
    
    def forward(self):
        # optimization loop
        for iteration in range(12):
            print("iteration", iteration )
            self.optimizer.step(self.closure)
        return self.generated_image
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Model Training Script")

    parser.add_argument("--task", type=str, default="train_from_scratch" , required=True, help="Path to the image directory")
    parser.add_argument("--with_texture_model", type=bool, default=True , required=True, help="Whether to create texture specific model")
    parser.add_argument("--image_dir", type=str, required=False, help="Path to the image directory")
    parser.add_argument("--target_dir", type=str, required=False, help="Path to the texture needs rectification directory")
    parser.add_argument("--image_size", type=int, default=256, help="Size of image patch")
    parser.add_argument("--dendritic_pattern", type=bool, default=False, required=False, help="If texture has long range dendritic dependancy")    
    parser.add_argument("--epochs", type=int, default=6000, help="Number of epochs to train")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate for non texture model")

    args = parser.parse_args()

    epochs = args.epochs
    task = args.task
    image_dir = args.image_dir
    if task == "rectify":
        target_dir = args.target_dir
        epochs = 1500
    use_bcos = args.use_bcos
    img_size = args.image_size
    dendritic_pattern = args.dendritic_pattern
    with_texture_model = args.with_texture_model
    num_images = args.num_images

    if dendritic_pattern:
        img_size = 256
    
    # image_dir is source image directory
    # target_dir is existing method image directory
    if task == "rectify":
        create_img_model = create_img_normal(image_dir, target_dir, use_bcos=use_bcos, task = "rectify")
        for i in range(epochs+1):
            create_img_model.fit(i)
        shutil.copy(f"./create_img_existing/epoch_{epochs}.png", f"./output/rectified_{os.path.basename(target_dir)}")

    elif task == "train_from_scratch" and with_texture_model:
        if not dendritic_pattern:
            create_img_model = create_img_normal(image_dir, target_dir, use_bcos=use_bcos , task = "train_from_scratch")
        else:
            create_img_model = create_img_dendritic(image_dir, use_bcos=use_bcos)
       
        for i in range(epochs+1):
            create_img_model.fit(i)

        model_save_path = rf"./models/{task}_{os.path.basename(image_dir)}_{img_size}.pth"
        torch.save(create_img_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    elif task == "train_from_scratch" and not with_texture_model:
        for i in range(num_images):
            create_img_model = create_img_wo_texture_model(image_dir)
            generated_image = create_img_model()
            save_image(generated_image, os.path.join("./output", f"{os.path.basename(image_dir[: -4])}_output_{i + 1}.png"))

        

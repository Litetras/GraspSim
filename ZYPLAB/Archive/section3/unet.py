import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# -------------------------- 基础卷积块 --------------------------
class ConvBlock(nn.Module):
    """Unet基础单元：2次3×3卷积 + ReLU激活，提取局部特征"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# -------------------------- Unet模型（带特征提取） --------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # 编码器（下采样，浓缩语义）
        self.enc1 = ConvBlock(in_channels, 64)  # 编码器上层（细节）
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化下采样
        
        # 瓶颈层（最底层，语义核心）
        self.bottleneck = ConvBlock(512, 1024)
        
        # 解码器（上采样，恢复细节）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)  # 解码器上层（接近原始图像）
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, return_features=False):
        """前向传播，return_features=True时返回关键层特征"""
        # 编码器前向（下采样）
        x1 = self.enc1(x)          # 编码器上层：256×256（细节）
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # 瓶颈层（最底层）：16×16（语义）
        bottleneck = self.bottleneck(self.pool(x4))
        
        # 解码器前向（上采样+拼接）
        x = self.up4(bottleneck)
        x = torch.cat([x, x4], dim=1)
        x4_dec = self.dec4(x)
        
        x = self.up3(x4_dec)
        x = torch.cat([x, x3], dim=1)
        x3_dec = self.dec3(x)
        
        x = self.up2(x3_dec)
        x = torch.cat([x, x2], dim=1)
        x2_dec = self.dec2(x)
        
        x = self.up1(x2_dec)
        x = torch.cat([x, x1], dim=1)
        x1_dec = self.dec1(x)      # 解码器上层：256×256（接近原始图像）
        
        out = self.out(x1_dec)
        
        if return_features:
            features = {
                "encoder_1": x1,          # 编码器上层（细节）
                "bottleneck": bottleneck, # 最底层（语义）
                "decoder_1": x1_dec       # 解码器上层（接近原始图像）
            }
            return out, features
        return out

# -------------------------- 图像加载与可视化 --------------------------
def load_test_image(image_path="test.jpg", size=(256, 256)):
    """加载并预处理测试图像（替换为你的图像路径）"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 增加batch维度
    return image, image_tensor

def visualize_features_and_segmentation(original_image, features, segmentation_output):
    """可视化不同层特征 + 最终分割结果"""
    # 改为2行3列的子图，展示更多内容
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Unet Features and Segmentation Result", fontsize=16)
    
    # 1. 原始图像
    axes[0,0].imshow(original_image)
    axes[0,0].set_title("Original Image (256×256)")
    axes[0,0].axis("off")
    
    # 2. 编码器上层特征（细节）
    enc1_feat = features["encoder_1"][0, 0, :, :].detach().cpu().numpy()
    axes[0,1].imshow(enc1_feat, cmap="gray")
    axes[0,1].set_title("Encoder Layer 1 Feature (256×256) - Details")
    axes[0,1].axis("off")
    
    # 3. 瓶颈层（最底层）特征（语义）
    bottleneck_feat = features["bottleneck"][0, 0, :, :].detach().cpu().numpy()
    axes[0,2].imshow(bottleneck_feat, cmap="gray")
    axes[0,2].set_title("Bottleneck (Bottom Layer) Feature (16×16) - Semantics")
    axes[0,2].axis("off")
    
    # 4. 解码器上层特征（接近原始图像）
    dec1_feat = features["decoder_1"][0, 0, :, :].detach().cpu().numpy()
    axes[1,0].imshow(dec1_feat, cmap="gray")
    axes[1,0].set_title("Decoder Layer 1 Feature (256×256) - Close to Original")
    axes[1,0].axis("off")
    
    # 5. 分割输出（概率图）
    seg_prob = segmentation_output[0, 0, :, :].detach().cpu().numpy()
    axes[1,1].imshow(seg_prob, cmap="gray")
    axes[1,1].set_title("Segmentation Probability Map (256×256)")
    axes[1,1].axis("off")
    
    # 6. 分割输出（二值掩码图，阈值0.5）
    seg_binary = np.where(seg_prob > 0.5, 1, 0)  # 阈值分割，1=勺子，0=背景
    axes[1,2].imshow(seg_binary, cmap="gray")
    axes[1,2].set_title("Segmentation Binary Mask (256×256) - Final Result")
    axes[1,2].axis("off")
    
    plt.tight_layout()
    plt.savefig("unet_features_and_segmentation.png")
    plt.show()

# -------------------------- 主函数（运行测试） --------------------------
if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(in_channels=3, out_channels=1).to(device)
    
    # 加载测试图像（替换为你的图像路径）
    original_img, img_tensor = load_test_image("ZYPLAB/section3/test.jpg", size=(256, 256))
    img_tensor = img_tensor.to(device)
    
    # 提取特征和分割输出
    with torch.no_grad():
        output, features = unet(img_tensor, return_features=True)
    
    # 打印特征尺寸（验证维度）
    print("特征尺寸验证：")
    print(f"原始图像tensor尺寸: {img_tensor.shape}")          # (1, 3, 256, 256)
    print(f"编码器上层特征尺寸: {features['encoder_1'].shape}")  # (1, 64, 256, 256)
    print(f"瓶颈层（最底层）特征尺寸: {features['bottleneck'].shape}")  # (1, 1024, 16, 16)
    print(f"解码器上层特征尺寸: {features['decoder_1'].shape}")  # (1, 64, 256, 256)
    print(f"最终分割输出尺寸: {output.shape}")  # (1, 1, 256, 256) —— 新增
    
    # 可视化特征 + 分割结果（调用修改后的可视化函数）
    visualize_features_and_segmentation(original_img, features, output)

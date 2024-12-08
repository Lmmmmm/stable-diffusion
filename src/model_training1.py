import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
import argparse
from tqdm import tqdm
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


# def load_dataset():
#     dataset = []
#     with open('kanji_jpg/metadata.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             image_path = os.path.join('kanji_jpg', item['file_name'])
#             image = Image.open(image_path)
#             # 转换为灰度图并调整大小
#             image = image.convert('L')
# 这个错误是因为输入图像通道数不匹配。StableDiffusionPipeline期望RGB图像（3通道），但我们提供的是灰度图像（1通道）。需要修改图像处理部分，将灰度图转换为3通道图像。以下是修正后的load_dataset函数：
#             image = image.resize((128, 128), Image.LANCZOS)
#             # 转换为tensor
#             image = transforms.ToTensor()(image)
#             dataset.append({
#                 "image": image,
#                 "meaning": item['text']
#             })
#     return dataset
def load_dataset():
    dataset = []
    with open('kanji_jpg/metadata.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join('kanji_jpg', item['file_name'])
            image = Image.open(image_path)
            # 转换为灰度图并调整大小
            image = image.convert('L')
            image = image.resize((128, 128), Image.LANCZOS)
            # 转换灰度图为RGB图像
            image = image.convert('RGB')
            # 转换为tensor
            image = transforms.ToTensor()(image)
            dataset.append({
                "image": image,
                "meaning": item['text']
            })
    return dataset

class KanjiDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def train_model(
    model_id="CompVis/stable-diffusion-v1-4",
    learning_rate=1e-5,
    batch_size=4,
    num_epochs=10
):
    # # 设置设备
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    # print(f"Using device: {device}")

    # # 加载模型
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float32,
    #     use_safetensors=True
    # ).to(device)

    # # 启用内存优化
    # pipeline.enable_attention_slicing(1)
    # pipeline.enable_sequential_cpu_offload()
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 加载模型 - 移除use_safetensors
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float32
    # ).to(device)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # 只启用attention slicing来优化内存
    pipeline.enable_attention_slicing()

    # 如果是PyTorch 1.13需要进行一次预热
    if torch.__version__.startswith("1.13"):
        _ = pipeline("warmup", num_inference_steps=1)
    # 初始化wandb
    wandb.init(project="kanji_diffusion", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs
    })

    # 加载数据集
    raw_dataset = load_dataset()
    dataset = KanjiDataset(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate
    )

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # for batch_idx, batch in enumerate(progress_bar):
        #     images = batch["image"].to(device)
        #     text = batch["meaning"]

        #     # 前向传播
        #     with torch.autocast("mps", dtype=torch.float16):
        #         loss = pipeline(
        #             prompt=text,
        #             image=images,
        #             return_dict=False
        #         )[0]

        #     # 反向传播
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # 修改训练循环部分
        for batch_idx, batch in enumerate(progress_bar):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            images = batch["image"].to(device)
            text = batch["meaning"]

            # 前向传播
            with torch.autocast("mps", dtype=torch.float16):
                # 使用text_embeddings
                text_inputs = pipeline.tokenizer(
                    text,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                text_embeddings = pipeline.text_encoder(text_inputs.input_ids)[0]

                # 计算latents
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                # 添加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # 获取模型预测
                noise_pred = pipeline.unet(noisy_latents, timesteps, text_embeddings).sample

                # 计算loss
                loss = F.mse_loss(noise_pred, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # for batch_idx, batch in enumerate(progress_bar):
        #     images = batch["image"].to(device)
        #     text = batch["meaning"]

        #     # 前向传播
        #     with torch.autocast("mps", dtype=torch.float16):
        #         output = pipeline(
        #             prompt=text,
        #             image=images,
        #             return_dict=False
        #         )
        #         # 获取第一个元素作为loss
        #         loss = output[0]
        #         if isinstance(loss, list):
        #             loss = torch.stack(loss).mean()

        #     # 反向传播
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     epoch_loss += loss.item()
        #     progress_bar.set_postfix({"loss": loss.item()})

            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

            # 每100个batch保存样本
            if batch_idx % 100 == 0:
                sample_images = pipeline(
                    prompt=text[:1],
                    num_images_per_prompt=1
                ).images
                wandb.log({"samples": [wandb.Image(img) for img in sample_images]})

        # 保存checkpoint
        if (epoch + 1) % 5 == 0:
            pipeline.save_pretrained(f"checkpoints/epoch_{epoch+1}")

        # 记录epoch平均损失
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch_loss": avg_loss})
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
# import torch
# from diffusers import StableDiffusionPipeline
# from torch.utils.data import DataLoader, Dataset  # 添加Dataset导入
# import numpy as np
# import wandb
# import argparse
# from tqdm import tqdm
# import json
# import os
# from PIL import Image
# import torchvision.transforms as transforms


# # def load_dataset():
# #     """加载处理好的数据集"""
# #     try:
# #         data = np.load("data/processed/kanji_dataset.npy", allow_pickle=True)
# #         if len(data) == 0:
# #             raise ValueError("数据集为空")

# #         # 转换为PyTorch数据集格式
# #         class KanjiDataset(torch.utils.data.Dataset):
# #             def __init__(self, data):
# #                 self.data = data

# #             def __len__(self):
# #                 return len(self.data)

# #             def __getitem__(self, idx):
# #                 item = self.data[idx]
# #                 return {
# #                     'image': torch.FloatTensor(item['image']),
# #                     'meaning': item['meaning']
# #                 }

# #         return KanjiDataset(data)
# #     except FileNotFoundError:
# #         raise FileNotFoundError("找不到数据集文件，请先运行data_processing.py生成数据集")
# #     except Exception as e:
# #         raise Exception(f"加载数据集时出错: {str(e)}")

# def load_dataset():
#     # 读取metadata.jsonl
#     dataset = []
#     with open('kanji_jpg/metadata.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             # 加载图像
#             image_path = os.path.join('kanji_jpg', item['file_name'])
#             image = Image.open(image_path)
#             # 转换为tensor
#             image = transforms.ToTensor()(image)
#             # 添加到dataset
#             dataset.append({
#                 "image": image,
#                 "meaning": item['text']
#             })

#     return dataset

# class KanjiDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# # 在train_model函数中修改数据加载部分
# dataset = KanjiDataset(load_dataset())
# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4
# )

# def train_model(
#     model_id="CompVis/stable-diffusion-v1-4",
#     learning_rate=1e-5,
#     batch_size=8,
#     num_epochs=10
# ):
#     if torch.backends.mps.is_available():
#         device = "mps"  # Apple Silicon GPU
#     else:
#         device = "cpu"  # CPU

#     # 加载模型到正确的设备
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float32  # 使用float32而不是float16
#     ).to(device)

#     # 初始化wandb
#     wandb.init(project="kanji_diffusion", config={
#         "learning_rate": learning_rate,
#         "batch_size": batch_size,
#         "epochs": num_epochs
#     })

#     # 加载数据集
#     dataset = load_dataset()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # 优化器
#     optimizer = torch.optim.AdamW(
#         pipeline.unet.parameters(),
#         lr=learning_rate
#     )

#     # 训练循环
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

#         for batch_idx, batch in enumerate(progress_bar):
#             images = batch["image"].to("cuda")
#             text = batch["meaning"]

#             # 前向传播
#             with torch.cuda.amp.autocast():
#                 loss = pipeline(
#                     prompt=text,
#                     image=images,
#                     return_dict=False
#                 )[0]

#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#             # 更新进度条
#             progress_bar.set_postfix({"loss": loss.item()})

#             # 记录到wandb
#             wandb.log({
#                 "batch_loss": loss.item(),
#                 "epoch": epoch,
#                 "batch": batch_idx
#             })

#             # 每100个batch保存一次样本
#             if batch_idx % 100 == 0:
#                 sample_images = pipeline(
#                     prompt=text[:1],
#                     num_images_per_prompt=1
#                 ).images
#                 wandb.log({"samples": [wandb.Image(img) for img in sample_images]})

#         # 保存checkpoint
#         if (epoch + 1) % 5 == 0:
#             pipeline.save_pretrained(f"checkpoints/epoch_{epoch+1}")

#         # 记录epoch平均损失
#         avg_loss = epoch_loss / len(dataloader)
#         wandb.log({"epoch_loss": avg_loss})
#         print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=1e-5)

#     args = parser.parse_args()

#     train_model(
#         batch_size=args.batch_size,
#         num_epochs=args.epochs,
#         learning_rate=args.lr
#     )
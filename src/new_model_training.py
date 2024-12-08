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
            # 转换为RGB图像
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
    batch_size=2,  # 减小batch size
    num_epochs=10
):
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 加载模型
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # 使用float16
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # 启用内存优化
    pipeline.enable_attention_slicing(slice_size="max")

    # 初始化wandb
    wandb.init(project="kanji_diffusion", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs
    })

    # load dataset
    raw_dataset = load_dataset()
    dataset = KanjiDataset(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 减少worker数量
        pin_memory=False
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate
    )

    # train loop
    accumulation_steps = 4
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

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
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # 获取模型预测
                noise_pred = pipeline.unet(noisy_latents, timesteps, text_embeddings).sample

                # 计算loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

            if batch_idx % 100 == 0:
                sample_images = pipeline(
                    prompt=text[:1],
                    num_images_per_prompt=1
                ).images
                wandb.log({"samples": [wandb.Image(img) for img in sample_images]})

            # 清理内存
            del loss
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # 保存checkpoint
        if (epoch + 1) % 5 == 0:
            pipeline.save_pretrained(f"checkpoints/epoch_{epoch+1}")

        # 记录epoch平均损失
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch_loss": avg_loss})
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
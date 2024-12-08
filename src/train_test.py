import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JPG_DIR = os.path.join(DATA_DIR, 'kanji_jpg')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')

class KanjiDataset(Dataset):
    def __init__(self, data_dir=JPG_DIR, metadata_file='metadata.jsonl', max_samples=10):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.data = []
        metadata_path = os.path.join(data_dir, metadata_file)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'meaning': item['text']
        }

def train_model(
    model_id="CompVis/stable-diffusion-v1-4",
    learning_rate=1e-6,
    batch_size=2,
    num_epochs=2,
    max_samples=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True
    ).to(device)

    # 启用内存优化
    pipeline.enable_attention_slicing(slice_size="max")
    pipeline.enable_vae_slicing()

    # 准备数据加载器
    dataset = KanjiDataset(max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate
    )

    # 创建检查点目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")

            images = batch["image"].to(device)
            text = batch["meaning"]

            # 前向传播
            with torch.autocast(device_type=device.type):
                text_inputs = pipeline.tokenizer(
                    text,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                text_embeddings = pipeline.text_encoder(text_inputs.input_ids)[0]
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                noise_pred = pipeline.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise)

            print(f"Batch loss: {loss.item():.4f}")

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 清理内存
            torch.cuda.empty_cache()

        # 保存检查点
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'test_epoch_{epoch+1}')
        os.makedirs(checkpoint_path, exist_ok=True)
        pipeline.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1}")

    return pipeline

def main():
    try:
        print("开始测试训练...")
        print(f"数据目录: {DATA_DIR}")
        print(f"图片目录: {JPG_DIR}")
        print(f"检查点目录: {CHECKPOINT_DIR}")
        train_model()
        print("测试训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
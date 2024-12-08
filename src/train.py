import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from tqdm import tqdm

# 定义数据目录（相对于src目录的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JPG_DIR = os.path.join(DATA_DIR, 'kanji_jpg')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')

class KanjiDataset(Dataset):
    def __init__(self, data_dir=JPG_DIR, metadata_file='metadata.jsonl'):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.data = []
        metadata_path = os.path.join(data_dir, metadata_file)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
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

class KanjiTrainer:
    def __init__(
        self,
        model_id="CompVis/stable-diffusion-v1-4",
        learning_rate=1e-6,
        batch_size=4,
        num_epochs=10
    ):
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def setup_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)

        self.pipeline.enable_attention_slicing(slice_size="max")
        self.pipeline.enable_vae_slicing()

    def setup_data(self):
        dataset = KanjiDataset()
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.pipeline.unet.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        text = batch["meaning"]

        with torch.autocast(device_type=self.device.type):
            text_inputs = self.pipeline.tokenizer(
                text,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]
            latents = self.pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.pipeline.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=self.device
            )
            noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = self.pipeline.unet(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

        return loss

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}')
        os.makedirs(checkpoint_path, exist_ok=True)
        self.pipeline.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                torch.cuda.empty_cache()

                loss = self.train_step(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.pipeline.unet.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

                torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

def main():
    try:
        print("开始训练...")
        print(f"数据目录: {DATA_DIR}")
        print(f"图片目录: {JPG_DIR}")
        print(f"检查点目录: {CHECKPOINT_DIR}")

        trainer = KanjiTrainer()
        trainer.train()

        print("训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
# import torch
# import torch.nn.functional as F
# from diffusers import StableDiffusionPipeline
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from PIL import Image
# import json
# import os
# from tqdm import tqdm

# # 定义数据目录
# DATA_DIR = 'data'
# JPG_DIR = os.path.join(DATA_DIR, 'kanji_jpg')
# CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')

# class KanjiDataset(Dataset):
#     """自定义数据集类"""
#     def __init__(self, data_dir=JPG_DIR, metadata_file='metadata.jsonl'):
#         self.data_dir = data_dir
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])

#         self.data = []
#         metadata_path = os.path.join(data_dir, metadata_file)
#         with open(metadata_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image_path = os.path.join(self.data_dir, item['file_name'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         return {
#             'image': image,
#             'meaning': item['text']
#         }

# class KanjiTrainer:
#     def __init__(
#         self,
#         model_id="CompVis/stable-diffusion-v1-4",
#         learning_rate=1e-6,
#         batch_size=4,
#         num_epochs=10
#     ):
#         self.model_id = model_id
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def setup_model(self):
#         """初始化模型"""
#         self.pipeline = StableDiffusionPipeline.from_pretrained(
#             self.model_id,
#             torch_dtype=torch.float32,
#             safety_checker=None,
#             requires_safety_checker=False,
#             use_safetensors=True
#         ).to(self.device)

#         # 启用内存优化
#         self.pipeline.enable_attention_slicing(slice_size="max")
#         self.pipeline.enable_vae_slicing()

#     def setup_data(self):
#         """设置数据加载器"""
#         dataset = KanjiDataset()
#         self.dataloader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )

#     def setup_optimizer(self):
#         """设置优化器"""
#         self.optimizer = torch.optim.AdamW(
#             self.pipeline.unet.parameters(),
#             lr=self.learning_rate,
#             eps=1e-8
#         )

#     def train_step(self, batch):
#         """单步训练"""
#         images = batch["image"].to(self.device)
#         text = batch["meaning"]

#         with torch.autocast(device_type=self.device.type):
#             text_inputs = self.pipeline.tokenizer(
#                 text,
#                 padding="max_length",
#                 max_length=self.pipeline.tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             ).to(self.device)

#             text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]
#             latents = self.pipeline.vae.encode(images).latent_dist.sample()
#             latents = latents * self.pipeline.vae.config.scaling_factor

#             noise = torch.randn_like(latents)
#             timesteps = torch.randint(
#                 0,
#                 self.pipeline.scheduler.config.num_train_timesteps,
#                 (latents.shape[0],),
#                 device=self.device
#             )
#             noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

#             noise_pred = self.pipeline.unet(noisy_latents, timesteps, text_embeddings).sample
#             loss = F.mse_loss(noise_pred, noise)

#         return loss

#     def save_checkpoint(self, epoch):
#         """保存检查点"""
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}')
#         os.makedirs(checkpoint_path, exist_ok=True)
#         self.pipeline.save_pretrained(checkpoint_path)

#     def train(self):
#         """训练流程"""
#         self.setup_model()
#         self.setup_data()
#         self.setup_optimizer()

#         # 创建检查点目录
#         os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#         for epoch in range(self.num_epochs):
#             epoch_loss = 0
#             progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

#             for batch_idx, batch in enumerate(progress_bar):
#                 torch.cuda.empty_cache()

#                 loss = self.train_step(batch)
#                 loss.backward()

#                 torch.nn.utils.clip_grad_norm_(self.pipeline.unet.parameters(), max_norm=1.0)
#                 self.optimizer.step()
#                 self.optimizer.zero_grad(set_to_none=True)

#                 epoch_loss += loss.item()
#                 progress_bar.set_postfix({"loss": loss.item()})

#                 torch.cuda.empty_cache()

#             if (epoch + 1) % 5 == 0:
#                 self.save_checkpoint(epoch)

#             avg_loss = epoch_loss / len(self.dataloader)
#             print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# def main():
#     """主函数"""
#     try:
#         trainer = KanjiTrainer()
#         trainer.train()
#     except Exception as e:
#         print(f"训练过程中出现错误: {str(e)}")

# if __name__ == "__main__":
#     main()
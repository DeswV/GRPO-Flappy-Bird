import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import os
import json
import gymnasium as gym
import flappy_bird_gymnasium
from actor import FlappyBirdActor
from sampling import sample_flappy_bird


class PretrainingDataset(Dataset):
    def __init__(self, human_play_data_dir: str):
        super().__init__()
        data = []  # elem_i = (observation_i, action_i)
        for filename in os.listdir(human_play_data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(human_play_data_dir, filename), 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                data.append((game_data['init_obs'], game_data['trajectory'][0]['action']))
                for step in range(1, len(game_data['trajectory'])):
                    data.append(
                        (game_data['trajectory'][step - 1]['next_obs'], game_data['trajectory'][step]['action']))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation, action = self.data[idx]
        return torch.tensor(observation, dtype=torch.float), torch.tensor(action, dtype=torch.long)


def pretrain_flappy_bird_actor(human_play_data_dir: str,
                               output_dir: str,
                               steps: int,
                               peak_lr: float,
                               warmup_steps: int,
                               batch_size: int,
                               log_dir: str,
                               save_every_n_steps: int,
                               eval_every_n_steps: int,
                               eval_samples: int,
                               model_config: dict = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据与模型
    dataset = PretrainingDataset(human_play_data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(loader)
    model_config = model_config if model_config is not None else {}
    actor = FlappyBirdActor(**model_config).train().to(device)
    optimizer = AdamW(actor.parameters(), lr=peak_lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps)
    # 使用当前日期和时间作为log文件夹的名字
    log_name = time.strftime("%Y-%m-%d__%H-%M-%S")
    writer = SummaryWriter(os.path.join(log_dir, log_name))
    os.makedirs(output_dir, exist_ok=True)

    # 评估环境（无渲染）
    eval_env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

    for step in tqdm(range(1, steps + 1), desc='Pretraining', leave=True):
        # 获取一个训练 batch
        try:
            obs_batch, act_batch = next(data_iter)
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
        except StopIteration:
            data_iter = iter(loader)
            obs_batch, act_batch = next(data_iter)

        # 前向+损失+反向
        logits = actor(obs_batch)
        loss = torch.nn.functional.cross_entropy(logits, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录训练损失
        writer.add_scalar("pretrain/loss", loss.item(), step)
        # 记录学习率
        writer.add_scalar("pretrain/lr", scheduler.get_last_lr()[0], step)

        # 定期保存模型
        if step % save_every_n_steps == 0:
            path = os.path.join(output_dir, f"actor_pretrained_{step}.pt")
            actor.save_checkpoint(path)

        # 定期评估平均 reward
        if step % eval_every_n_steps == 0:
            print(f"Evaluating at step {step}...")
            avg_reward = sample_flappy_bird(actor, eval_env, rounds=eval_samples)['avg_reward']
            print(f"Average reward at step {step}: {avg_reward:.2f}")
            writer.add_scalar("pretrain/avg_reward", avg_reward, step)
            actor.train()

    # 资源清理
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    pretrain_flappy_bird_actor(
        human_play_data_dir="human_play_data",
        output_dir="outputs/pretrained_models/128-128-64_lr1e-3",
        steps=1000,
        peak_lr=1e-3,
        warmup_steps=100,
        batch_size=256,
        log_dir="logs/pretraining",
        save_every_n_steps=100,
        eval_every_n_steps=100,
        eval_samples=50,
        model_config={'hidden_dims': (128, 128, 64)}
    )

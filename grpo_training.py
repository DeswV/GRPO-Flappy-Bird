import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import os
import copy
import random
from actor import FlappyBirdActor
from sampling import sample_flappy_bird


class GRPOTrainingDataset(Dataset):
    def __init__(self, samples_from_all_groups: list[list]):
        """
        GRPO算法中采样后用于训练的数据集
        :param samples_from_all_groups: samples的数组，每个元素是来自一个Group的samples列表
        """
        super().__init__()
        self.data = []  # elem_i = [observation_i, action_i, old_prob_i, ref_prob_i, A_i]

        for group_samples in samples_from_all_groups:
            # 对于每个Group内的samples
            r_array = []  # 用于计算Group内reward的均值和标准差
            group_data = []  # 每个元素是每个sample的data数组

            for sample in group_samples:
                # 对于每个sample
                sample_data = []  # 这个sample的data数组

                r_array.extend([step['reward'] for step in sample['trajectory']])
                # A_i设为0，之后再计算
                sample_data.append([sample['init_obs'], sample['trajectory'][0]['action'],
                                    sample['trajectory'][0]['prob'], sample['trajectory'][0]['ref_prob'], 0.0])
                for i in range(1, len(sample['trajectory'])):
                    last_step = sample['trajectory'][i - 1]
                    step = sample['trajectory'][i]
                    sample_data.append(
                        [last_step['next_obs'], step['action'], step['prob'], step['ref_prob'], 0.0])

                group_data.append(sample_data)

            # 现在计算reward的均值和标准差
            r_mean = sum(r_array) / len(r_array)
            r_std = (sum((r - r_mean) ** 2 for r in r_array) / len(r_array)) ** 0.5

            # 现在来计算A_i
            for sample_idx, sample_data in enumerate(group_data):
                sample_r_array = [step['reward'] for step in group_samples[sample_idx]['trajectory']]
                sample_A_array = [0] * len(sample_data)

                # 从后向前计算每个A_i，A_i等于从这个时间步之后的每个step的normalized reward的和
                for i in range(len(sample_data) - 1, -1, -1):
                    if i == len(sample_data) - 1:
                        sample_A_array[i] = (sample_r_array[i] - r_mean) / r_std
                    else:
                        sample_A_array[i] = sample_A_array[i + 1] + (sample_r_array[i] - r_mean) / r_std

                # 将A_i存入sample_data
                for j in range(len(sample_data)):
                    sample_data[j][-1] = sample_A_array[j]

            # 将二维数组group_data展平并添加到self.data
            flatten_group_data = []
            for sample_data in group_data:
                flatten_group_data.extend(sample_data)
            self.data.extend(flatten_group_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation, action, old_prob, ref_prob, A = self.data[idx]
        return {
            'observation': torch.tensor(observation, dtype=torch.float),
            'action': torch.tensor(action, dtype=torch.long),
            'old_prob': torch.tensor(old_prob, dtype=torch.float),
            'ref_prob': torch.tensor(ref_prob, dtype=torch.float),
            'A': torch.tensor(A, dtype=torch.float)
        }


def calculate_grpo_loss(actor: FlappyBirdActor,
                        observations: torch.Tensor,
                        actions: torch.Tensor,
                        old_probs: torch.Tensor,
                        A: torch.Tensor,
                        clip_epsilon: float,
                        kl_coefficient: float,
                        ref_probs: torch.Tensor):
    logits = actor(observations)
    probs = F.softmax(logits, dim=-1)
    # 取actions对应的概率
    probs = torch.gather(probs, dim=-1, index=actions.unsqueeze(1)).squeeze(1)

    # 计算与Reference Model的KL散度
    D_kl = ref_probs / probs - torch.log(ref_probs / probs) - 1
    grpo_objective = (torch.min(
        (probs / old_probs) * A,
        torch.clamp(probs / old_probs, 1 - clip_epsilon, 1 + clip_epsilon) * A
    )
                      - kl_coefficient * D_kl)
    loss = -torch.mean(grpo_objective)
    return loss


def train_flappy_bird_actor_with_grpo(actor_checkpoint_path: str,
                                      output_dir: str,
                                      iterations: int,
                                      steps_per_iteration: int,
                                      num_groups: int,
                                      group_size: int,
                                      repeat_samples_n_times: int,
                                      batch_size: int,
                                      log_dir: str,
                                      learning_rate: float,
                                      save_every_n_steps: int,
                                      eval_every_n_steps: int = 1,
                                      clip_epsilon: float = 0.4,
                                      kl_coefficient: float = 0.04):
    """
    使用GRPO算法训练Flappy Bird Actor。
    注意到输入参数中的step是指在环境中进行采样然后若干次更新模型权重的一个过程，不是指单次更新模型权重
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    actor = FlappyBirdActor.load_checkpoint(actor_checkpoint_path).to(device)
    optimizer = AdamW(actor.parameters(), lr=learning_rate)

    os.makedirs(output_dir, exist_ok=True)
    log_name = time.strftime("%Y-%m-%d__%H-%M-%S")
    writer = SummaryWriter(os.path.join(log_dir, log_name))

    global_step = 0  # 这里的step是指在环境中进行采样然后若干次更新模型权重的一个过程
    global_training_step = 0  # # 这里的step是指单次更新模型权重

    for iteration in range(iterations):
        # Reference Model
        actor_ref = copy.deepcopy(actor).to(device)

        progress_bar = tqdm(total=steps_per_iteration, desc=f"Iteration {iteration + 1}/{iterations}", leave=True,
                            position=0)

        for step in range(steps_per_iteration):
            # 首先，在环境中进行采样
            samples_from_all_groups = []

            sampling_progress_bar = tqdm(total=num_groups, desc="Sampling", unit="group", leave=True, position=1)
            for _ in range(num_groups):
                # 每个Group内的环境使用相同的seed
                reset_seed = random.randint(0, 2 ** 31 - 1)

                group_samples = sample_flappy_bird(actor, env_reset_seed=reset_seed,
                                                   rounds=group_size, ref_actor=actor_ref)['samples']
                samples_from_all_groups.append(group_samples)

                sampling_progress_bar.update(1)
            sampling_progress_bar.close()

            # 使用采样的数据创建GRPO训练数据集
            dataset = GRPOTrainingDataset(samples_from_all_groups)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 然后，使用GRPO训练模型权重
            actor.train()
            update_progress_bar = tqdm(total=repeat_samples_n_times * len(data_loader),
                                       desc="Updating model weights", unit="train_step", leave=True, position=1)
            for _ in range(repeat_samples_n_times):
                for batch in data_loader:
                    observations = batch['observation'].to(device)
                    actions = batch['action'].to(device)
                    old_probs = batch['old_prob'].to(device)
                    ref_probs = batch['ref_prob'].to(device)
                    A = batch['A'].to(device)

                    # 计算GRPO损失
                    loss = calculate_grpo_loss(actor, observations, actions, old_probs, A,
                                               clip_epsilon, kl_coefficient, ref_probs)

                    # 更新模型权重
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 记录单步loss
                    writer.add_scalar('GRPO/current_loss', loss.item(), global_training_step)
                    global_training_step += 1
                    update_progress_bar.update(1)

            update_progress_bar.close()

            # 完整地完成了一个采样和更新过程
            progress_bar.update(1)

            # 定期保存模型
            if (global_step + 1) % save_every_n_steps == 0:
                path = os.path.join(output_dir, f"actor_grpo_{global_step}.pt")
                actor.save_checkpoint(path)

            # 定期评估平均 reward
            if (global_step + 1) % eval_every_n_steps == 0:
                avg_reward = sample_flappy_bird(actor, rounds=64, display=False)['avg_reward']
                writer.add_scalar("GRPO/avg_reward", avg_reward, global_step)
                print(f"Iteration {iteration + 1}, Step {global_step + 1}: Average Reward: {avg_reward:.2f}")

            global_step += 1

        progress_bar.close()

    print("Training completed.")


if __name__ == '__main__':
    train_flappy_bird_actor_with_grpo(
        actor_checkpoint_path='models/temp/actor_grpo_0.pt',
        output_dir='outputs/grpo_training_2',
        iterations=1,
        steps_per_iteration=20,
        num_groups=16,
        group_size=128,
        repeat_samples_n_times=1,
        batch_size=512,
        log_dir='logs/grpo_training',
        learning_rate=1e-5,
        save_every_n_steps=1,
        eval_every_n_steps=1,
        kl_coefficient=0.0
    )

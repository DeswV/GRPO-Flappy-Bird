import random
import time
import torch
import numpy as np
from collections import OrderedDict
import gymnasium as gym
import flappy_bird_gymnasium
import pygame
from configs import device
from actor import FlappyBirdActor


def _sample_flappy_bird_serial(actor: FlappyBirdActor, env_: gym.Env = None, env_reset_seed: int | None = None,
                               rounds: int = 5, display: bool = False, ref_actor: FlappyBirdActor = None):
    actor.eval()

    if env_ is None:
        env = gym.make("FlappyBird-v0", render_mode="human" if display else None, use_lidar=False)
    else:
        env = env_

    # 如果要显示，固定帧率为 30
    clock = None
    FPS = 30
    if display:
        clock = pygame.time.Clock()

    # 记录轨迹
    total_reward = 0.0
    samples = []

    for _ in range(rounds):
        obs, _ = env.reset(seed=env_reset_seed)
        sample = {'init_obs': list(obs), 'trajectory': []}

        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                action, probs = actor.get_action(obs)
            prob = probs[action]
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += r

            sample['trajectory'].append({
                'action': action,
                'prob': prob,
                'next_obs': list(obs),
                'reward': r
            })

            if ref_actor is not None:
                ref_actor.eval()
                with torch.no_grad():
                    _, ref_probs = actor.get_action(obs)
                    ref_prob = ref_probs[action]
                sample['trajectory'][-1]['ref_prob'] = ref_prob

            done = terminated or truncated
            if display:
                clock.tick(FPS)

        total_reward += ep_reward
        samples.append(sample)

    if env_ is None:
        env.close()

    return {
        'avg_reward': total_reward / rounds,
        'samples': samples
    }


def _sample_flappy_bird_parallel(actor: FlappyBirdActor, env_reset_seed: int | None = None, rounds: int = 5,
                                 num_envs: int = None, env_vectorization_mode: str = 'sync',
                                 ref_actor: FlappyBirdActor = None):
    envs = gym.make_vec("FlappyBird-v0", render_mode=None, use_lidar=False, num_envs=num_envs,
                        vectorization_mode=env_vectorization_mode)
    actor.eval()
    if ref_actor is not None:
        ref_actor.eval()

    # 记录轨迹
    total_reward = 0.0
    samples = []

    sampling_rounds = (rounds + num_envs - 1) // num_envs
    for round_ in range(sampling_rounds):
        batch_samples = []
        num_valid_envs = min(num_envs, rounds - round_ * num_envs)

        obs, _ = envs.reset(seed=env_reset_seed)
        for i in range(num_valid_envs):
            sample = {'init_obs': list(obs[i]), 'trajectory': []}
            batch_samples.append(sample)

        done = np.array([False] * num_valid_envs + [True] * (num_envs - num_valid_envs))
        while not np.all(done):
            # 使用actor采样
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(actor.output_layer.weight.device)
            with torch.no_grad():
                logits = actor(obs_tensor)
                probs = torch.softmax(logits, dim=1)
                actions = torch.multinomial(probs, 1).squeeze(1)
                # 取实际使用的action对应的概率
                probs = torch.gather(probs, 1, actions.unsqueeze(1)).squeeze(1)

                # 计算ref_actor采取对应actions的概率
                if ref_actor is not None:
                    ref_logits = ref_actor(obs_tensor)
                    ref_probs = torch.softmax(ref_logits, dim=1)
                    ref_probs = torch.gather(ref_probs, 1, actions.unsqueeze(1)).squeeze(1)

            obs, r, terminated, truncated, _ = envs.step(actions.cpu().numpy())

            # 更新轨迹的记录
            for i in range(num_valid_envs):
                if not done[i]:
                    total_reward += float(r[i])

                    batch_samples[i]['trajectory'].append({
                        'action': actions[i].item(),
                        'prob': probs[i].item(),
                        'next_obs': list(obs[i]),
                        'reward': float(r[i])
                    })

                    if ref_actor is not None:
                        batch_samples[i]['trajectory'][-1]['ref_prob'] = ref_probs[i].item()

            done = done | terminated | truncated

        samples.extend(batch_samples)

    return {
        'avg_reward': total_reward / rounds,
        'samples': samples
    }


def sample_flappy_bird(actor: FlappyBirdActor, env_: gym.Env = None, env_reset_seed: int | None = None, rounds: int = 5,
                       display: bool = False, num_envs: int | None = None, env_vectorization_mode: str = 'sync',
                       ref_actor: FlappyBirdActor = None):
    """
    使用给定的actor评估Flappy Bird游戏的表现，返回平均奖励和所有轨迹的记录。
    """
    if num_envs is None:
        num_envs = min(rounds, 128)

    if display or num_envs == 1:
        # 单进程执行
        return _sample_flappy_bird_serial(actor, env_, env_reset_seed, rounds, display, ref_actor)

    else:
        # 多进程执行
        return _sample_flappy_bird_parallel(actor, env_reset_seed, rounds, num_envs, env_vectorization_mode, ref_actor)


if __name__ == "__main__":
    checkpoint_path = 'models/actor_grpo.pt'

    actor = FlappyBirdActor.load_checkpoint(checkpoint_path).eval().to(device)

    start_time = time.time()
    avg_reward = sample_flappy_bird(actor, rounds=5, display=True)['avg_reward']
    end_time = time.time()

    print(f"Sampling took {end_time - start_time:.2f} seconds")
    print(f"Average reward: {avg_reward:.2f}")

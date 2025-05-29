import flappy_bird_gymnasium
import gymnasium as gym
import pygame
import json
import time
import os

"""
让人类游玩flappy bird游戏并记录数据，
以此使用监督学习初始化Agent。
"""


def log_game(output_dir:str, rounds: int = 5):
    """
    游玩flappy bird游戏并记录数据，结果保存为JSON文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 pygame
    pygame.init()
    clock = pygame.time.Clock()
    FPS = 30  # 固定帧率为 30

    # 创建环境
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    for _ in range(rounds):
        observation, info = env.reset()

        # 数据记录结构
        game_data = {
            "init_obs": list(observation),
            "trajectory": []
        }

        # 游戏主循环
        running = True
        while running:
            action = 0  # 默认不 flap

            # 监听事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1  # flap
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        break

            # 环境交互
            next_observation, reward, terminated, truncated, info = env.step(action)

            # print(f"action: {action}, type: {type(action)}")
            # print(f"next_observation: {next_observation}, type: {type(next_observation)}")
            # print(f"reward: {reward}, type: {type(reward)}")
            # 记录数据
            game_data["trajectory"].append({
                "action": action,
                "next_obs": list(next_observation),
                "reward": reward
            })

            # 结束检测
            if terminated or truncated:
                running = False

            # 控制帧率
            clock.tick(FPS)

        # 保存数据为 JSON 文件，使用当前时间作为文件名
        file_name = f"human_play_data_{int(time.time())}.json"
        with open(os.path.join(output_dir, file_name), "w", encoding='utf-8') as f:
            json.dump(game_data, f, indent=2)

        print(f"数据已保存为 {file_name}")

    # 清理
    env.close()
    pygame.quit()

if __name__ == "__main__":
    output_directory = "human_play_data"
    rounds_to_play = 5
    log_game(output_directory, rounds_to_play)
    print("游戏记录完成。")

import gymnasium as gym

def choose_action(observation):
    pos, v, ang, rot = observation
                                            # 根據竿子的角度和速度來決定移動方向
    if ang < -0.1 or (ang < 0 and v < 0):   #角度向左，或者角度微左且速度向左
        return 0                            # 向左移動
    elif ang > 0.1 or (ang > 0 and v > 0):  #角度向右，或者角度微右且速度向右
        return 1                            # 向右移動
    else:
        return 1 if pos < 0 else 0          # 如果角度接近0，根據位置移動

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = gym.make('CartPole-v1', render_mode="human") # 如果要顯示動畫，可以使用這行

    total_rewards = 0                       # 記錄總的獎勵
    episodes = 10                           # 設定運行的回合數

    for i_episode in range(episodes):
        observation, info = env.reset()     # 將每集的環境重設為初始狀態
        rewards = 0                         # 累積每集獎勵
        for t in range(250):
            env.render()

            action = choose_action(observation) #根據手工製定的規則選擇一個動作
            observation, reward, terminated, truncated, info = env.step(action) #執行獲得獎勵
            done = terminated or truncated
            rewards += reward

            if done:
                print(f'Episode {i_episode+1} finished after {t+1} timesteps, total rewards {rewards}')
                break

        total_rewards += rewards  # 加上每回合的總獎勵

    print(f'Average rewards over {episodes} episodes: {total_rewards / episodes}')
    env.close() 

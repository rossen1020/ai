import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") # 若改用這個，會畫圖
env = gym.make("CartPole-v1", render_mode="rgb_array")
score = 0
max_score = 0
observation, info = env.reset(seed=42)
for _ in range(1000):
   
   env.render()
   
   if observation[2] < 0:
      if observation[1] > 0:
         action = 0  
      else:
         action = 1 
   else:
      if observation[1] <0:
         action = 1 
      else:
         action = 0  
        
         
   '''if observation[2] < 0:
      action = 0  
   else:
      action = 0 
   '''
   observation, reward, terminated, truncated, info = env.step(action)
   
   score += reward
   
   print('observation=', observation)
   print(score)
   if terminated or truncated:
      observation, info = env.reset()
      max_score = max(max_score, score)  # 更新最高分
      print('max_score=', max_score)
      print('done')
      score= 0
print('max_score=', max_score)
env.close()

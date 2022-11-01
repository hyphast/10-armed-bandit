import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from rlglue.rl_glue import RLGlue
import modules.main_agent as main_agent
import modules.ten_arm_env as ten_arm_env
import modules.test_env as test_env

def argmax(q_values):
  # Принимает список q_values ​​и возвращает индекс элемента
  # с наибольшим значением. Разрывает связи случайным образом.
  
  # возвращает: int - индекс самого высокого значения в q_values

  top_value = float("-inf")
  ties = []
  
  for i in range(len(q_values)):
    # Если ценность в q_values ​​больше, чем наивысшая, обновить top_value и сбросить ties в ноль
    # если ценность равна top_value, добавить индекс к ties
    # вернуть случайно выбранный индекс из ties.
    # ВАШ КОД ЗДЕСЬ
    if (q_values[i] > top_value):
      top_value = q_values[i]
      ties = []
      ties.append(i)
    elif (q_values[i] == top_value):
      ties.append(i)

  return np.random.choice(ties)

class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        # Takes one step for the agent. It takes in a reward and observation and 
        # returns the action the agent chooses at that time step.
        
        # Arguments:
        # reward -- float, the reward the agent recieved from the environment after taking the last action.
        # observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
        # until future lessons
        # Returns:
        # current_action -- int, the action chosen by the agent at the current time step.

        
        ### Useful Class Variables ###
        # self.q_values : Массив, содержащий, по мнению агента, ценности каждой руки.
        # self.arm_count : Массив с подсчетом количества выбора каждой руки.
        # self.last_action : Действие, которое агент предпринял на предыдущем шаге времени.
        # self.epsilon : Вероятность того, что эпсилон-жадный агент  будет разведывать (колеблется от 0 до 1)
        #######################
        
        # Обновление ценностей Q - это должно быть то же обновление, что и у вашего жадного агента выше
        # ВАШ КОД ЗДЕСЬ
        random_num = np.random.random()

        current_action = -1
        if (random_num < self.epsilon):
          current_action = np.random.randint(0, len(self.q_values))
        else:
          current_action=argmax(self.q_values)

        self.arm_count[current_action] += 1
        step_count = np.sum(self.arm_count)
        self.q_values[self.last_action] = reward / step_count

        # Выбрать действие, используя эпсилон-жадность
        # Случайно выбрать число от 0 до 1 и посмотреть, не меньше ли оно, чем self.epsilon
        # (подсказка: загляните в np.random.random()). Если это так, установите для current_action случайное действие.
        # в противном случае жадно выбирайте current_action, как делали выше.

        self.last_action = current_action

        return current_action

# Рисует результаты Epsilon-жадных и жадных
num_runs = 200
num_steps = 1000
epsilon = 0.1
agent = EpsilonGreedyAgent
env = ten_arm_env.Environment
agent_info = {"num_actions": 10, "epsilon": epsilon}
env_info = {}
all_averages = []

for run in tqdm(range(num_runs)):
    np.random.seed(run)
    
    rl_glue = RLGlue(env, agent)
    rl_glue.rl_init(agent_info, env_info)
    rl_glue.rl_start()

    scores = [0]
    averages = []
    for i in range(num_steps):
        reward, _, action, _ = rl_glue.rl_step() # Среда и агент делают шаг и возвращают
                                                 # награду и выбранное действие.
        scores.append(scores[-1] + reward)
        averages.append(scores[-1] / (i + 1))
    all_averages.append(averages)

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
# plt.plot(greedy_scores)
plt.title("сравнение средних наград епсилон-жадого и жадного")
plt.plot(np.mean(all_averages, axis=0))
plt.legend(("Наилучшее из возможных", "Жадный", "Эпсилон: 0.1"))
plt.xlabel("Шаги")
plt.ylabel("Средняя награда")
plt.show()
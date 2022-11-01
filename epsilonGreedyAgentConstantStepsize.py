import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from rlglue.rl_glue import RLGlue
import modules.main_agent as main_agent
import modules.ten_arm_env as ten_arm_env
import modules.test_env as test_env

import epsilonGreedyAgent

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

class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Делает один шаг за агента. Получает награду и разведку и
        возвращает действие, которое агент выбирает на этом шаге времени.
        
        Аргументы:
        reward - float, награда, полученная агентом от окружения после выполнения последнего действия.
        observation - float, наблюдаемое состояние, в котором находится агент. Не беспокойтесь об этом, так как вы не будете его использовать
        до будущих уроков
        Возврат:
        current_action - int, действие, выбранное агентом на текущем временном шаге.
        """
        
        ### Useful Class Variables ###
        # self.q_values: массив, содержащий, по мнению агента, ценности каждой руки.
        # self.arm_count: массив со счетчиком количества выбора каждой руки.
        # self.last_action: int, действие, которое агент выполнил на предыдущем временном шаге.
        # self.step_size: float, текущий размер шага агента.
        # self.epsilon: вероятность разведки эпсилон-жадного агента(от 0 до 1)
        #######################
        
        random_num = np.random.random()

        current_action = -1
        if (random_num < self.epsilon):
          current_action = np.random.randint(0, len(self.q_values))
        else:
          current_action=argmax(self.q_values)        

        self.arm_count[current_action] += 1
        self.q_values[self.last_action] = reward / self.step_size

        # Обновить q_values для действий, предпринятых на предыдущем временном шаге
        # используя self.step_size вместо self.arm_count
        # ВАШ КОД ЗДЕСЬ
        # raise NotImplementedError()
        
        # Выберите действие с эпсилон-жадным. Это то же самое, что вы реализовали выше.
        # ВАШ КОД ЗДЕСЬ
        # raise NotImplementedError()
        
        self.last_action = current_action
        
        return current_action

step_sizes = [0.01, 0.1, 0.5, 1.0, '1/N(A)']

epsilon = 0.1
num_steps = 1000
num_runs = 200
env = ten_arm_env.Environment

fig, ax = plt.subplots(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')

q_values = {step_size: [] for step_size in step_sizes}
true_values = {step_size: None for step_size in step_sizes}
best_actions = {step_size: [] for step_size in step_sizes}

for step_size in step_sizes:
    all_averages = []
    for run in tqdm(range(num_runs)):
        np.random.seed(run)
        agent = EpsilonGreedyAgentConstantStepsize if step_size != '1/N(A)' else epsilonGreedyAgent.EpsilonGreedyAgent
        agent_info = {"num_actions": 10, "epsilon": epsilon, "step_size": step_size, "initial_value": 0.0}
        env_info = {}

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        
        best_arm = np.argmax(rl_glue.environment.arms)

        scores = [0]
        averages = []
        
        if run == 0:
            true_values[step_size] = np.copy(rl_glue.environment.arms)
            
        best_action_chosen = []
        for i in range(num_steps):
            reward, state, action, is_terminal = rl_glue.rl_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
            if action == best_arm:
                best_action_chosen.append(1)
            else:
                best_action_chosen.append(0)
            if run == 0:
                q_values[step_size].append(np.copy(rl_glue.agent.q_values))
        best_actions[step_size].append(best_action_chosen)
    ax.plot(np.mean(best_actions[step_size], axis=0))

plt.legend(step_sizes)
plt.title("% раз выбрано лучшее действие")
plt.xlabel("Шаги")
plt.ylabel("% раз выбрано лучшее действие")
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
plt.show()
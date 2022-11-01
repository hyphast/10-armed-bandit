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

class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        # Один шаг агента. Входные параметры: награда, наблюдения
        # возвращает действие, которое агент выбирает на этом временном шаге.
        
        # Аргументы:
        # reward -- float, награда, которую агент получил от окружения после выполнения последнего действия.
        # observation -- float, наблюдаемое состояние, в котором находится агент. 
        # Не беспокойтесь об этом, так как вы не будете его использовать до будущих уроков
        # Возвращаемое значение:
        # current_action -- int, действие, выбранное агентом на текущем временном шаге.

        ### Переменные класса ###
        # self.q_values : Массив, содержащий, по мнению агента, все ценности действий (рук).
        # self.arm_count : Массив со счетчиком количества опускания каждой руки
        # self.last_action : Действие, которое агент совершил на предыдущем временном шаге.
        #######################
        
        # Update Q values Обновление значения Q 
        # Подсказка: посмотрите алгоритм в разделе "Инкрементная реализация" лекции "Многорукий бандит"
        # увеличить счетчик в self.arm_count для действия с предыдущего шага времени
        # обновление размера шага с использованием self.arm_count
        # обновление self.q_values для действия с предыдущего шага времени
        
        # ВАШ КОД ЗДЕСЬ
        current_action=argmax(self.q_values)
        print("Делаем действие")
        print(current_action)


        print("Увеличиваем счетчик")
        self.arm_count[current_action] += 1
        print(self.arm_count)
        step_count = np.sum(self.arm_count)
        print(step_count)

        self.q_values[self.last_action] = reward / step_count
    
        self.last_action = current_action
        
        return current_action

num_runs = 200                    # количество запусков эксперимента
num_steps = 1000                  # Сколько раз агентом выбрана каждая рука
env = ten_arm_env.Environment     # Устанавливаем, какую среду мы хотим использовать для тестирования
agent = GreedyAgent               # Выбираем, какого агента мы хотим использовать
agent_info = {"num_actions": 10}  # Передаем агенту необходимую информацию. (Здесь - сколько всего рук).
env_info = {}                     # Передаем среде необходимую информацию. (В этом случае - ничего не передаем)

all_averages = []

average_best = 0
for run in tqdm(range(num_runs)):           # tqdm - создает индикатор выполнения
    np.random.seed(run)
    
    rl_glue = RLGlue(env, agent)          # Создает новый эксперимент RLGlue с окружением и агентом, которые мы выбрали выше.
    rl_glue.rl_init(agent_info, env_info) # передаем RLGlue все, что нужно для инициализации агента и среды.
    rl_glue.rl_start()                    # запускаем эксперимент

    average_best += np.max(rl_glue.environment.arms)
    
    scores = [0]
    averages = []
    
    for i in range(num_steps):
        reward, _, action, _ = rl_glue.rl_step() # Среда и агент делают шаг и возвращают
                                                 # награду и выбранное действие.
        scores.append(scores[-1] + reward)
        averages.append(scores[-1] / (i + 1))
    all_averages.append(averages)

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([average_best / num_runs for _ in range(num_steps)], linestyle="--")
plt.plot(np.mean(all_averages, axis=0))
plt.legend(["Среда и агент делают шаг и возвращаются", "Жадный"])
plt.title("Средняя награда жадного агента")
plt.xlabel("Шаги")
plt.ylabel("средняя награда")
plt.show()
greedy_scores = np.mean(all_averages, axis=0)
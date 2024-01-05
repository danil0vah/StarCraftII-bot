from sc2 import maps
from sc2.main import run_game
from sc2.data import  Race, Difficulty
from sc2.player import Bot, Computer
from VoidRayRush import VoidRayRushBot

import os
import multiprocessing

def run_match(map_name, bot1, bot2):
    run_game(maps.get(map_name), [bot1, bot2], realtime=False)

if __name__ == "__main__":
    # Получить количество потоков процессора
    cores = multiprocessing.cpu_count() / 2

    # Создать пул потоков
    pool = multiprocessing.Pool(cores)

    # Создать список матчей
    matches = []
    for _ in range(cores):
        matches.append((
            "AbyssalReefLE",
            Bot(Race.Protoss, VoidRayRushBot()),
            Computer(Race.Terran, Difficulty.Hard)
        ))

    # Запустить матчи в пуле
    results = []
    while len(os.listdir("stuff/train_data")) < 15:
        for match in matches:
            result = pool.map_async(run_match, [match])
            results.append(result)

        # Подождать завершения всех матчей
        for result in results:
            result.get()
    
    
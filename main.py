import sc2
from sc2 import maps,position
from sc2.bot_ai import BotAI
from sc2.main import run_game, Result
from sc2.data import  Race, Difficulty
from sc2.player import Bot, Computer
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
import random

import cv2
import numpy as np
import time

HEADLESS = True

class SentdeBot(BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 60
        self.do_something_after = 0
        self.train_data = []


    async def on_end(self, result):
        if result == Result.Victory:
            np.save(f'stuff/train_data/{str(int(time.time()))}.npy', np.array(self.train_data, dtype="object"))

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()        
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()
        await self.intel()
    
    def random_location_variance(self, enemy_start_location):
        x, y = enemy_start_location

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        x = self.game_info.map_size[0] if x > self.game_info.map_size[0] else x
        y = self.game_info.map_size[1] if y > self.game_info.map_size[1] else y

        
        return position.Point2(position.Pointlike((x,y)))

    async def scout(self):
        if len(self.units(UnitTypeId.OBSERVER)) > 0:
            observer = self.units(UnitTypeId.OBSERVER)[0]
            if observer.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                observer.move(move_to)
        else:
            if len(self.structures(UnitTypeId.ROBOTICSFACILITY)) > 0:
                for rb in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                        rb.train(UnitTypeId.OBSERVER)


    async def intel(self):

        draw_dict = {
            UnitTypeId.NEXUS: [10, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (30, 255, 0)],
            UnitTypeId.PROBE: [1, (55, 255, 0)],

            UnitTypeId.ASSIMILATOR: [3, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 200, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 200, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [3, (255, 150, 0)],
            UnitTypeId.ROBOTICSFACILITY: [3, (255, 225, 100)]
        }
        main_base_list = ['nexus', 'supplydepot', 'hatchery']
        workers_list = ['probe', 'scv', 'drone']
        max_line = 50

        map_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for TYPE in draw_dict:
            for UNIT in self.units(TYPE).ready:
                unit_position = UNIT.position
                cv2.circle(img=map_data, center=(int(unit_position[0]), int(unit_position[1])),
                           radius=draw_dict[TYPE][0], color=draw_dict[TYPE][1], thickness=-1)
                
        
        for enemy_building in self.enemy_structures:
            structure_position = enemy_building.position
            if enemy_building.name.lower() in main_base_list:
                cv2.circle(img=map_data, center=(int(structure_position[0]), int(structure_position[1])), 
                                                 radius=10, color=(0,200,255), thickness=-1)
            else:
                cv2.circle(img=map_data, center=(int(structure_position[0]), int(structure_position[1])), 
                                                 radius=3, color=(0,150,255), thickness=-1)
        
        for enemy_unit in self.enemy_units:
            if not enemy_unit.is_structure:
                unit_position = enemy_unit.position
                if enemy_unit.name.lower() in workers_list:
                    cv2.circle(img=map_data, center=(int(unit_position[0]), int(unit_position[1])), 
                                                    radius=1, color=(50,0,255), thickness=-1)
                else:
                    cv2.circle(img=map_data, center=(int(unit_position[0]), int(unit_position[1])), 
                                                    radius=1, color=(100,0,255), thickness=-1)
                
        for observer in self.units(UnitTypeId.OBSERVER).ready:
            obs_position = observer.position
            cv2.circle(img=map_data, center=(int(obs_position[0]), int(obs_position[1])),
                           radius=3, color=(255, 225, 200), thickness=-1)
            
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0
        vespene_ratio = self.vespene/ 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0
        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(img=map_data, pt1=(0, 19), pt2=(int(max_line*military_weight), 19), color=(250,250,200), thickness=3)
        cv2.line(img=map_data, pt1=(0, 15), pt2=(int(max_line*plausible_supply), 15), color=(220,220,200), thickness=3)
        cv2.line(img=map_data, pt1=(0, 11), pt2=(int(max_line*population_ratio), 11), color=(170,150,150), thickness=3)
        cv2.line(img=map_data, pt1=(0, 7), pt2=(int(max_line*vespene_ratio), 7), color=(220,150,100), thickness=3)
        cv2.line(img=map_data, pt1=(0, 3), pt2=(int(max_line*mineral_ratio), 3), color=(150,100,250), thickness=3)

        
        
        self.flipped = cv2.flip(map_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_workers(self):
        nexus = self.townhalls.ready.random
        if self.workers.amount < self.townhalls.amount * 18:
            if self.workers.amount < self.MAX_WORKERS:
                for nexus in self.townhalls.ready.idle:
                    if self.can_afford(UnitTypeId.PROBE):
                        nexus.train(UnitTypeId.PROBE)

    async def build_pylons(self):
        nexus = self.townhalls.ready.random
        if self.supply_left < 5 and self.already_pending(UnitTypeId.PYLON) == 0:
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus)

    async def build_assimilators(self):
        for nexus in self.townhalls.ready:
            vespene = self.vespene_geyser.closer_than(15, nexus)
            for vespene in vespene:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.gas_buildings or not self.gas_buildings.closer_than(1, vespene):
                    worker.build_gas(vespene)
                    worker.stop(queue=True)


    async def expand(self):
        if self.units(UnitTypeId.NEXUS).amount * 1.5 < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(UnitTypeId.NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random

            if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

            elif len(self.structures(UnitTypeId.GATEWAY)) < 1:
                if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                    await self.build(UnitTypeId.GATEWAY, near=pylon)

            if len(self.structures(UnitTypeId.ROBOTICSFACILITY)) < 1:
                worker = self.workers.random
                abilities = await self.get_available_abilities(worker)
                if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and self.already_pending(UnitTypeId.ROBOTICSFACILITY) == 0:
                    if AbilityId.PROTOSSBUILD_ROBOTICSFACILITY in abilities:
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)

            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.structures(UnitTypeId.STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(UnitTypeId.STARGATE) and self.already_pending(UnitTypeId.STARGATE) == 0:
                        await self.build(UnitTypeId.STARGATE, near=pylon)

    async def build_offensive_force(self):
        
        for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                sg.train(UnitTypeId.VOIDRAY)

    def find_target(self, state):
        if len(self.enemy_units) > 0:
            return random.choice(self.enemy_units)
        elif len(self.enemy_structures) > 0:
            return random.choice(self.enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        unit_refs = {UnitTypeId.VOIDRAY: [8, 3]}

        if len(self.units(UnitTypeId.VOIDRAY).idle) > 0:
            choice = random.randrange(0, 4)
            target = False

            if self.iteration > self.do_something_after:
                if choice == 0:
                    wait = random.randrange(20,165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    if len(self.enemy_units) > 0:
                        target = self.enemy_units.closest_to(random.choice(self.structures(UnitTypeId.NEXUS)))

                elif choice == 2:
                    if len(self.enemy_structures) > 0:
                        target = random.choice(self.enemy_structures)
                
                elif choice == 3:
                    target = self.enemy_start_locations[0]

                if target:
                    for voidray in self.units(UnitTypeId.VOIDRAY).idle:
                        voidray.attack(target)
                
                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y, self.flipped])

        
        for UNIT in unit_refs:
            if self.units(UNIT).amount > unit_refs[UNIT][0]:
                for unemp in self.units(UNIT).idle:
                    unemp.attack(self.find_target(self.state))

            elif self.units(UNIT).amount > unit_refs[UNIT][1]:
                if len(self.enemy_units) > 0:
                    for unemp in self.units(UNIT).idle:
                        unemp.attack(random.choice(self.enemy_units))



run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime=False)
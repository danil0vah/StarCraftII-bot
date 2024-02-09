import sc2
from sc2 import maps,position
from sc2.bot_ai import BotAI, Units
from sc2.main import run_game, Result
from sc2.data import  Race, Difficulty
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
import random

#import keras
import cv2
import numpy as np
import time
import math

class VoidRayRushBot(BotAI):
    def __init__(self, use_model = False, title = 1, HEADLESS = False):
        self.HEADLESS = HEADLESS
        self.MAX_WORKERS = 60
        self.do_something_after = 0
        self.title = title
        self.use_model = use_model
        self.train_data = []
        self.scouts_and_spots = {}
        self.expand_dis_dir = {}
        self.choices = {
            0 : self.build_worker,
            1 : self.build_scout,
            2 : self.build_zealot,
            3 : self.build_stalker,
            4 : self.build_voidray,
            5 : self.build_stargate,
            6 : self.build_gateway,
            7 : self.build_pylon,
            8 : self.build_assimilator,
            9 : self.defend_nexus,
            10 : self.attack_known_enemy_unit,
            11 : self.attack_known_enemy_structure,
            12 : self.expand,
            13 : self.do_nothing
        }
        
        if self.use_model:
            print("USING MODEL!")
            self.model = None #keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")

    async def on_end(self, result):
        if result == Result.Victory:
            np.save(f'stuff/train_data/{str(int(time.time()))}.npy', np.array(self.train_data, dtype="object"))

    async def on_step(self, iteration):
        self.currently_time = self.state.game_loop / 22.4 / 60
        await self.scout()        
        await self.distribute_workers()
        await self.intel()
        wait = 10
        self.do_something_after = self.currently_time + wait
        await self.do_something()
        
 
    async def do_something(self):
        if self.time > self.do_something_after:
            if self.use_model:
                worker_weight = 1
                zealot_weight = 1
                stalker_weight = 1
                voidray_weight = 1
                stargate_weight = 1
                gateway_weight = 1
                pylon_weight = 1
                assimilator_weight = 1

                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 1])])
                weights = [worker_weight, 1, zealot_weight, stalker_weight, voidray_weight, stargate_weight, gateway_weight, pylon_weight, assimilator_weight,   1, 1, 1, 1, 1]
                weighted_prediction = prediction[0]*weights
                choice = np.argmax(weighted_prediction)
            else:
                worker_weight = 8
                zealot_weight = 3
                voidray_weight = 25
                stalker_weight = 8
                pylon_weight = 1
                stargate_weight = 5
                gateway_weight = 2
                expand_weight = 30
                
                choice_weights = (worker_weight*[0]+ 
                                  1*[1]+ #scout
                                  zealot_weight*[2]+
                                  stalker_weight*[3]+
                                  voidray_weight*[4]+
                                  stargate_weight*[5]+
                                  gateway_weight*[6]+
                                  pylon_weight*[7]+
                                  1*[8]+ #assimilator
                                  1*[9]+ #defend_nexus
                                  1*[10]+ #attack_known_enemy_unit
                                  1*[11]+ #attack_known_enemy_structure
                                  expand_weight*[12]+
                                  1*[13]) #do_nothing
                choice = random.choice(choice_weights)
            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e), '  -----  EXCEPTION ---- ', choice)
            
            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped])


    async def build_worker(self):
        if self.townhalls.ready.idle:
            if self.can_afford(UnitTypeId.PROBE):
                nexus = self.townhalls.ready.idle.random
                nexus.train(UnitTypeId.PROBE)


    async def build_scout(self):
        if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.exists:
            for rf in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    rf.train(UnitTypeId.OBSERVER)
                    break
        else:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if self.structures(UnitTypeId.PYLON).ready.exists:
                    if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(UnitTypeId.ROBOTICSFACILITY):
                        pylon = self.structures(UnitTypeId.PYLON).ready.random
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
        
    async def build_zealot(self):
        if self.structures(UnitTypeId.GATEWAY).ready.idle:
            if self.can_afford(UnitTypeId.ZEALOT):
                gateway = self.structures(UnitTypeId.GATEWAY).ready.idle.random
                gateway.train(UnitTypeId.ZEALOT)



    async def build_stalker(self):
        if self.structures(UnitTypeId.GATEWAY).ready.idle:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if self.can_afford(UnitTypeId.STALKER):
                    gateway = self.structures(UnitTypeId.GATEWAY).ready.idle.random
                    gateway.train(UnitTypeId.STALKER)
            else:
                await self.build_cyberneticscore()

    
    async def build_voidray(self):
        if self.structures(UnitTypeId.STARGATE).ready.exists:
            if self.can_afford(UnitTypeId.VOIDRAY):
                stargate = self.structures(UnitTypeId.STARGATE).ready.random
                stargate.train(UnitTypeId.VOIDRAY)


    async def build_stargate(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    await self.build(UnitTypeId.STARGATE, near=pylon)
            else:
                await self.build_cyberneticscore()


    async def build_gateway(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                await self.build(UnitTypeId.GATEWAY, near=pylon)

            
            
    async def build_cyberneticscore(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
            
    async def build_pylon(self):
        if self.townhalls.ready:
            if self.structures(UnitTypeId.PYLON).ready.exists:
                if self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                    pylon = self.structures(UnitTypeId.PYLON).ready.random
                    await self.build(UnitTypeId.PYLON, near=pylon.position.towards_with_random_angle(self.game_info.map_center, distance=5))
            else:
                if self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                    nexus = self.townhalls.ready.random
                    await self.build(UnitTypeId.PYLON, near=nexus.position.towards(self.game_info.map_center, 5))
    
                
    async def build_assimilator(self):
        for nexus in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(15, nexus)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.gas_buildings or not self.gas_buildings.closer_than(1, vespene):
                    worker.build_gas(vespene)
                    worker.stop(queue=True)
                    

    async def defend_nexus(self):
        if len(self.enemy_units) > 0:
            target = self.enemy_units.closest_to(random.choice(self.structures(UnitTypeId.NEXUS)))
            await self.attack_something_target(target)
            
            
    async def attack_known_enemy_unit(self):
        if len(self.enemy_units) > 0:
            target = self.enemy_units.random
            await self.attack_something_target(target)
            
            
    async def attack_known_enemy_structure(self):
        if len(self.enemy_structures) > 0:
            target = self.enemy_structures.random
            await self.attack_something_target(target)
            
            
    async def expand(self):
        if self.get_next_expansion:
            if self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
                    await self.expand_now()
    
    
    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.currently_time + wait
    
            
    async def attack_something_target(self, target):
        for unit in self.units(UnitTypeId.ZEALOT).idle:
            unit.attack(target)
        for unit in self.units(UnitTypeId.STALKER).idle:
            unit.attack(target)
        for unit in self.units(UnitTypeId.VOIDRAY).idle:
            unit.attack(target)
        
    async def random_location_variance(self, location):
        x, y = location

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        x = self.game_info.map_size[0] if x > self.game_info.map_size[0] else x
        y = self.game_info.map_size[1] if y > self.game_info.map_size[1] else y

        
        return position.Point2(position.Pointlike((x,y)))


    async def scout(self):
        if len(self.expand_dis_dir) == 0:
            for expansion_location in self.expansion_locations:
                distance_to_enemy_start = expansion_location.distance_to(self.enemy_start_locations[0])
                #add to dict
                self.expand_dis_dir[distance_to_enemy_start] = expansion_location
            self.ordered_expansion_distances = sorted(k for k in self.expand_dis_dir)
            print(self.ordered_expansion_distances)
        # removing of scouts that are actually dead now.
        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)           
        for scout in to_be_removed:
            del self.scouts_and_spots[scout]
        if len(self.structures(UnitTypeId.ROBOTICSFACILITY).ready) == 0:
            unit_type = UnitTypeId.PROBE
            unit_limit = 1
        else:
            unit_type = UnitTypeId.OBSERVER
            unit_limit = 15

        assign_scout = True
        if unit_type == UnitTypeId.PROBE:
            for unit in self.units(unit_type):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False
        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for observer in self.units(unit_type).idle[:unit_limit]:
                    if observer.tag not in self.scouts_and_spots:
                        for distance in self.ordered_expansion_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == distance)
                                active_locations = [self.scouts_and_spots[key] for key in self.scouts_and_spots]
                                if location not in active_locations:
                                    if unit_type == UnitTypeId.PROBE:
                                        for unit in self.units(unit_type):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    observer.move(location)
                                    self.scouts_and_spots[observer.tag] = location
                                    break
                            except:
                                pass
        for observer in self.units(unit_type):
            if observer.tag in self.scouts_and_spots:
                if observer in [probe for probe in self.units(UnitTypeId.PROBE)]:
                    observer.move(await self.random_location_variance(self.scouts_and_spots[observer.tag]))


    async def intel(self):

        map_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        


        for unit in self.units.ready:
            pos = unit.position
            cv2.circle(img=map_data, center=(int(pos[0]), int(pos[1])), radius=int(unit.radius*8), color=(255,255,255), thickness=math.ceil(int(unit.radius*0.5)))
        for unit in self.structures.ready:
            pos = unit.position
            cv2.circle(img=map_data, center=(int(pos[0]), int(pos[1])), radius=int(unit.radius*8), color=(255,255,255), thickness=math.ceil(int(unit.radius*0.5)))
                
        for unit in self.enemy_units.ready:
            pos = unit.position
            cv2.circle(img=map_data, center=(int(pos[0]), int(pos[1])), radius=int(unit.radius*8), color=(255,255,255), thickness=math.ceil(int(unit.radius*0.5)))
        for unit in self.enemy_structures.ready:
            pos = unit.position
            cv2.circle(img=map_data, center=(int(pos[0]), int(pos[1])), radius=int(unit.radius*8), color=(255,255,255), thickness=math.ceil(int(unit.radius*0.5)))
        
        max_line= 50    
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0
        vespene_ratio = self.vespene/ 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0
        if self.supply_cap !=0:
            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0
        else:
            population_ratio = 0

        plausible_supply = self.supply_cap / 200.0

        if self.supply_cap-self.supply_left != 0:
            military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap-self.supply_left)
            if military_weight > 1.0:
                military_weight = 1.0
        else:
            military_weight = 0
            
        worker_weight = len(self.units(UnitTypeId.PROBE)) / (self.supply_cap - self.supply_left)
        if worker_weight > 1.0:
            worker_weight = 1.0
            
        draw_data = {
            'military_weight' :  [(0, 23), (int(max_line*military_weight), 23), (250,250,200), 3],
            'worker_weight' :    [(0, 19), (int(max_line*military_weight), 19), (220,220,200),  3],
            'plausible_supply' : [(0, 15), (int(max_line*military_weight), 15), (170,150,150) , 3],
            'population_ratio' : [(0, 11), (int(max_line*military_weight), 11), (220,150,100) , 3],
            'vespene_ratio' :    [(0, 7), (int(max_line*military_weight), 7), (220,100,50), 3],
            'mineral_ratio' :    [(0, 3), (int(max_line*military_weight), 3), (150,100,250), 3]
            
        }
            
        for item in list(draw_data.values())[1:]:
            cv2.line(img=map_data, pt1=item[0], pt2=item[1], color=item[2], thickness=item[3])

        grayed = cv2.cvtColor(map_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not self.HEADLESS:
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)


if __name__=='__main__':
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, VoidRayRushBot()),
        Computer(Race.Terran, Difficulty.Easy)
        ], realtime = False)
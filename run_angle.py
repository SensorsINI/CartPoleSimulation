import yaml
from time import sleep
from math import pi
from run_data_generator import run_data_generator
import gc

import cProfile
import pstats
from memory_profiler import profile

file = "config_data_gen.yml"

def edit_config(angle):
    f = open("config_data_gen.yml", 'r')
    config = yaml.safe_load(f)
    f.close()

    config["random_initial_state"]["angle"] = angle

    f = open("config_data_gen.yml", 'w')
    yaml.safe_dump(config, f)
    f.close()

def main():
    for a in range(50, 175, 5):
        print(a)
        a = a*pi/180.0
        edit_config(a)

        run_data_generator()

        collected = gc.collect()
        print("GC collected: {}".format(collected))

        sleep(5)

if __name__ == "__main__":
    main()
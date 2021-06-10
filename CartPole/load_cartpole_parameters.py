from types import SimpleNamespace
import csv

def load_cartpole_parameters(dataset_path):
    p = SimpleNamespace()

    # region Get information about the pretrained network from the associated txt file
    with open(dataset_path) as f:
        reader = csv.reader(f)
        updated_features = 0
        for line in reader:
            line = line[0]
            if line[:len('# m: ')] == '# m: ':
                p.m = float(line[len('# m: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# M: ')] == '# M: ':
                p.M = float(line[len('# M: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# L: ')] == '# L: ':
                p.L = float(line[len('# L: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# u_max: ')] == '# u_max: ':
                p.u_max = float(line[len('# u_max: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# M_fric: ')] == '# M_fric: ':
                p.M_fric = float(line[len('# M_fric: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# J_fric: ')] == '# J_fric: ':
                p.J_fric = float(line[len('# J_fric: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# v_max: ')] == '# v_max: ':
                p.v_max = float(line[len('# v_max: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# TrackHalfLength: ')] == '# TrackHalfLength: ':
                p.TrackHalfLength = float(line[len('# TrackHalfLength: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# controlDisturbance: ')] == '# controlDisturbance: ':
                p.controlDisturbance = float(line[len('# controlDisturbance: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# sensorNoise: ')] == '# sensorNoise: ':
                p.sensorNoise = float(line[len('# sensorNoise: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# g: ')] == '# g: ':
                p.g = float(line[len('# g: '):].rstrip("\n"))
                updated_features += 1
                continue
            if line[:len('# k: ')] == '# k: ':
                p.k = float(line[len('# k: '):].rstrip("\n"))
                updated_features += 1
                continue

            if updated_features == 12:
                break

    return p

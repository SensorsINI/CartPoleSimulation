import pandas as pd
import matplotlib.pyplot as plt

#Path to Recording
path = "Experiments/CP_neural-imitator-tfmppi_2023-03-09_12-13-48.csv"
start_time = 0
end_time = 20

def plot_data(path, value, target_value, start_time, end_time):
    recording = pd.read_csv(path, skiprows=28)
    recording_section = recording[(recording['time'] >= start_time) & (recording['time'] <= end_time)]
    plt.plot(recording_section['time'], recording_section[value], label=value)
    plt.plot(recording_section['time'], recording_section[target_value], linestyle='--', label=target_value)
    plt.title('Data Entries')
    plt.xlabel('time')
    plt.ylabel(value)
    plt.legend()
    plt.show()
plot_data(path,'position','target_position',start_time,end_time)





# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 08:29:42 2024

@author: richard.blakey
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import json
import csv
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import math
from torch.utils.data import Dataset, DataLoader
import itertools
from sklearn.model_selection import train_test_split
import subprocess
import re
import gc

os.chdir(r'/home/richard/Documents/netlist_TLM')

speech = subprocess.call(['speech-dispatcher'])
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(device=None)
        torch.cuda.reset_accumulated_memory_stats(device=None)
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(9)
        torch.cuda.manual_seed_all(9)
        print("CUDA cache emptied")
    else:
        device = torch.device('cpu')
        
        print("CUDA is not available. Using CPU.")
    return device

device = get_device()


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'DejaVu Serif'

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(9)
#%% Generating training and validation data
def angf(f):
    return f*np.pi*2

def cap(c,freq):
    return 0-(1j/((c)*angf(freq)))

def ind(i,freq):
    return 0+(1j*(i)*angf(freq))

def res(r):
    return (r)+(0*1j)

def genFreqSweep(freq_min, freq_max, freq_points, lin_log):
    if lin_log == 'lin':
        frequencies = np.linspace(freq_min, freq_max, freq_points)
    elif lin_log == 'log':
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), freq_points)
    else:
        raise ValueError("lin_log should be 'lin' or 'log'")
    return frequencies

def genImpSpec_IndFour(Lp, Cp, Rp, freq_sweep):
    imp_spectra = 1/(1/(ind(Lp, freq_sweep))+1/(res(Rp))+1/(cap(Cp, freq_sweep)))
    return imp_spectra

def genNetList_IndFour(Lp, Cp, Rp):
    netlist = f".net\nL 1 2 {Lp:.2e}\nC 1 2 {Cp:.2e}\nR 1 2 {Rp:.2e}\n.end"
    return netlist

def genImpSpec_CapFour(Rs, Ls, Cs, freq_sweep):
    imp_spectra = res(Rs) + ind(Ls, freq_sweep) + cap(Cs, freq_sweep)
    return imp_spectra

def genNetList_CapFour(Rs, Ls, Cs):
    netlist = f".net\nR 1 2 {Rs:.2e}\nL 2 3 {Ls:.2e}\nC 3 4 {Cs:.2e}\n.end"
    return netlist

def genImpSpec_ResFour(Rs, Ls, Cp, freq_sweep):
    imp_spectra = 1/(1/(res(Rs) + ind(Ls, freq_sweep)) + 1/(cap(Cp, freq_sweep)))
    return imp_spectra

def genNetList_ResFour(Rs, Ls, Cp):
    netlist = f".net\nR 1 2 {Rs:.2e}\nL 2 3 {Ls:.2e}\nC 1 3 {Cp:.2e}\n.end"
    return netlist

def genImpSpec_FerEight(Lp1, Cp1, Rp1, Lp2, Cp2, Rp2, freq_sweep):
    imp_spectra_1 = 1/(1/(ind(Lp1, freq_sweep))+1/(res(Rp1))+1/(cap(Cp1, freq_sweep)))
    imp_spectra_2 = 1/(1/(ind(Lp2, freq_sweep))+1/(res(Rp2))+1/(cap(Cp2, freq_sweep)))
    return imp_spectra_1 + imp_spectra_2

def genNetList_FerEight(Lp1, Cp1, Rp1, Lp2, Cp2, Rp2):
    netlist = f".net\nL 1 2 {Lp1:.2e}\nC 1 2 {Cp1:.2e}\nR 1 2 {Rp1:.2e}\nL 2 3 {Lp2:.2e}\nC 2 3 {Cp2:.2e}\nR 2 3 {Rp2:.2e}\n.end"
    return netlist

def plotImpedanceSpectra(frequency, impedance_spectra):
    magnitude = np.abs(impedance_spectra)
    phase = np.angle(impedance_spectra, deg=True)

    fig, ax1 = plt.subplots()

    ax1.plot(frequency, magnitude, 'b-', label='Magnitude')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Impedance Magnitude (Ohms)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(frequency, phase, 'r-', label='Phase')
    ax2.set_xscale('log')
    ax2.set_ylabel('Phase (deg)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title('Impedance Magnitude and Phase vs Frequency')

    plt.tight_layout()
    plt.show()
    
def minMaxImpMag(frequency, impedance_spectra):
    magnitude = np.abs(impedance_spectra)

    min_index = np.argmin(magnitude)
    max_index = np.argmax(magnitude)

    min_freq = frequency[min_index]
    max_freq = frequency[max_index]

    return min_freq, max_freq

def calcResonantFreq_Ind(Lp, Cp):
    return 1 / (2 * np.pi * np.sqrt(Lp * Cp))

def calcResonantFreq_Cap(Ls, Cs):
    return 1 / (2 * np.pi * np.sqrt(Ls * Cs))

def isResonantInLogRange(f_res, freq_min, freq_max):
    log_freq_min = np.log10(freq_min)
    log_freq_max = np.log10(freq_max)
    
    log_freq_mid_min = log_freq_min + 0.6 * (log_freq_max - log_freq_min)
    log_freq_mid_max = log_freq_max - 0.1 * (log_freq_max - log_freq_min)
    
    mid_min = 10 ** log_freq_mid_min
    mid_max = 10 ** log_freq_mid_max
    
    return mid_min < f_res < mid_max

def generate_inductor_entry(mantissa):
    
    freq_min = 10 ** random.uniform(1, 3)
    freq_max = 10 ** random.uniform(6, 10)  
    freq_points = random.randint(801, 1601)
    lin_log = 'log' 
    freq_sweep = genFreqSweep(freq_min, freq_max, freq_points, lin_log)

    
    log_freq_min = np.log10(freq_min)
    log_freq_max = np.log10(freq_max)
    log_freq_mid_min = log_freq_min + 0.5 * (log_freq_max - log_freq_min)
    log_freq_mid_max = log_freq_max - 0.2 * (log_freq_max - log_freq_min)

    while True:
        
        log_f_res = random.uniform(log_freq_mid_min, log_freq_mid_max)
        f_res = 10 ** log_f_res
        Lp_exponent = np.random.uniform(-9, 0)
        Lp = mantissa * 10 ** Lp_exponent
        Cp = 1 / ((2 * np.pi * f_res) ** 2 * Lp)
        if 1e-13 <= Cp <= 1e-5:
            Lp_imp = 2 * np.pi * f_res * Lp
            Rp = 10 ** random.uniform(np.log10(Lp_imp)*0.9, np.log10(Lp_imp)*1.2)
            Lp = float(f"{Lp:.2e}")
            Cp = float(f"{Cp:.2e}")
            Rp = float(f"{Rp:.2e}")
            break

    imp_spectra = genImpSpec_IndFour(Lp, Cp, Rp, freq_sweep)
    netlist = genNetList_IndFour(Lp, Cp, Rp)

    return {
        'type': 'inductor',
        'frequency': list(freq_sweep),
        'imp_spectra_real': {'0': list(np.real(imp_spectra))},
        'imp_spectra_imag': {'0': list(np.imag(imp_spectra))},
        'imp_spectra_mag':  {'0': list(np.absolute(imp_spectra))},
        'imp_spectra_ang':  {'0': list(np.angle(imp_spectra, deg=True))},
        'netlist': netlist
    }

def generate_capacitor_entry(mantissa):
    freq_min = 1
    freq_max = 10 ** random.uniform(6, 10)
    freq_points = random.randint(801, 1601)
    lin_log = 'log'
    freq_sweep = genFreqSweep(freq_min, freq_max, freq_points, lin_log)

    log_freq_min = np.log10(freq_min)
    log_freq_max = np.log10(freq_max)
    log_freq_mid_min = log_freq_min + 0.5 * (log_freq_max - log_freq_min)
    log_freq_mid_max = log_freq_max - 0.2 * (log_freq_max - log_freq_min)

    while True:
        f_res = 10 **random.uniform(log_freq_mid_min, log_freq_mid_max)
        
        Cs_exponent = np.random.uniform(-13, -3)
        Cs = mantissa * 10 ** Cs_exponent
        Ls = 1 / ((2 * np.pi * f_res) ** 2 * Cs)

        if 1e-11 <= Ls <= 1e-3:
            Cs_imp = 1 / (2 * np.pi * f_res * Cs)
            Rs = 10 ** random.uniform(np.log10(Cs_imp)*0.5, np.log10(Cs_imp)*1.2)

            Ls = float(f"{Ls:.2e}")
            Cs = float(f"{Cs:.2e}")
            Rs = float(f"{Rs:.2e}")
            break

    imp_spectra = genImpSpec_CapFour(Rs, Ls, Cs, freq_sweep)
    netlist = genNetList_CapFour(Rs, Ls, Cs)

    return {
        'type': 'capacitor',
        'frequency': list(freq_sweep),
        'imp_spectra_real': {'0': list(np.real(imp_spectra))},
        'imp_spectra_imag': {'0': list(np.imag(imp_spectra))},
        'imp_spectra_mag':  {'0': list(np.absolute(imp_spectra))},
        'imp_spectra_ang':  {'0': list(np.angle(imp_spectra, deg=True))},
        'netlist': netlist
    }

def generate_resistor_entry(mantissa):
    freq_min = 10 ** random.uniform(1, 3)
    freq_max = 10 ** random.uniform(8, 10)  # Uniformly distributed on log scale between 1e6 and 1e10 Hz
    freq_points = random.randint(801, 1601)
    lin_log = 'log' 
    freq_sweep = genFreqSweep(freq_min, freq_max, freq_points, lin_log)

    log_freq_min = np.log10(freq_min)
    log_freq_max = np.log10(freq_max)
    log_freq_mid_min = log_freq_min + 0.9 * (log_freq_max - log_freq_min)
    log_freq_mid_max = log_freq_max - 0.0 * (log_freq_max - log_freq_min)
    
    freq_mid_min = 10 ** log_freq_mid_min
    freq_mid_max = 10 ** log_freq_mid_max
    while True:
        log_f_res = random.uniform(log_freq_mid_min, log_freq_mid_max)
        f_res = 10 ** log_f_res
        Rs_exponent = np.random.uniform(-3, 7)
        Rs = mantissa * 10 ** Rs_exponent
        
        Ls = 8e-24
        
        Cp_exponent = np.random.uniform(-13, -10)
        Cp_mantissa = np.random.choice(np.arange(1.00, 10.00, 0.01))
        Cp = Cp_mantissa * 10 ** Cp_exponent
        
        f_res = 1 / (2 * np.pi * Rs * Cp)
        if freq_mid_min <= f_res <= freq_mid_max:
            Ls = float(f"{Ls:.2e}")
            Cp = float(f"{Cp:.2e}")
            Rs = float(f"{Rs:.2e}")
            break  

    imp_spectra = genImpSpec_ResFour(Rs, Ls, Cp, freq_sweep)
    netlist = genNetList_ResFour(Rs, Ls, Cp)

    return {
        'type': 'resistor',
        'frequency': list(freq_sweep),
        'imp_spectra_real': {'0': list(np.real(imp_spectra))},
        'imp_spectra_imag': {'0': list(np.imag(imp_spectra))},
        'imp_spectra_mag':  {'0': list(np.absolute(imp_spectra))},
        'imp_spectra_ang':  {'0': list(np.angle(imp_spectra, deg=True))},
        'netlist': netlist
    }

def generate_ferrite_entry(mantissa):

    freq_min = 10 ** random.uniform(1, 3)
    freq_max = 10 ** random.uniform(6, 10)
    freq_points = random.randint(801, 1601)
    lin_log = 'log'
    freq_sweep = genFreqSweep(freq_min, freq_max, freq_points, lin_log)

    log_freq_min = np.log10(freq_min)
    log_freq_max = np.log10(freq_max)
    log_freq_mid_min = log_freq_min + 0.5 * (log_freq_max - log_freq_min)
    log_freq_mid_max = log_freq_max - 0.2 * (log_freq_max - log_freq_min)

    while True:
        log_f_res1 = random.uniform(log_freq_mid_min, log_freq_mid_max)
        f_res1 = 10 ** log_f_res1
        Lp1_exponent = np.random.uniform(-9, 0)
        Lp1 = mantissa * 10 ** Lp1_exponent
        Cp1 = 1 / ((2 * np.pi * f_res1) ** 2 * Lp1)
        if 1e-13 <= Cp1 <= 1e-5:
            Lp_imp1 = 2 * np.pi * f_res1 * Lp1
            Rp1 = 10 ** random.uniform(np.log10(Lp_imp1)*0.8, np.log10(Lp_imp1)*0.9)
            Lp1 = float(f"{Lp1:.2e}")
            Cp1 = float(f"{Cp1:.2e}")
            Rp1 = float(f"{Rp1:.2e}")
            break
    
    while True:
        Lp2 = random.uniform(Lp1*1.2, Lp1*5)
        Cp2 = random.uniform(Cp1*0.7, Cp1*0.9)
        f_res2 = 1 / (2 * np.pi * np.sqrt(Lp2 * Cp2))
        
        if f_res1*0.6 <= f_res2 <= f_res1*0.8:
            Rp2 = random.uniform(Rp1*0.6, Rp1*0.7)

            Lp2 = float(f"{Lp2:.2e}")
            Cp2 = float(f"{Cp2:.2e}")
            Rp2 = float(f"{Rp2:.2e}")
            break
    
    imp_spectra = genImpSpec_FerEight(Lp1, Cp1, Rp1, Lp2, Cp2, Rp2, freq_sweep)
    netlist = genNetList_FerEight(Lp1, Cp1, Rp1, Lp2, Cp2, Rp2)

    return {
        'type': 'ferrite',
        'frequency': list(freq_sweep),
        'imp_spectra_real': {'0': list(np.real(imp_spectra))},
        'imp_spectra_imag': {'0': list(np.imag(imp_spectra))},
        'imp_spectra_mag':  {'0': list(np.absolute(imp_spectra))},
        'imp_spectra_ang':  {'0': list(np.angle(imp_spectra, deg=True))},
        'netlist': netlist
    }

def genDatabase(num_entries, test_size=0.1, format='JSON'):
    data = []
    labels = []
    
    model_name = f'datasets/IndCapResFer_{num_entries}_{test_size}'
    entries = num_entries // 4
    mantissas = [round(i / 100, 2) for i in range(100, 1000)]
    print(len(mantissas))
    
    with tqdm(total=num_entries, desc='Generating database') as pbar:
        for i in range(entries):
            mantissa = mantissas[i % len(mantissas)]

            data.append(generate_inductor_entry(mantissa))
            labels.append('inductor')
            pbar.update(1)
            
            data.append(generate_capacitor_entry(mantissa))
            labels.append('capacitor')
            pbar.update(1)
            
            data.append(generate_resistor_entry(mantissa))
            labels.append('resistor')
            pbar.update(1)
            
            data.append(generate_ferrite_entry(mantissa))
            labels.append('ferrite')
            pbar.update(1)
            
        train_data, val_data = train_test_split(data, test_size=test_size, stratify=labels, random_state=9)
        
        plot_random_spectra(train_data)
    
    if format == 'CSV':
        keys = train_data[0].keys()
        with open(f'{model_name}_train_database.json', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(train_data)
        with open(f'{model_name}_val_database.json', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(val_data)
    elif format == 'JSON':
        with open(f'{model_name}_train_database.json', 'w') as jsonfile:
            json.dump(train_data, jsonfile, indent=4)
        with open(f'{model_name}_val_database.json', 'w') as jsonfile:
            json.dump(val_data, jsonfile, indent=4)

def plot_random_spectra(data, num_plots_per_type=5, dc_bias_key='0'):
    data_by_type = defaultdict(list)
    
    for entry in data:
        data_by_type[entry['type']].append(entry)
    
    num_types = len(data_by_type)
    
    fig, axs = plt.subplots(num_types, num_plots_per_type, figsize=(30, 4 * num_types))
    
    if num_types == 1:
        axs = [axs]

    for row_index, (type_name, entries) in enumerate(data_by_type.items()):
        sampled_entries = random.sample(entries, min(num_plots_per_type, len(entries)))
        
        for col_index, entry in enumerate(sampled_entries):
            frequency = np.array(entry['frequency'])
            
            imp_spectra_real = np.array(entry['imp_spectra_real'].get(dc_bias_key, []))
            imp_spectra_imag = np.array(entry['imp_spectra_imag'].get(dc_bias_key, []))
            
            if len(imp_spectra_real) == 0 or len(imp_spectra_imag) == 0:
                continue
            
            imp_spectra = imp_spectra_real + 1j * imp_spectra_imag
            
            magnitude = np.abs(imp_spectra)
            phase = np.angle(imp_spectra, deg=True)
            
            ax1 = axs[row_index, col_index]
            ax2 = ax1.twinx()
            
            ax1.plot(frequency, magnitude, label='Magnitude', color='blue')
            ax1.set_yscale('log')  
            ax1.set_xscale('log')  
            ax1.set_ylabel('Magnitude (Ohms)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2.plot(frequency, phase, label='Phase', color='orange')
            ax2.set_ylabel('Phase (Degrees)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            if col_index == 0:
                ax1.set_title(f"{type_name.capitalize()} Spectra (DC Bias: {dc_bias_key})")
            ax1.set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()

database_size = [108000]#[100]#[25000] #[100, 1000, 10000, 100000, 1000000]
for size in database_size:
    genDatabase(num_entries=size, format='JSON')

#%% Start model training here
def plot_random_spectra(data, num_plots_per_type=3, dc_bias_key='0'):
    data_by_type = defaultdict(list)
    
    for entry in data:
        data_by_type[entry['type']].append(entry)
    
    num_types = len(data_by_type)
    
    fig, axs = plt.subplots(num_types, num_plots_per_type, figsize=(18, 4 * num_types))
    
    if num_types == 1:
        axs = [axs]

    for row_index, (type_name, entries) in enumerate(data_by_type.items()):
        sampled_entries = random.sample(entries, min(num_plots_per_type, len(entries)))
        
        for col_index, entry in enumerate(sampled_entries):
            frequency = np.array(entry['frequency'])
            
            imp_spectra_real = np.array(entry['imp_spectra_real'].get(dc_bias_key, []))
            imp_spectra_imag = np.array(entry['imp_spectra_imag'].get(dc_bias_key, []))
            
            if len(imp_spectra_real) == 0 or len(imp_spectra_imag) == 0:
                continue
            
            imp_spectra = imp_spectra_real + 1j * imp_spectra_imag
            
            magnitude = np.abs(imp_spectra)
            phase = np.angle(imp_spectra, deg=True)
            
            
            ax1 = axs[row_index, col_index]
            ax2 = ax1.twinx()
            
            ax1.plot(frequency, magnitude, label='Magnitude', color='black')
            ax1.set_yscale('log')  
            ax1.set_xscale('log')  
            ax1.set_ylabel('Impedance (Ω)', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0E'))

            ax2.plot(frequency, phase, label='Phase', color='black', linestyle='--')
            ax2.set_ylabel('Phase (ϕ)', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.set_ylim(-90, 90)
            ax2.set_yticks([-90, -45, 0, 45, 90])
            ax1.set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()

plot_random_spectra(train_data)
#%% Loading data
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

load_name = 'IndCapResFer_100_0.1'

train_data = load_data(f'datasets/{load_name}_train_database.json')
val_data = load_data(f'datasets/{load_name}_val_database.json')
print(f'Loaded [{load_name}] training and validation datasets')
plot_random_spectra(train_data)

#%% Pre-processing data (log transform then min max scaling)

def compute_global_min_max_log(data, index):
    values = []
    for entry in data:
        for dc_bias_key in entry['imp_spectra_real']:
            imp_spectra_real = np.array(entry['imp_spectra_real'][dc_bias_key])
            imp_spectra_imag = np.array(entry['imp_spectra_imag'][dc_bias_key])
            frequency = np.array(entry['frequency'])
            if index == 0:  # Frequency
                values.extend(np.log10(frequency))
            elif index == 2:  # Real part
                values.extend(np.sign(imp_spectra_real) * np.log10(np.abs(imp_spectra_real)))
            elif index == 3:  # Imaginary part
                values.extend(imp_spectra_imag)
    
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    return min_val, max_val

def log_transform_min_max_scale_and_pad_data(data, min_freq, max_freq, min_real, max_real, min_imag, max_imag, max_len=1001):
    processed_data = []

    for entry in data:
        frequency = np.array(entry['frequency'])
        frequency = np.log10(frequency)

        for dc_bias_key, imp_spectra_real in entry['imp_spectra_real'].items():
            dc_bias = float(dc_bias_key)
            imp_spectra_real = np.array(imp_spectra_real)
            imp_spectra_imag = np.array(entry['imp_spectra_imag'][dc_bias_key])

            imp_spectra_real = np.sign(imp_spectra_real) * np.log10(np.abs(imp_spectra_real))
            temp_data = []
            for i in range(len(frequency)):
                vector = [frequency[i], dc_bias, imp_spectra_real[i], imp_spectra_imag[i]]
                temp_data.append(vector)

            temp_data = torch.tensor(temp_data, dtype=torch.float32)

            temp_data[:, 0] = (temp_data[:, 0] - min_freq) / (max_freq - min_freq)  # Frequency
            temp_data[:, 2] = (temp_data[:, 2] - min_real) / (max_real - min_real)  # Real part
            temp_data[:, 3] = (temp_data[:, 3] - min_imag) / (max_imag - min_imag)  # Imaginary part

            temp_data[:, 0] = torch.nan_to_num(temp_data[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            temp_data[:, 2] = torch.nan_to_num(temp_data[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            temp_data[:, 3] = torch.nan_to_num(temp_data[:, 3], nan=0.0, posinf=0.0, neginf=0.0)

            if temp_data.shape[0] < max_len:
                padding_size = max_len - temp_data.shape[0]
                end_sequence = torch.zeros((padding_size, 4), dtype=torch.float32)
                temp_data = torch.cat((temp_data, end_sequence), dim=0)
            elif temp_data.shape[0] > max_len:
                temp_data = temp_data[:max_len, :]

            processed_data.append(temp_data)

    return torch.stack(processed_data)

def preprocess_data(train_data, val_data):
    min_freq, max_freq = compute_global_min_max_log(train_data, 0)
    min_real, max_real = compute_global_min_max_log(train_data, 2)
    min_imag, max_imag = compute_global_min_max_log(train_data, 3)

    train_processed_data = log_transform_min_max_scale_and_pad_data(
        train_data, min_freq, max_freq, min_real, max_real, min_imag, max_imag
    )

    val_processed_data = log_transform_min_max_scale_and_pad_data(
        val_data, min_freq, max_freq, min_real, max_real, min_imag, max_imag
    )

    return train_processed_data, val_processed_data

processed_train_data, processed_val_data = preprocess_data(train_data, val_data)

#%% Generating token libraries
def genTokenLibrary():
    value_token = ['<val>']
    special_tokens = ['<unk>', '<pad>']
    netlist_tokens = ['.net', '.end']
    component_tokens = ['R', 'L', 'C']
    node_tokens = [str(i) for i in range(0, 100)]
    
    tokens = special_tokens + netlist_tokens + value_token + component_tokens + node_tokens
    
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    
    return token_to_id, id_to_token

token_to_id, id_to_token = genTokenLibrary()

#%% Tokenising data
def tokenize_netlist(netlist):
    tokens = []
    values = []
    value_mask = []
    target_netlist = []
    
    lines = netlist.strip().split('\n')
    
    scientific_notation_pattern = r'^-?\d+\.\d+e[+-]\d+$'
    
    for line in lines:
        parts = re.split(r'\s+', line.strip())
        
        for part in parts:
            if re.match(scientific_notation_pattern, part):

                tokens.append('<val>')
                target_netlist.append(part)
                values.append(part)
                value_mask.append(1)
            else:
                tokens.append(part)
                target_netlist.append(part)
                values.append(0.0)
                value_mask.append(0)
    
    return tokens, target_netlist, values, value_mask

train_tokenized_netlists = []
train_values = []
train_value_mask = []
train_target_netlist = []

for entry in train_data:
    tokens, target_netlist, values, value_mask = tokenize_netlist(entry['netlist'])
    train_tokenized_netlists.append(tokens)
    train_target_netlist.append(target_netlist)
    train_values.append(values)
    train_value_mask.append(value_mask)

val_tokenized_netlists = []
val_values = []
val_value_mask = []
val_target_netlist = []

for entry in val_data:
    tokens, target_netlist, values, value_mask = tokenize_netlist(entry['netlist'])
    val_tokenized_netlists.append(tokens)
    val_target_netlist.append(target_netlist)
    val_values.append(values)
    val_value_mask.append(value_mask)

def apply_log_and_scale(data):
    flat_data = [np.log10(float(value)) if value != 0 else value for sublist in data for value in sublist]
    non_zero_values = [val for val in flat_data if val != 0]
    if non_zero_values:
        min_val = min(non_zero_values)
        max_val = max(non_zero_values)
        scaled_data = [(val - min_val) / (max_val - min_val) for val in flat_data]
    else:
        scaled_data = flat_data  # If all values are zero, nothing to scale
    reshaped_data = []
    idx = 0
    for sublist in data:
        reshaped_data.append(scaled_data[idx:idx+len(sublist)])
        idx += len(sublist)
    return reshaped_data, min_val, max_val

train_values, min_value, max_value = apply_log_and_scale(train_values)
val_values, _, _ = apply_log_and_scale(val_values)

def tokens_to_tensor(tokenized_netlist, token_to_id):
    token_ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in tokenized_netlist]
    return torch.tensor(token_ids, dtype=torch.long)

train_token_ids_tensors = [tokens_to_tensor(netlist, token_to_id) for netlist in train_tokenized_netlists]
val_token_ids_tensors = [tokens_to_tensor(netlist, token_to_id) for netlist in val_tokenized_netlists]

train_values_tensors = [torch.tensor(values, dtype=torch.float) for values in train_values]
val_values_tensors = [torch.tensor(values, dtype=torch.float) for values in val_values]

train_value_masks_tensors = [torch.tensor(mask, dtype=torch.bool) for mask in train_value_mask]
val_value_masks_tensors = [torch.tensor(mask, dtype=torch.bool) for mask in val_value_mask]

train_inputs = processed_train_data
val_inputs = processed_val_data
train_targets = train_token_ids_tensors
val_targets = val_token_ids_tensors

#%% Defining dataset
class ImpedanceDataset(Dataset):
    def __init__(self, spectra, token_ids, values, value_masks):
        assert len(spectra) == len(token_ids) == len(values) == len(value_masks)
        self.spectra = spectra
        self.token_ids = token_ids
        self.values = values
        self.value_masks = value_masks

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        netlist_tokens = self.token_ids[idx]
        values = self.values[idx]           
        value_mask = self.value_masks[idx]  
        return spectrum, netlist_tokens, values, value_mask

train_dataset = ImpedanceDataset(
    train_inputs,
    train_token_ids_tensors,
    train_values_tensors,
    train_value_masks_tensors
)

val_dataset = ImpedanceDataset(
    val_inputs,
    val_token_ids_tensors,
    val_values_tensors,
    val_value_masks_tensors
)

#%% Batching data
def collate_fn(batch):
    spectra, token_ids_list, values_list, value_masks_list = zip(*batch)

    inputs = torch.stack(spectra)  # Shape: [batch_size, ...]
    
    token_ids_padded = torch.nn.utils.rnn.pad_sequence(
        token_ids_list,
        batch_first=True,
        padding_value=token_to_id['<pad>']
    )
    
    values_padded = torch.nn.utils.rnn.pad_sequence(
        values_list,
        batch_first=True,
        padding_value=0.0
    )
    
    value_masks_padded = torch.nn.utils.rnn.pad_sequence(
        value_masks_list,
        batch_first=True,
        padding_value=0
    ).bool()
    
    return {
        'inputs': inputs,                     # Encoder inputs
        'token_ids': token_ids_padded,        # Decoder targets
        'values': values_padded,              # value values
        'value_masks': value_masks_padded
    }

#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1001):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NeuralNetworkEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims):
        super(NeuralNetworkEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        layers = []

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, self.input_dim)

        x = self.network(x)

        x = x.view(batch_size, seq_len, self.embedding_dim)
        return x

class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_value=10000.0):
        super(ContinuousPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_value = max_value

    def forward(self, x):
        position = x[..., 0] / self.max_value  # Shape: (batch_size, seq_len)
        device = x.device
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() *
            -(math.log(10000.0) / self.d_model)
        )

        sin_part = torch.sin(position.unsqueeze(-1) * div_term)
        cos_part = torch.cos(position.unsqueeze(-1) * div_term)
        pe = torch.cat([sin_part, cos_part], dim=-1)  # Shape: (batch_size, seq_len, d_model)
        return pe


class TransformerModel(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=256, nhead=8, nn_encoder_layers=[64, 64], num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder_input_layer = NeuralNetworkEmbedding(input_dim, d_model, nn_encoder_layers)
        self.pos_encoder = ContinuousPositionalEncoding(d_model)
        self.decoder_input_layer = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        batch_first=True,
                                        dropout=dropout
                                        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.value_regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embedded = self.encoder_input_layer(src) * math.sqrt(self.d_model)
        src_pos_encoded = self.pos_encoder(src)
        src = src_embedded + src_pos_encoded
        tgt = self.decoder_input_layer(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        token_output = self.fc_out(output)  # Shape: [batch_size, seq_len, vocab_size]
        value_output = self.value_regression_head(output).squeeze(-1)  # Shape: [batch_size, seq_len]
        return token_output, value_output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(
            torch.ones((sz, sz), device=device, dtype=torch.bool),
            diagonal=1
            )
        return mask
    
class lossCom_weighted(nn.Module):
    def __init__(self, classification_weight=0.5, regression_weight=0.5):
        super(lossCom_weighted, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
                                                            token_preds.view(-1, token_preds.size(-1)),
                                                            token_targets.reshape(-1)
                                                            )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = 0.0
        total_loss = (self.classification_weight * classification_loss + self.regression_weight * regression_loss)
        return total_loss, classification_loss, regression_loss if isinstance(regression_loss, float) else regression_loss
    
class lossCom_gradNorm(nn.Module):
    def __init__(self, model):
        super(lossCom_gradNorm, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())

    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        classification_grad = torch.autograd.grad(
            classification_loss,
            self.model_parameters,
            create_graph=True,
            allow_unused=True,
            retain_graph=True
        )
        regression_grad = torch.autograd.grad(
            regression_loss,
            self.model_parameters,
            create_graph=True,
            allow_unused=True,
            retain_graph=True
        )
        valid_classification_grads = [g.view(-1) for g in classification_grad if g is not None]
        valid_regression_grads = [g.view(-1) for g in regression_grad if g is not None]
        if valid_classification_grads and valid_regression_grads:
            grad_norm = torch.norm(torch.cat(valid_classification_grads)) / torch.norm(torch.cat(valid_regression_grads))
        else:
            grad_norm = 1.0
        norm_loss = classification_loss + grad_norm * regression_loss
        return norm_loss, classification_loss, regression_loss if isinstance(regression_loss, float) else regression_loss

class GradNormLoss(nn.Module):
    def __init__(self, model_parameters, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.alpha = alpha
        self.model_parameters = list(model_parameters)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()

    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        L_t = torch.stack([classification_loss, regression_loss])
        if self.L_0 is None:
            self.L_0 = L_t.detach()
        weighted_losses = L_t * self.w
        total_loss = weighted_losses.sum()
        grads = []
        for i in range(2):  # For each task
            grad = torch.autograd.grad(weighted_losses[i], self.model_parameters, retain_graph=True, create_graph=True)
            grad = torch.cat([g.contiguous().view(-1) for g in grad if g is not None])
            grad_norm = torch.norm(grad, 2)
            grads.append(grad_norm)
        grads = torch.stack(grads)
        G_avg = grads.mean().detach()
        L_hat = (L_t / self.L_0).detach()
        r_t = L_hat / L_hat.mean()
        target_grads = G_avg * (r_t ** self.alpha)
        grad_norm_loss = self.l1_loss(grads, target_grads)
        total_loss = total_loss + grad_norm_loss
        return total_loss, classification_loss, regression_loss

class lossCom_PCGrad(nn.Module):
    def __init__(self, model):
        super(lossCom_PCGrad, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())  # Store model parameters

    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        token_preds = token_preds.reshape(-1, token_preds.size(-1))  # [batch_size * seq_len, vocab_size]
        token_targets = token_targets.reshape(-1).long()             # [batch_size * seq_len]
        classification_loss = self.classification_loss_fn(token_preds, token_targets)
        value_preds_masked = value_preds[value_mask]
        value_targets_masked = value_targets[value_mask]
        if value_preds_masked.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds_masked, value_targets_masked)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True, device=token_preds.device)
        classification_grad = torch.autograd.grad(
            classification_loss,
            self.model_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )
        regression_grad = torch.autograd.grad(
            regression_loss,
            self.model_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        pc_grad = []
        for grad_class, grad_regr in zip(classification_grad, regression_grad):
            if grad_class is not None and grad_regr is not None:
                grad_class_flat = grad_class.flatten()
                grad_regr_flat = grad_regr.flatten()
                dot_product = torch.dot(grad_regr_flat, grad_class_flat)
                if dot_product < 0:
                    norm_class = torch.dot(grad_class_flat, grad_class_flat) + 1e-8
                    proj = dot_product / norm_class
                    projected_grad_flat = grad_regr_flat - proj * grad_class_flat
                    projected_grad = projected_grad_flat.view_as(grad_regr)
                    pc_grad.append(projected_grad)
                else:
                    pc_grad.append(grad_regr)
            elif grad_class is not None:
                pc_grad.append(grad_class)
            elif grad_regr is not None:
                pc_grad.append(grad_regr)
            else:
                pc_grad.append(None)
        for param, adj_grad in zip(self.model_parameters, pc_grad):
            if adj_grad is not None:
                if param.grad is None:
                    param.grad = adj_grad.clone()
                else:
                    param.grad += adj_grad.clone()
        combined_loss = (classification_loss + regression_loss) / 2
        return combined_loss, classification_loss, regression_loss

class DTP_Loss(nn.Module):
    def __init__(self, model, alpha=2):
        super(DTP_Loss, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())
        self.alpha = alpha
        self.classification_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.regression_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.classification_best_loss = float('inf')
        self.regression_best_loss = float('inf')
        
    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        if classification_loss.item() < self.classification_best_loss:
            self.classification_best_loss = classification_loss.item()
        else:
            self.classification_weights.data = (self.classification_weights.data * (self.alpha)).clone()
        if regression_loss.item() < self.regression_best_loss:
            self.regression_best_loss = regression_loss.item()
        else:
            self.regression_weights.data = (self.regression_weights.data * (self.alpha)).clone()
        total_weight = max(self.classification_weights.data, self.regression_weights.data)
        self.classification_weights.data = (self.classification_weights.data / total_weight).clone()
        self.regression_weights.data = (self.regression_weights.data / total_weight).clone()
        combined_loss = (
            self.classification_weights * classification_loss +
            self.regression_weights * regression_loss
        )
        return combined_loss, classification_loss, regression_loss if isinstance(regression_loss, float) else regression_loss

class DTP_Loss_RB1(nn.Module):
    def __init__(self, model, alpha=2):
        super(DTP_Loss_RB1, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())
        self.alpha = alpha
        self.classification_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.regression_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.classification_best_loss = float('inf')
        self.regression_best_loss = float('inf')
        
    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        if classification_loss.item() > regression_loss.item():
            self.classification_weights.data = (self.classification_weights.data * (self.alpha)).clone()
        else:
            self.regression_weights.data = (self.regression_weights.data * (self.alpha)).clone()
        total_weight = max(self.classification_weights.data, self.regression_weights.data)
        self.classification_weights.data = (self.classification_weights.data / total_weight).clone()
        self.regression_weights.data = (self.regression_weights.data / total_weight).clone()
        combined_loss = (
            self.classification_weights * classification_loss +
            self.regression_weights * regression_loss
        )
        return combined_loss, classification_loss, regression_loss if isinstance(regression_loss, float) else regression_loss

class DTP_Loss_RB2(nn.Module):
    def __init__(self, model, alpha=2):
        super(DTP_Loss_RB2, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())
        self.alpha = alpha
        self.classification_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.regression_weights = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.classification_best_loss = float('inf')
        self.regression_best_loss = float('inf')
        
    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        if classification_loss.item() < regression_loss.item():
            self.classification_weights.data = (self.classification_weights.data * (self.alpha)).clone()
        else:
            self.regression_weights.data = (self.regression_weights.data * (self.alpha)).clone()
        total_weight = max(self.classification_weights.data, self.regression_weights.data)
        self.classification_weights.data = (self.classification_weights.data / total_weight).clone()
        self.regression_weights.data = (self.regression_weights.data / total_weight).clone()
        combined_loss = (
            self.classification_weights * classification_loss +
            self.regression_weights * regression_loss
        )
        return combined_loss, classification_loss, regression_loss if isinstance(regression_loss, float) else regression_loss
    
class CaGrad_Loss(nn.Module):
    def __init__(self, model):
        super(CaGrad_Loss, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())  # Store model parameters
    
    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.reshape(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds_masked = value_preds[value_mask]
        value_targets_masked = value_targets[value_mask]
        if value_preds_masked.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds_masked, value_targets_masked)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        classification_grad = torch.autograd.grad(
            classification_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        regression_grad = torch.autograd.grad(
            regression_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        for param in self.model_parameters:
            param.grad = torch.zeros_like(param)
        for param, g1, g2 in zip(self.model_parameters, classification_grad, regression_grad):
            if g1 is not None and g2 is not None:
                cos_sim = torch.dot(g1.reshape(-1), g2.reshape(-1)) / (torch.norm(g1.reshape(-1)) * torch.norm(g2.reshape(-1)) + 1e-8)
                if cos_sim < 0:
                    adjusted_grad = g1 - (torch.dot(g1.reshape(-1), g2.reshape(-1)) / (torch.norm(g2.reshape(-1))**2 + 1e-8)) * g2
                    param.grad += adjusted_grad + g2
                else:
                    param.grad += g1 + g2
            else:
                if g1 is not None:
                    param.grad += g1
                if g2 is not None:
                    param.grad += g2
        combined_loss = (classification_loss + regression_loss) / 2
        return combined_loss, classification_loss, regression_loss

class GradVac_Loss(nn.Module):
    def __init__(self, model, variance_threshold=1.0):
        super(GradVac_Loss, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())
        self.variance_threshold = variance_threshold

    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        classification_grad = torch.autograd.grad(
            classification_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        regression_grad = torch.autograd.grad(
            regression_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        for param in self.model_parameters:
            param.grad = torch.zeros_like(param)
        for param, g1, g2 in zip(self.model_parameters, classification_grad, regression_grad):
            if g1 is not None and g2 is not None:
                grad_stack = torch.stack([g1, g2], dim=0)
                variance = torch.var(grad_stack, dim=0)
                high_variance_mask = variance > self.variance_threshold
                g1_clipped = torch.where(high_variance_mask, torch.clamp(g1, -self.variance_threshold, self.variance_threshold), g1)
                g2_clipped = torch.where(high_variance_mask, torch.clamp(g2, -self.variance_threshold, self.variance_threshold), g2)
                combined_grad = g1_clipped + g2_clipped
                param.grad += combined_grad
            else:
                if g1 is not None:
                    param.grad += g1
                if g2 is not None:
                    param.grad += g2
        combined_loss = (classification_loss + regression_loss) / 2
        return combined_loss, classification_loss, regression_loss

class MGDA_Loss(nn.Module):
    def __init__(self, model):
        super(MGDA_Loss, self).__init__()
        self.classification_loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_id['<pad>'])
        self.regression_loss_fn = nn.MSELoss()
        self.model_parameters = list(model.parameters())
        
    def forward(self, token_preds, token_targets, value_preds, value_targets, value_mask):
        classification_loss = self.classification_loss_fn(
            token_preds.view(-1, token_preds.size(-1)), token_targets.reshape(-1)
        )
        value_preds = value_preds[value_mask]
        value_targets = value_targets[value_mask]
        if value_preds.numel() > 0:
            regression_loss = self.regression_loss_fn(value_preds, value_targets)
        else:
            regression_loss = torch.tensor(0.0, requires_grad=True).to(token_preds.device)
        classification_grad = torch.autograd.grad(
            classification_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        regression_grad = torch.autograd.grad(
            regression_loss, self.model_parameters, retain_graph=True, create_graph=True, allow_unused=True
        )
        grads = []
        for g1, g2 in zip(classification_grad, regression_grad):
            if g1 is not None and g2 is not None:
                grads.append(torch.stack([g1.view(-1), g2.view(-1)], dim=0))
            else:
                grads.append(None)
        weights = torch.tensor([0.5, 0.5], device=token_preds.device)
        for param in self.model_parameters:
            param.grad = torch.zeros_like(param)
        for param, grad_pair in zip(self.model_parameters, grads):
            if grad_pair is not None:
                combined_grad = weights[0] * grad_pair[0].view(param.shape) + weights[1] * grad_pair[1].view(param.shape)
                param.grad += combined_grad
        combined_loss = (classification_loss + regression_loss) / 2
        return combined_loss, classification_loss, regression_loss

#%%
def create_model_name(d_model, n_head, num_encoder, num_decoder, lr, batch_size, num_epochs, criterion_name, optimizer_name):
    model_name = f"{criterion_name}"
    return model_name

def save_loss_data(model_name, epoch, train_loss, val_loss, file_path):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Model', 'Epoch', 'Train Loss', 'Val Loss'])
        
        writer.writerow([model_name, epoch, train_loss, val_loss])

def plot_training_validation_loss(train_total_loss, val_total_loss, 
                                  train_cls_loss, val_cls_loss, 
                                  train_reg_loss, val_reg_loss):
    
    epochs = range(1, len(train_total_loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_total_loss, label='Training Total Loss', color='blue', marker='o')
    plt.plot(epochs, val_total_loss, label='Validation Total Loss', color='blue', linestyle='--', marker='o')

    plt.plot(epochs, train_cls_loss, label='Training Classification Loss', color='green', marker='^')
    plt.plot(epochs, val_cls_loss, label='Validation Classification Loss', color='green', linestyle='--', marker='^')

    plt.plot(epochs, train_reg_loss, label='Training Regression Loss', color='red', marker='v')
    plt.plot(epochs, val_reg_loss, label='Validation Regression Loss', color='red', linestyle='--', marker='v')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%
def load_model_checkpoint(filepath, model_class, criterion_class, optimizer_class, device='cuda'):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        checkpoint_hyperparams = checkpoint['hyperparameters']
        model = model_class(
            input_dim=checkpoint_hyperparams['input_dim'],
            vocab_size=checkpoint_hyperparams['vocab_size'],
            d_model=checkpoint_hyperparams['d_model'],
            nhead=checkpoint_hyperparams['nhead'],
            num_encoder_layers=checkpoint_hyperparams['num_encoder_layers'],
            num_decoder_layers=checkpoint_hyperparams['num_decoder_layers']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer_hyperparams = checkpoint_hyperparams.get('optimizer_params', {})
        optimizer = optimizer_class(model.parameters(), **optimizer_hyperparams)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion_params = checkpoint_hyperparams.get('criterion_params', {})
        criterion = criterion_class(model, **criterion_params)
        epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        train_cls_losses = checkpoint.get('train_cls_losses', [])
        train_reg_losses = checkpoint.get('train_reg_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_cls_losses = checkpoint.get('val_cls_losses', [])
        val_reg_losses = checkpoint.get('val_reg_losses', [])

        plot_training_validation_loss(train_losses, val_losses, train_cls_losses, val_cls_losses, train_reg_losses, val_reg_losses)
        return (model, optimizer, criterion, epoch, train_losses, train_cls_losses, 
                train_reg_losses, val_losses, val_cls_losses, val_reg_losses, checkpoint_hyperparams)
    else:
        print(f"No checkpoint found at {filepath}.")
        return None, None, None, 0, [], [], [], [], [], [], {}

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, vocab, clip=1.0, 
                device='cpu', save_dir='models/', checkpoint_path=None, model_class=None, 
                criterion_class=None, optimizer_class=None,
                directions=directions):
    hyperparams = {
        'input_dim': model.encoder_input_layer.input_dim,
        'vocab_size': model.decoder_input_layer.num_embeddings,
        'd_model': model.d_model,
        'nhead': model.transformer.encoder.layers[0].self_attn.num_heads,
        'num_encoder_layers': len(model.transformer.encoder.layers),
        'num_decoder_layers': len(model.transformer.decoder.layers),
        'learning_rate': optimizer.param_groups[0]['lr'],
        'batch_size': train_loader.batch_size,
        'num_epochs': num_epochs,
        'optimizer_params': optimizer.defaults,
        'criterion_params': {}
    }
    
    best_total_loss = float('inf')
    best_classification_loss = float('inf')
    best_regression_loss = float('inf')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_losses = []
    train_cls_losses = []
    train_reg_losses = []
    val_losses = []
    val_cls_losses = []
    val_reg_losses = []
    start_epoch = 0
    
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        model, optimizer, criterion, start_epoch, train_losses, train_cls_losses, train_reg_losses, \
        val_losses, val_cls_losses, val_reg_losses, checkpoint_hyperparams = \
            load_model_checkpoint(checkpoint_path, model_class, criterion_class, optimizer_class, device)

        if checkpoint_hyperparams:
            hyperparams.update(checkpoint_hyperparams)
            print("Model hyperparameters updated from checkpoint.")
        
    scaler = torch.amp.GradScaler('cuda')
    criterion_type = 'criterion_com'
    current_lr = optimizer.param_groups[0]['lr']
    
    parameter_projections = []
    
    for epoch in range(start_epoch, num_epochs):
        
        projections = []
        
        for direction in directions:
            projection_value = 0.0
            for param, dir_vec in zip(model.parameters(), direction):
                if param.requires_grad:
                    projection_value += torch.sum(param.data * dir_vec).item()
            projections.append(projection_value)
        
        parameter_projections.append(projections)
        
        with open(save_dir + 'projections.pkl', 'wb') as f:
            pickle.dump(parameter_projections, f)
        
        torch.cuda.reset_peak_memory_stats(device=None)
        torch.cuda.reset_accumulated_memory_stats(device=None)
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f"Current Learning Rate: {current_lr:.2e}")
        print(f"Current Criterion:     {criterion_type}")
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        train_progress_bar = tqdm(train_loader, desc='Training', leave=False)
        batch_no = 1
        
        epoch_criterion = criterion
        for batch in train_progress_bar:
            
            batch_inputs = batch['inputs'].to(device)
            batch_token_ids = batch['token_ids'].to(device)
            batch_values = batch['values'].to(device)
            batch_value_masks = batch['value_masks'].to(device)

            decoder_input = batch_token_ids[:, :-1]
            token_targets = batch_token_ids[:, 1:]
            value_targets = batch_values[:, 1:]
            value_masks = batch_value_masks[:, 1:]

            tgt_seq_len = decoder_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_padding_mask = (decoder_input == token_to_id['<pad>'])

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                token_preds, value_preds = model(
                    src=batch_inputs,
                    tgt=decoder_input,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )

                loss, cls_loss, reg_loss = epoch_criterion(
                    token_preds, token_targets, value_preds, value_targets, value_masks
                )
            
            if isinstance(criterion, (lossCom_PCGrad, CaGrad_Loss, GradVac_Loss, MGDA_Loss)):
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
            
            
            
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            run_avg_loss = total_loss / batch_no
            run_avg_cls_loss = total_cls_loss / batch_no
            run_avg_reg_loss = total_reg_loss / batch_no
            
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}/{run_avg_loss:.4f}", cla=f"{cls_loss.item():.4f}/{run_avg_cls_loss:.4f}", reg=f"{reg_loss.item():.4f}/{run_avg_reg_loss:.4f}")
            batch_no += 1
            
        if math.isnan(run_avg_loss) or math.isnan(run_avg_cls_loss) or math.isnan(run_avg_reg_loss):
            print("\nNaN detected in loss during training. Stopping training.")
            break
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_cls_loss = total_cls_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader)
        print(f"Training Loss:   {avg_train_loss:.4f} Classification Loss: {avg_train_cls_loss:.4f} Regression Loss: {avg_train_reg_loss:.4f}")
        train_losses.append(avg_train_loss)
        train_cls_losses.append(avg_train_cls_loss)
        train_reg_losses.append(avg_train_reg_loss)
        
        

        
        model.eval()
        total_val_loss = 0
        total_val_cls_loss = 0
        total_val_reg_loss = 0
        
        val_progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        batch_no = 1
        
        with torch.enable_grad():
            for batch in val_progress_bar:
                batch_inputs = batch['inputs'].to(device)
                batch_token_ids = batch['token_ids'].to(device)
                batch_values = batch['values'].to(device)
                batch_value_masks = batch['value_masks'].to(device)
        
                
                decoder_input = batch_token_ids[:, :-1]
                token_targets = batch_token_ids[:, 1:]
                value_targets = batch_values[:, 1:]
                value_masks = batch_value_masks[:, 1:]
        
                tgt_seq_len = decoder_input.size(1)
                tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
                tgt_padding_mask = (decoder_input == token_to_id['<pad>'])
        

                token_preds, value_preds = model(
                    src=batch_inputs,
                    tgt=decoder_input,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )
        
                val_loss, val_cls_loss, val_reg_loss = epoch_criterion(
                    token_preds, token_targets, value_preds, value_targets, value_masks
                )
        
                total_val_loss += val_loss.item()
                total_val_cls_loss += val_cls_loss.item()
                total_val_reg_loss += val_reg_loss.item()
                run_avg_val_loss = total_val_loss / batch_no
                run_avg_val_cls_loss = total_val_cls_loss / batch_no
                run_avg_val_reg_loss = total_val_reg_loss / batch_no
                
                val_progress_bar.set_postfix(loss=f"{val_loss.item():.4f}/{run_avg_val_loss:.4f}", cla=f"{val_cls_loss.item():.4f}/{run_avg_val_cls_loss:.4f}", reg_loss=f"{val_reg_loss.item():.4f}/{run_avg_val_reg_loss:.4f}")
                batch_no += 1
                
            if val_loss.item() < best_total_loss:
                best_total_loss = val_loss.item()
            if val_cls_loss.item() < best_total_loss:
                best_classification_loss = val_cls_loss.item()
            if val_reg_loss.item() < best_total_loss:
                best_regression_loss = val_reg_loss.item()
            
            best_val_loss = float('inf')
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
              
        if math.isnan(run_avg_val_loss) or math.isnan(run_avg_val_cls_loss) or math.isnan(run_avg_val_reg_loss):
            print("\nNaN detected in loss during validation. Stopping training.")
            break
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_cls_loss = total_val_cls_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f} Classification Loss: {avg_val_cls_loss:.4f} Regression Loss: {avg_val_reg_loss:.4f}")
        val_losses.append(avg_val_loss)
        val_cls_losses.append(avg_val_cls_loss)
        val_reg_losses.append(avg_val_reg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(save_dir, model_filename)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'train_losses': train_losses,
            'train_cls_losses': train_cls_losses,
            'train_reg_losses': train_reg_losses,
            'val_losses': val_losses,
            'val_cls_losses': val_cls_losses,
            'val_reg_losses': val_reg_losses,
            'timestamp': timestamp
        }

        torch.save(checkpoint, save_path)
        
        if no_improve_epoch > 5:
            break
        
        
    plot_training_validation_loss(
                                 train_total_loss=train_losses,   # Total training losses
                                 val_total_loss=val_losses,       # Total validation losses
                                 train_cls_loss=train_cls_losses, # Training classification losses
                                 val_cls_loss=val_cls_losses,     # Validation classification losses
                                 train_reg_loss=train_reg_losses, # Training regression losses
                                 val_reg_loss=val_reg_losses      # Validation regression losses
                                 )
    
    data = zip(train_losses, train_cls_losses, train_reg_losses, val_losses, val_cls_losses, val_reg_losses)
    
    headers = [
        "train_total_loss", 
        "train_cls_loss",
        "train_reg_loss", 
        "val_total_loss", 
        "val_cls_loss",  
        "val_reg_loss"
    ]
    
    output_file = save_dir + "losses.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    subprocess.call(['spd-say', '"model training complete"'])

    return model, best_total_loss, best_classification_loss, best_regression_loss

def clear_cuda(model):
    for param in model.fc_out.parameters():
        del param
    
    for param in model.encoder_input_layer.parameters():
        del param
        
    for param in model.decoder_input_layer.parameters():
        del param
    
    for param in model.transformer.encoder.parameters():
        del param
    
    for param in model.transformer.decoder.parameters():
        del param
    
    for param in model.value_regression_head.parameters():
        del param
        
    for param in model.pos_encoder.parameters():
        del param
    
    for param in model.pos_decoder.parameters():
        del param
    del model

    local_vars = list(locals().keys())
    for var_name in local_vars:
        var = locals()[var_name]
        if isinstance(var, torch.Tensor):
            del locals()[var_name]

    global_vars = list(globals().keys())
    for var_name in global_vars:
        var = globals()[var_name]
        if isinstance(var, torch.Tensor):
            del globals()[var_name]
    
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

#%% Hyperparameter tuning
vocab_size = len(token_to_id)
num_epochs = 5
input_dim = 4
batch_size = 10

def worker_init_fn(worker_id):
    np.random.seed(9 + worker_id)
    random.seed(9 + worker_id)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=int(batch_size*0.1),
    shuffle=False,
    collate_fn=collate_fn
)

d_models = [128]#[128, 256, 512]
nheads = [4]
num_encoder_layers_list = [4]#[4, 8, 16]
num_decoder_layers_list = [8]#[4, 8, 16]
learning_rates = [1e-5]
dropouts = [0.1]
clip = [3]
criterion_functions = [
    'lossCom_weighted',
    'lossCom_PCGrad',
    # 'DTP_Loss',
    'CaGrad_Loss',
    'GradVac_Loss',
    'MGDA_Loss'
]

optimizers = ['Adam']

def get_criterion(name, model):
    if name == 'lossCom_weighted':
        return lossCom_weighted(classification_weight=0.5, regression_weight=0.5).to(device)
    elif name == 'lossCom_PCGrad':
        return lossCom_PCGrad(model).to(device)
    elif name == 'DTP_Loss':
        return DTP_Loss(model).to(device)
    elif name == 'DTP_Loss_RB1':
        return DTP_Loss_RB1(model).to(device)
    elif name == 'DTP_Loss_RB2':
        return DTP_Loss_RB2(model).to(device)
    elif name == 'CaGrad_Loss':
        return CaGrad_Loss(model).to(device)
    elif name == 'GradVac_Loss':
        return GradVac_Loss(model).to(device)
    elif name == 'MGDA_Loss':
        return MGDA_Loss(model).to(device)
    else:
        raise ValueError(f"Unknown criterion function: {name}")

def get_optimizer(name, model_params, lr):
    if name == 'Adam':
        return optim.Adam(model_params, lr=lr)
    elif name == 'AdamW':
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

hyperparameter_combinations = list(itertools.product(
    d_models,
    nheads,
    num_encoder_layers_list,
    num_decoder_layers_list,
    learning_rates,
    dropouts,
    clip,
    criterion_functions,
    optimizers
))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for idx, (d_model, nhead, num_encoder_layers, num_decoder_layers,
          learning_rate, dropout, clip, criterion_name, optimizer_name) in enumerate(hyperparameter_combinations):
    set_random_seed(9)
    model_name = create_model_name(
        d_model, nhead, num_encoder_layers, num_decoder_layers,
        learning_rate, batch_size, num_epochs, criterion_name, optimizer_name
    )
    
    try:
        print(f"\nTraining model {idx+1}/{len(hyperparameter_combinations)} with hyperparameters:")
        print(f"input_dim={input_dim}, d_model={d_model}, nhead={nhead}, "
              f"num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}, "
              f"learning_rate={learning_rate}, dropout={dropout}, "
              f"criterion={criterion_name}, optimizer={optimizer_name}")
        
        model = TransformerModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        ).to(device)
        
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start_time = datetime.now()
        
        
        
        save_dir = f'models/{timestamp}/{model_name}/'
        os.makedirs(save_dir, exist_ok=True)
    
        criterion = get_criterion(criterion_name, model)
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
        trained_model, best_total_loss, best_classification_loss, best_regression_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            clip=clip,
            vocab=token_to_id,
            device=device,
            save_dir=save_dir,
            directions=directions
        )
        
        end_time = datetime.now()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        time_used = (end_time - start_time).total_seconds()
        memory_used = end_memory - start_memory
    
        hyperparams = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'criterion': criterion_name,
            'optimizer': optimizer_name,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'time_used': time_used,
            'peak_memory_used_MB': peak_memory / (1024 ** 2),
            'best_total_loss': best_total_loss,
            'best_classification_loss': best_classification_loss,
            'best_regression_loss': best_regression_loss
        }
        
        
        metrics_file_path = os.path.join(f'models/{timestamp}/', f'hyp_training_{timestamp}.json')
        
        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        
        existing_data[f'{model_name}'] = hyperparams
        
        with open(metrics_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    
        try:
            clear_cuda(model)
        except:
            continue
        
    except Exception as e:
        print(f"An error occurred while training model {idx+1}: {e}")
        try:
            clear_cuda(model)
        except:
            continue
        with open('error_log.txt', 'a') as f:
            f.write(f"Error in model {model_name}: {e}\n")
        continue 

#%% Go!
clear_cuda(model)
batch_size = 25

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=int(batch_size*0.1),
    shuffle=False,
    collate_fn=collate_fn
)

input_dim = 4
vocab_size = len(token_to_id)
d_model = 256   #1024#768#512#256#128
nhead = 16
nn_encoder_layers = [64,128,256,128,64]
num_encoder_layers = 16
num_decoder_layers = 16
learning_rate = 1e-24
num_epochs = 1
dropout = 0.1

model = TransformerModel(
    input_dim=input_dim,
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    nn_encoder_layers=nn_encoder_layers,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout
)

model_name = create_model_name(d_model, nhead, num_encoder_layers, num_decoder_layers, learning_rate, batch_size, num_epochs)
file_path = f'models/{model_name}/{model_name}_losses.csv'
save_dir = f'models/{model_name}/'

model.to(device)

criterion = lossCom_weighted(classification_weight=0.5, regression_weight=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###PCGrad loss clip=1.5
# criterion = lossCom_PCGrad(model).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###DTP loss
# criterion = DTP_Loss_RB(model).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###CaGrad Loss
# criterion = CaGrad_Loss(model).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###GradVac Loss
criterion = GradVac_Loss(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###MGDA Loss
# criterion = MGDA_Loss(model).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
# scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, 
#     base_lr=1e-8,    # Lower boundary of learning rate
#     max_lr=1e-3,     # Upper boundary of learning rate
#     step_size_up=200,  # Number of iterations to reach the max_lr
#     mode='exp_range'   # 'triangular', 'triangular2', or 'exp_range'
# )

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    clip=5,
    vocab=token_to_id,
    device=device,
    save_dir=save_dir
)

#%%
def generate_netlist(model, input_data, token_to_id, id_to_token, max_length=100, device='cpu'):
    model.eval()
    input_data = input_data.to(device)

    input_data = input_data.unsqueeze(1)

    generated_tokens = [token_to_id['.net']]

    for _ in range(max_length):
        tgt_input = torch.tensor(generated_tokens, dtype=torch.long).to(device).unsqueeze(1)

        output = model(input_data, tgt_input)

        next_token_logits = output[-1, 0, :]

        next_token_id = torch.argmax(next_token_logits).item()

        generated_tokens.append(next_token_id)

        if id_to_token[next_token_id] == '.end':
            break

    generated_netlist = [id_to_token[token_id] for token_id in generated_tokens]

    return ' '.join(generated_netlist)

def inverse_log_and_scale_single(value, min_val, max_val):#maybe remove the if clause
    if value != 0:
        unscaled_value = value * (max_val - min_val) + min_val
        original_value = 10**unscaled_value
        return original_value
    else:
        return 0 

def beam_search_generate_netlist(model, input_data, token_to_id, id_to_token, beam_width=5, max_length=100, device='cpu'):
    model.eval()
    input_data = input_data.unsqueeze(0).to(device)  # Shape: (1, seq_len_src, input_dim)

    beams = [([token_to_id['.net']], [], 0.0)]

    for _ in range(max_length):
        new_beams = []
        for tokens, values, cum_log_prob in beams:
            tgt_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Shape: (1, seq_len_tgt)

            tgt_seq_len = tgt_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)

            with torch.no_grad():
                token_preds, value_preds = model(input_data, tgt_input, tgt_mask=tgt_mask)

            next_token_logits = token_preds[:, -1, :]  # Shape: (1, vocab_size)
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1).squeeze(0)  # Shape: (vocab_size,)

            next_value_pred = value_preds[:, -1].item()

            topk_log_probs, topk_indices = torch.topk(next_token_log_probs, beam_width)

            for log_prob, idx in zip(topk_log_probs, topk_indices):
                idx = idx.item()
                token = id_to_token[idx]

                new_tokens = tokens + [idx]
                new_cum_log_prob = cum_log_prob + log_prob.item()
                new_values = values.copy()
                if token == '<val>':
                    new_values.append(next_value_pred)

                new_beams.append((new_tokens, new_values, new_cum_log_prob))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

    if beams:
        best_beam = beams[0]
    else:
        best_beam = sorted(new_beams, key=lambda x: x[2], reverse=True)[0]

    tokens, values, _ = best_beam
    generated_tokens = [id_to_token[token_id] for token_id in tokens]

    generated_netlist = []
    value_idx = 0
    i = 0
    while i < len(generated_tokens):
        token = generated_tokens[i]
        if token == '<val>':
            value = inverse_log_and_scale_single(values[value_idx], min_value, max_value)
            value_idx += 1
            value = f"{value:.2e}"
            generated_netlist.append(str(value))
        else:
            generated_netlist.append(token)
        if token == '.end':
            break
        i += 1

    return ' '.join(generated_netlist)


def generate_random_examples(model, val_inputs, val_tokenized_netlists, token_to_id, id_to_token, num_examples=10, device='cpu'):
    random_indices = random.sample(range(len(val_inputs)), num_examples)

    model.to(device)

    for idx in random_indices:
        input_data = val_inputs[idx].to(device)
        target_tokens = val_target_netlist[idx]

        generated_netlist = beam_search_generate_netlist(model, input_data, token_to_id, id_to_token, device=device)
        
        target_netlist = ' '.join(target_tokens)

        print(f"Sample {idx + 1}:")
        print("Generated Netlist:", generated_netlist)
        print("Target Netlist:   ", target_netlist)
        print("-" * 80)

generate_random_examples(
    model=trained_model,
    val_inputs=val_inputs,
    val_tokenized_netlists=val_tokenized_netlists,
    token_to_id=token_to_id,
    id_to_token=id_to_token,
    num_examples=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

#%%
new_num_epochs=10

checkpoint_path = '20241030_182448/model_20241030_1824_dmodel256_nhead8_enc16_dec16_lr1e-05_batch10_epochs1_MGDA_Loss_Adam'
save_dir = f'models/{checkpoint_path}'
checkpoint_epoch = 1

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=new_num_epochs,
    vocab=token_to_id,
    clip=1.0,
    device=device,
    save_dir=save_dir,
    checkpoint_path=f'models/{checkpoint_path}/epoch_{checkpoint_epoch}.pth',
    model_class=TransformerModel,
    criterion_class=type(criterion),
    optimizer_class=type(optimizer)
)


#%%
def delete_tensors():
    local_vars = list(locals().keys())
    for var_name in local_vars:
        var = locals()[var_name]
        if isinstance(var, torch.Tensor):
            del locals()[var_name]
    global_vars = list(globals().keys())
    for var_name in global_vars:
        var = globals()[var_name]
        if isinstance(var, torch.Tensor):
            del globals()[var_name]
    torch.cuda.empty_cache()

clear_cuda(model)
delete_tensors()
torch.cuda.synchronize()
torch.cuda.empty_cache()

#%%
def get_random_direction(model, layer_names=None):
    direction = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if layer_names is None or any(layer_name in name for layer_name in layer_names):
                direction.append(torch.randn_like(param))
            else:
                direction.append(torch.zeros_like(param))
    return direction

selected_layers = ["fc_out", "value_regression_head"]

directions = []
for i in range(10):
    direction = get_random_direction(model, layer_names=selected_layers)
    directions.append(direction)

import pickle
with open('saved_directions.pkl', 'wb') as f:
    pickle.dump(directions, f)
#%%
direction_fixed_1 = direction_1
direction_fixed_2 = direction_2

steps = 40
rang = 0.01 #0.2 #0.01
alpha_range = np.linspace(-rang, rang, steps)
beta_range = np.linspace(-rang, rang, steps)

loss_grid = np.zeros((steps, steps))
cls_loss_grid = np.zeros((steps, steps))
reg_loss_grid = np.zeros((steps, steps))

for batch in train_loader:
    batch = batch
    break  # Only use one batch

criterion = lossCom_weighted()#MGDA_Loss(model).to(device)#
    
def compute_loss_at_point(alpha, beta, direction_1, direction_2):
    batch_inputs = batch['inputs'].to(device)
    batch_token_ids = batch['token_ids'].to(device)
    batch_values = batch['values'].to(device)
    batch_value_masks = batch['value_masks'].to(device)

    decoder_input = batch_token_ids[:, :-1]
    token_targets = batch_token_ids[:, 1:]
    value_targets = batch_values[:, 1:]
    value_masks = batch_value_masks[:, 1:]

    tgt_seq_len = decoder_input.size(1)
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (decoder_input == token_to_id['<pad>'])
    
    for param, d1, d2 in zip(model.parameters(), direction_1, direction_2):
        if param.requires_grad:
            param.data += alpha * d1 + beta * d2

    token_preds, value_preds = model(
        src=batch_inputs,
        tgt=decoder_input,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_padding_mask
    )
    
   
    loss, cls_loss, reg_loss  = criterion(
        token_preds, token_targets, value_preds, value_targets, value_masks
    )
    
    loss_value = loss.item()
    cls_loss_value = cls_loss.item()
    reg_loss_value = reg_loss.item()
    
    for param, d1, d2 in zip(model.parameters(), direction_1, direction_2):
        if param.requires_grad:
            param.data -= alpha * d1 + beta * d2

    return loss_value, cls_loss_value, reg_loss_value

for i, alpha in enumerate(tqdm(alpha_range, desc="Alpha Range Progress")):
    for j, beta in enumerate(beta_range):
        loss_grid[i, j], cls_loss_grid[i, j], reg_loss_grid[i, j] = compute_loss_at_point(alpha, beta)
        
#%%
def gen_contour_charts(alpha_range, beta_range, loss_grid, cls_loss_grid, reg_loss_grid):        
    X, Y = np.meshgrid(alpha_range, beta_range)
    
    for grid_name, grid in zip(["loss_grid", "cls_loss_grid", "reg_loss_grid"], [loss_grid, cls_loss_grid, reg_loss_grid]):
        grid = np.log10(grid)
        plt.rcParams.update({'font.size': 16})
        max_grid = round(np.max(grid),0)
        min_grid = round(np.min(grid),0)
        plt.figure(figsize=(8, 6))
        contour = plt.contour(X, Y, grid,levels=8, colors='black')
        shaded_contour = plt.contourf(X, Y, grid, levels=18, cmap='Greys', alpha=0.7)  # Filled greyscale
        plt.clabel(contour, inline=True, fontsize=16, fmt="%.2f")
        norm = plt.Normalize(vmin=min_grid, vmax=max_grid)
        
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.3f}"))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.3f}"))
        plt.xlabel('Parameter 1', fontsize=24)
        plt.ylabel('Parameter 2', fontsize=24)
        plt.show()

gen_contour_charts(alpha_range, beta_range, loss_grid, cls_loss_grid, reg_loss_grid)


#%%
def load_projections(parent_folder):
    projections_dict = {}
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    for subfolder_path in subfolders:
        projection_files = [f for f in os.listdir(subfolder_path) if 'projection' in f]
        for proj_file in projection_files:
            proj_path = os.path.join(subfolder_path, proj_file)
            with open(proj_path, 'rb') as f:
                projections = pickle.load(f)
                key = f"{os.path.basename(subfolder_path)}/{proj_file}"
                projections_dict[key] = projections
    return projections_dict

parent_folder = 'models/20241209_115104'

model_projections = load_projections(parent_folder)

d1 = 5
d2 = 9

with open('saved_directions.pkl', 'rb') as f:
    directions = pickle.load(f)

direction_a = directions[d1]
direction_b = directions[d2]

steps = 10  
rang = 0.4 #0.2 #0.01
alpha_range = np.linspace(-rang, rang, steps)
beta_range = np.linspace(-rang, rang, steps)

loss_grid = np.zeros((steps, steps))
cls_loss_grid = np.zeros((steps, steps))
reg_loss_grid = np.zeros((steps, steps))

for batch in train_loader:
    batch = batch
    break  # Only use one batch

criterion = lossCom_weighted()#MGDA_Loss(model).to(device)#
    

def compute_loss_at_point(alpha, beta, direction_1, direction_2):
    batch_inputs = batch['inputs'].to(device)
    batch_token_ids = batch['token_ids'].to(device)
    batch_values = batch['values'].to(device)
    batch_value_masks = batch['value_masks'].to(device)

    decoder_input = batch_token_ids[:, :-1]
    token_targets = batch_token_ids[:, 1:]
    value_targets = batch_values[:, 1:]
    value_masks = batch_value_masks[:, 1:]

    tgt_seq_len = decoder_input.size(1)
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (decoder_input == token_to_id['<pad>'])
    
    for param, d1, d2 in zip(model.parameters(), direction_1, direction_2):
        if param.requires_grad:
            param.data += alpha * d1 + beta * d2

    token_preds, value_preds = model(
        src=batch_inputs,
        tgt=decoder_input,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_padding_mask
    )
    
   
    loss, cls_loss, reg_loss  = criterion(
        token_preds, token_targets, value_preds, value_targets, value_masks
    )
    
    loss_value = loss.item()
    cls_loss_value = cls_loss.item()
    reg_loss_value = reg_loss.item()
    for param, d1, d2 in zip(model.parameters(), direction_1, direction_2):
        if param.requires_grad:
            param.data -= alpha * d1 + beta * d2

    return loss_value, cls_loss_value, reg_loss_value

for i, alpha in enumerate(tqdm(alpha_range, desc="Alpha Range Progress")):
    for j, beta in enumerate(beta_range):
        loss_grid[i, j], cls_loss_grid[i, j], reg_loss_grid[i, j] = compute_loss_at_point(alpha, beta, direction_a, direction_b)

def gen_contour_charts(alpha_range, beta_range, loss_grid, cls_loss_grid, reg_loss_grid, trajectory_alpha, trajectory_beta):
    import matplotlib.pyplot as plt
    import numpy as np

    X, Y = np.meshgrid(alpha_range, beta_range)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    plt.rcParams.update({'font.size': 16})
    
    grid_names = ["Classification Loss", "Total Loss", "Regression Loss"]
    grids = [cls_loss_grid, loss_grid, reg_loss_grid]
    for ax, grid_name, grid in zip(axes, grid_names, grids):
        grid = np.log10(grid)
        contour = ax.contour(X, Y, grid, levels=8, colors='black')
        ax.clabel(contour, inline=True, fontsize=16, fmt="%.2f")
        
        ax.plot(trajectory_alpha, trajectory_beta, marker='o', color='red', label='Training Trajectory')
        
        ax.set_xlim([min(alpha_range), max(alpha_range)])
        ax.set_ylim([min(beta_range), max(beta_range)])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.3f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.3f}"))
        ax.set_xlabel('Parameter 1', fontsize=24)
        ax.set_ylabel('Parameter 2', fontsize=24)
        ax.set_title(grid_name, fontsize=14)
    
    plt.tight_layout()
    plt.show()


for key in model_projections:
    projections_array = np.array(model_projections[key])
    print(key)
    trajectory_alpha = projections_array[:, d1] - projections_array[:, d1][-1]
    trajectory_beta  = projections_array[:, d2] - projections_array[:, d2][-1]
    gen_contour_charts(alpha_range, beta_range, loss_grid, cls_loss_grid, reg_loss_grid,trajectory_alpha, trajectory_beta)
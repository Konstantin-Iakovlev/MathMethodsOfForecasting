import numpy as np
from matplotlib import pyplot as plt
import json

SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# read data
with open('cross_val.json', 'r') as inp:
    data = json.load(inp)

# cross validadion exp
# plot the data
plt.figure(figsize=(12, 8))
mean_train = np.array([np.mean(data[r][0]) for r in data])
mean_valid = np.array([np.mean(data[r][1]) for r in data])
std_train = np.array([np.std(data[r][0]) for r in data])
std_valid = np.array([np.std(data[r][1]) for r in data])
plt.plot(list(data.keys()), mean_train, label='train')
plt.fill_between(list(data.keys()), mean_train - std_train, mean_train + std_train,
        alpha=0.4)
plt.plot(list(data.keys()), mean_valid, label='valid')
plt.fill_between(list(data.keys()), mean_valid - std_valid, mean_valid + std_valid,
        alpha=0.4)
plt.xlabel(r'$R$')
plt.ylabel(r'$Q^2$')
plt.legend()
plt.savefig('report/sections/example/figures/cross_val.png')
plt.show()

# robustness
plt.figure(figsize=(12, 8))
with open('robustness.json', 'r') as inp:
    data = json.load(inp)

sigmas = list(data.keys())
mean_train = np.array([np.mean(data[sigma][0]) for sigma in data])
mean_valid = np.array([np.mean(data[sigma][1]) for sigma in data])
std_train = np.array([np.std(data[sigma][0]) for sigma in data])
std_valid = np.array([np.std(data[sigma][1]) for sigma in data])
plt.plot(list(data.keys()), mean_train, label='train')
plt.fill_between(sigmas, mean_train - std_train, mean_train + std_train,
        alpha=0.4)
plt.plot(sigmas, mean_valid, label='valid')
plt.fill_between(sigmas, mean_valid - std_valid, mean_valid + std_valid,
        alpha=0.4)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$Q^2$')
plt.legend()
plt.savefig('report/sections/example/figures/robustness.png')
plt.show()







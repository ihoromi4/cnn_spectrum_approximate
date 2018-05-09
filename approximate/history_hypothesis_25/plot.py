import json
import matplotlib.pyplot as plt

file_path = 'hypothesis_27.json'
file_path = 'adaptive_loss_50_epochs_32_batch_size.json'

with open(file_path) as f:
    history = json.load(f)

for key in history:
    plt.plot(history[key])

plt.show()


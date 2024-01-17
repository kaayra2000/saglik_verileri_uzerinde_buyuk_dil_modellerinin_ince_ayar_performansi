import matplotlib.pyplot as plt
import os
def plot_and_save_metric(history, metric_name, file_name, gorsel_yolu):
    epochs = range(1, len(history) + 1)
    train_values = [log.get(metric_name) for log in history]
    val_values = [log.get('val_' + metric_name) for log in history]

    plt.figure()
    plt.plot(epochs, train_values, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_values, 'r*-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.savefig(os.path.join(gorsel_yolu, file_name))
    plt.close()
import os
import visdom
import numpy as np
import matplotlib.pyplot as plt

use_visdom = True

MODEL_DIR = '/home/yous/Desktop/thesis/CartPoleSimulation/SI_Toolkit_ASF/' \
            'Experiments/CPS-Tutorial/Models'
MODELS = os.listdir(MODEL_DIR)
NUM_STEPS = 50

per_k_results = [[] for _ in range(NUM_STEPS)]
avg_results = []

FINAL_MODELS = []
for model in sorted(MODELS):
    try:
        history = np.load(os.path.join(MODEL_DIR, model, 'training_history.pkl'), allow_pickle=True)
    except FileNotFoundError:
        continue
    FINAL_MODELS.append(model)
    evals = history['eval_results']
    for idx, avg in enumerate(evals):
        per_k_results[idx].append(avg)
    avg_results.append(np.mean(evals[:15]))

NUM_MODELS = len(FINAL_MODELS)
print(''.join([f'{idx}:{x}\n' for idx, x in enumerate(FINAL_MODELS)]))

if use_visdom:
    vis = visdom.Visdom()
    vis.line(Y=avg_results, X=list(range(NUM_MODELS)),
             opts=dict(title=f'Average Loss Over All Timesteps', xlabel='Model Index/K Value',
                       ylabel='Loss'))
    vis.line(Y=[np.argmin(x) for x in per_k_results], X=list(range(NUM_STEPS)),
             opts=dict(title=f'Best Model for each K value', xlabel='K',
                       ylabel='Model Index/K Value'))
    for idx, entry in enumerate(per_k_results):
        vis.line(Y=entry, X=list(range(NUM_MODELS)), opts=dict(title=f'Loss at K={idx+1}',
                                                               xlabel='K Value', ylabel='Loss'))
else:
    fig, axs = plt.subplots(NUM_STEPS, 2)
    for idx, entry in enumerate(per_k_results):
        row, col = idx // 2, idx % 2
        axs[row, col].plot(list(range(NUM_MODELS)), entry)
        axs[row, col].set_title(f'Evaluation at timestep {idx+1}')
        axs[row, col].set_xlabel('K Value')
        axs[row, col].set_ylabel('Loss')
    plt.savefig('/home/yous/Desktop/thesis/intermediate_results.png')
    plt.subplot_tool()
    plt.show()
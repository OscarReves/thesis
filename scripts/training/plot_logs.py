import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

logdir = "logs/e5_finetune/"  # change this
scalar_tag = "test/loss_single_batch"  # change this to whatever tag you want to plot

plt.figure()

for filename in os.listdir(logdir):
    if filename.startswith("events.out.tfevents"):
        filepath = os.path.join(logdir, filename)
        ea = event_accumulator.EventAccumulator(filepath)
        try:
            ea.Reload()
            if scalar_tag in ea.Tags().get("scalars", []):
                scalars = ea.Scalars(scalar_tag)
                steps = [s.step for s in scalars]
                values = [s.value for s in scalars]
                plt.plot(steps, values, label=filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

plt.xlabel("Step")
plt.ylabel(scalar_tag)
plt.title(f"{scalar_tag} over training steps")
plt.legend()
plt.grid(True)
plt.savefig('results/figures/loss')
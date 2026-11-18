Make sure to use Python 3.13.7

Install the required Python packages using pip
```bash
pip install torch torchvision tqdm
```

Make sure that the checkpoints folder is in the same directory as proof.py, and that it contains the following:
- `mnist.pt`
- `mnist_distil_dict.pt`
- `latest_T1.pt`
- `distil_dict.pt`

MNIST and CIFAR10 datasets should be downloaded automatically by torchvision on first run, if this errors then download the official MNIST and CIFAR10 .tar.gz archives and put them in folders called mnist and cifar in the active directory.

Run the proof.py file as so, this will enumerate all the results, reproduce them and print out the results.
```bash
python -c "import proof; proof.runProofs(n=1, bsize=1)"
```

Either include the checkpoint directory in the local or allow the script to train replacements (note running the script without the preloaded checkpoints in the .zip will take significantly longer)


def draw(filepath, title):
    with open(filepath,'r') as f:
        lines = f.readlines()

    G_loss = []
    L1_loss = []
    D_real_loss = []
    D_fake_loss = []
    D_loss = []

    for line in lines:
        if line[0] != '(':
            continue
        line = line.split(')')[1]
        line = line.split(':')[1:]
        losses = [float(t.split()[0]) for t in line]
        G_loss.append(losses[0])
        L1_loss.append(losses[1])
        D_real_loss.append(losses[2])
        D_fake_loss.append(losses[3])
        D_loss.append(losses[2]+losses[3])

    num_epoch = len(G_loss)
    indices = slice(0, num_epoch,4)
    x = list(range(num_epoch))[indices]

    import matplotlib.pyplot as plt
    import numpy as np

    lw = 0.9

    plt.plot(x, G_loss[indices], label="G loss",linewidth=lw)
    plt.plot(x, D_loss[indices], label="D loss",linewidth=lw)
    plt.plot(x, np.asarray(L1_loss)[indices], label="L1 loss",linewidth=lw)
    plt.ylim(-1,5)
    plt.legend()
    plt.title(title)
    plt.show()
    return plt

filepath = './checkpoints/oracle_pix2pix/loss_log.txt'
draw(filepath, 'baseline')

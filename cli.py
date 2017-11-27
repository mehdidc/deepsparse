from clize import run
import torch.nn as nn
from torch.autograd import Variable
import torch
from data import load_dataset
from machinedesign.viz import grid_of_images_default
from skimage.io import imsave

from torch.nn.init import xavier_uniform


def train(*, dataset='mnist'):
    z1 = 100
    z2 = 512
    batch_size = 64
    lr = 0.1

    dataset = load_dataset(dataset, split='train')
    x0, _ = dataset[0]
    c, h, w = x0.size()
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )

    w1 = torch.rand(w*h*c, z1).cuda()
    w1 = Variable(w1, requires_grad=True)
    xavier_uniform(w1.data)
    """
    w1_2 = torch.rand(z1, z2)
    w1_2 = Variable(w1_2, requires_grad=True)
    xavier_uniform(w1_2.data)
    w1_2 = w1_2.cuda()
        

    wx_2 = torch.rand(w*h*c, z2)
    wx_2 = Variable(wx_2, requires_grad=True)
    xavier_uniform(wx_2.data)
    wx_2 = wx_2.cuda()
    """
    
    bias = torch.zeros(w*h*c).cuda()
    bias = Variable(bias, requires_grad=True)

    print(w1.is_leaf, bias.is_leaf)

    grads = {}
    momentum = 0.9
    def save_grad(v):
        def hook(grad):
            v.grad = grad
            if not hasattr(v, 'mem'):
                v.mem = 0.0
            v.mem = v.mem * momentum + v.grad.data * (1 - momentum)
        return hook
    
    #params = [w1, w1_2, wx_2, bias]
    params = [w1, bias]
    optim = torch.optim.Adadelta(params, lr=0.1)
    #for p in params:
    #    p.register_hook(save_grad(p))
    
    gamma = 5.0
    nb_updates = 0
    for _ in range(1000):
        for X, y in dataloader:
            optim.zero_grad()
            X = Variable(X)
            #w2 = torch.matmul(w1, w1_2)
            X = X.cuda()
            X = X.view(X.size(0), -1)
            """
            a2 = torch.matmul(X, wx_2)
            a2 = a2 * (a2 > 0.8).float()
            Xrec = torch.matmul(a2, w2.transpose(0, 1)) + bias
            Xrec = torch.nn.Sigmoid()(Xrec)
            """
            hid = torch.matmul(X, w1)
            hid = hid * (hid > 1.0).float()
            Xrec = torch.matmul(hid, w1.transpose(1, 0).contiguous()) + bias
            Xrec = torch.nn.Sigmoid()(Xrec)
            e1 = ((Xrec - X)**2).sum(1).mean()
            e2 = e1
            e3 = e1
            #e2 = torch.abs(w1_2).mean()
            #e3 = torch.abs(a2).mean()
            loss = e1
            loss.backward()
            optim.step()
            #for p in params:
            #    p.data -= lr * p.mem
            if nb_updates % 100 == 0:
                print('loss : %.3f %.3f %.3f' % (e1.data[0], e2.data[0], e3.data[0]))
                
                active = (hid.data>0).float().sum(1)
                print('nbActive : {:.4f} +- {:.4f}'.format(active.mean(), active.std()))
                im = Xrec.data.cpu().numpy()
                im = im.reshape(im.shape[0], c, h, w)
                im = grid_of_images_default(im, normalize=True)
                imsave('x.png', im)

                im = w1.data.cpu().numpy()
                im = im.reshape((c, h, w, z1)).transpose((3, 0, 1, 2))
                im = grid_of_images_default(im, normalize=True)
                imsave('w1.png', im)
                """
                im = wx_2.data.cpu().numpy()
                im = im.reshape((c, h, w, z2)).transpose((3, 0, 1, 2))
                im = grid_of_images_default(im, normalize=True)
                imsave('w2.png', im)
                """

            nb_updates += 1

run(train)

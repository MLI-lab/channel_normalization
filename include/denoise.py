from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard

from .helpers import *

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def DIPdenoise( net,
                img_noisy_var,
                num_channels,
                img_clean_var,
                net_type="decoder",
                num_iter = 5000,
                LR = 0.01,
                OPTIMIZER='adam',
                opt_input = False,
                reg_noise_std = 0,
                reg_noise_decayevery = 100000,
                mask_var = None,
                apply_f = None,
                decaylr = False,
                net_input = None,          
                net_input_gen = "random",
	):

    if net_input is not None:
        print("input provided")
    else:
        # feed noise into the network
        if net_type == "decoder":
            totalupsample = 2**len(num_channels)
            width = int(img_clean_var.data.shape[2]/totalupsample)
            height = int(img_clean_var.data.shape[3]/totalupsample)
            shape = [1,num_channels[0], width, height]
            print("shape: ", shape)
            if(net_input_gen == "random"):
                net_input = Variable(torch.zeros(shape))
                net_input.data.uniform_()
                net_input.data *= 1./10
            elif(net_input_gen == "hadamard"):
                H = hadamard(width*height)
                ni = np.zeros(shape)
                for i in range(shape[1]):
                    ni[0,i]  = np.reshape(H[i],(width,height))
                net_input = np_to_var(ni[0])
                net_input.data *= 1./20
            elif(net_input_gen == "rademacher"):
                ni = np.random.randint(2, size = shape) - 0.5
                net_input = np_to_var(ni[0])
                net_input.data *= 1./10
            print(net_input.data.cpu().numpy()[0])

        elif net_type == "hourglass":
            print("hourglass mode")
            input_depth = 32
            shape = [1, input_depth, img_clean_var.data.shape[2], img_clean_var.data.shape[3]]
            net_input = Variable(torch.zeros(shape))
            net_input.data.uniform_()
            net_input.data *= 1./10
        elif net_type == "noup":
            print("no upsampling mode")
            shape = [1, num_channels[0], img_clean_var.data.shape[2], img_clean_var.data.shape[3]]
            net_input = Variable(torch.zeros(shape))
            net_input.data.uniform_()
            net_input.data *= 1./10
    #print(net_input.data.cpu().numpy() )
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters() ]

    if(opt_input == True):
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)
    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR)
    elif OPTIMIZER == 'adadelta':
        print("optimize with adadelta", LR)
        optimizer = torch.optim.Adadelta(p, lr=LR, rho=0.9, eps=1e-06, weight_decay=0)

    mse = torch.nn.MSELoss().type(dtype)
    if apply_f is None:
        noise_energy = mse(img_noisy_var, img_clean_var)
    else:
        noise_energy = mse(img_noisy_var, img_noisy_var)

    for i in range(num_iter): 
        if decaylr is True:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=100)
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        optimizer.zero_grad()
        out = net(net_input.type(dtype))
        # training loss
        if mask_var is not None:
            loss = mse( out * mask_var , img_noisy_var * mask_var )
        elif apply_f:
            loss = mse( apply_f(out) , img_noisy_var )
        else:
            loss = mse(out, img_noisy_var)
        loss.backward()
        mse_wrt_noisy[i] = var_to_np(loss)
        # the actual loss
        true_loss = mse(Variable(out.data, requires_grad=False), img_clean_var)
        mse_wrt_truth[i] = var_to_np(true_loss)
        if i % 10 == 0:
            out2 = net(Variable(net_input_saved).type(dtype))
            loss2 = mse(out2, img_clean_var)
            print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f' % (i, loss.data[0],true_loss.data[0],loss2.data[0],noise_energy.data[0]), '\r', end='')
        optimizer.step()
    return mse_wrt_noisy, mse_wrt_truth,net_input_saved



'''

def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        apply_f = None,
        decaylr = False,
        net_input = None,
        net_input_gen = "random",
        ):

    if net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network 
        totalupsample = 2**len(num_channels)
        width = int(img_clean_var.data.shape[2]/totalupsample)
        height = int(img_clean_var.data.shape[3]/totalupsample)
        shape = [1,num_channels[0], width, height]
        print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters() ]

    if(opt_input == True):
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)

    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR)

    mse = torch.nn.MSELoss() #.type(dtype) 
    noise_energy = mse(img_noisy_var, img_clean_var)

    best_net = copy.deepcopy(net)
    best_mse = 1000000.0

    for i in range(num_iter):
        if decaylr is True:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=100)
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        optimizer.zero_grad()
        out = net(net_input.type(dtype))

        # training loss 
        if mask_var is not None:
            loss = mse( out * mask_var , img_noisy_var * mask_var )
        elif apply_f:
            loss = mse( apply_f(out) , img_noisy_var )
        else:
            loss = mse(out, img_noisy_var)
        loss.backward()
        mse_wrt_noisy[i] = loss.data.cpu().numpy()

                # the actual loss 
        true_loss = mse(Variable(out.data, requires_grad=False), img_clean_var)
        mse_wrt_truth[i] = true_loss.data.cpu().numpy()
        if i % 10 == 0:
            out2 = net(Variable(net_input_saved).type(dtype))
            loss2 = mse(out2, img_clean_var)
            #print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f'
            #       % (i, loss.data.item(),true_loss.data.item(),loss2.data.item(),noise_energy.data.item()), '\r', end='')
            print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f' % (i, loss.data[0],true_loss.data[0],loss2.data[0],noise_energy.data[0]), '\r', end='')
        
        # if training loss improves by at least one percent, we found a new best net
        if best_mse > 1.005*loss.data[0]:
            best_mse = loss.data[0]
            best_net = copy.deepcopy(net)

        optimizer.step()

    net = best_net
    return mse_wrt_noisy, mse_wrt_truth,net_input_saved

'''


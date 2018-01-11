import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.light_cnn import LightCNN_9Layers, LightCNN_29Layers
from model.tp-gan import Global
from model.tp-gan import Discriminator
 
_batch_size = 64

#cuda = False
cuda = True
cudnn.benchmark = True
        
#load dataloader
train_loader = data.DataLoader(
    my_datasets(data_root = "/home/hezhenhao/OFD/OFD_full_DB_labelled",
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size = _batch_size, shuffle = True)

#load model
D = Discriminator()
G = Global()

#loss function and regularization
alpha = 1e-3
lamda0 = 10.0
lamda1 = 0.3
lamda2 = 1e-3
lamda3 = 3e-3
lamda4 = 1e-5
l1_loss = nn.L1Loss()
bce_loss = nn.BCELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

learning_rate = 2e-3

#optimizer
optimizer_d = torch.optim.Adam(D.parameters(), lr = learning_rate)
optimizer_g = torch.optim.Adam(G.parameters(), lr = learning_rate)

#define global variable
noise = Variable(torch.FloatTensor(_batch_size, 100), requires_grad = False)
image_profile128 = Variable(torch.FloatTensor(_batch_size, 3, 128, 128), requires_grad = False)
image_profile32 = Variable(torch.FloatTensor(_batch_size, 3, 32, 32), requires_grad = False)
image_profile64 = Variable(torch.FloatTensor(_batch_size, 3, 64, 64), requires_grad = False)
image_real = Variable(torch.FloatTensor(_batch_size, 3, 128, 128), requires_grad = False)
image_fake_flip = Variable(torch.FloatTensor(_batch_size, 3, 128, 128), requires_grad = False)
label = Variable(torch.FloatTensor(_batch_size), requires_grad = False)
real_label = 1
fake_label = 0

#light_cnn
light_cnn_model = LightCNN_9Layers(num_classes=1247)
if cuda:
    #light_cnn_model = torch.nn.DataParallel(light_cnn_model).cuda()
    light_cnn_model = light_cnn_model.cuda()

checkpoint_path = '/home/hezhenhao/LightCNN/OFD_checkpoint/lightCNN_80_checkpoint.pth.tar'
#checkpoint_path = '/home/hezhenhao/lightCNN_80_checkpoint.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

#light_cnn_model.load_state_dict(checkpoint['state_dict'])

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    #load params
light_cnn_model.load_state_dict(new_state_dict)

light_cnn_model.eval()


if cuda:
    D = torch.nn.DataParallel(D).cuda()
    G = torch.nn.DataParallel(G).cuda()
    #D = D.cuda()
    #G = G.cuda()
    l1_loss = l1_loss.cuda()
    bce_loss = bce_loss.cuda()
    cross_entropy_loss = cross_entropy_loss.cuda()
    noise = noise.cuda()
    image_profile128 = image_profile128.cuda()
    image_profile32 = image_profile32.cuda()
    image_profile64 = image_profile64.cuda()
    image_real = image_real.cuda()
    image_fake_flip = image_fake_flip.cuda()
    label = label.cuda()

iter = 200
D.train()
G.train()

cudnn.benchmark = True

#record the log in log_tp_gan.txt

log = open('log2.txt', "a")

for epoch in range(iter):
#     if (epoch != 0) & (epoch % 10 == 0):
#         for param_group in optimizer_d.param_groups:
#             param_group['lr'] = param_group['lr'] / 10.0
#         for param_group in optimizer_g.param_groups:
#             param_group['lr'] = param_group['lr'] / 10.0
    current_loss_g = 0
    current_loss_d = 0
    for batch_id, (out1, out2, out3, out4, image_label) in enumerate(train_loader):
        n_batch = out1.size()[0]
        #D
        D.zero_grad()
        optimizer_d.zero_grad()  
        
        #real
        image_real.data.resize_(out4.size()).copy_(out4)
        label.data.resize_(n_batch).fill_(real_label)
        output = D(image_real)
        loss_d_real = bce_loss(output, label)
        loss_d_real.backward()
        
        #fake
        image_profile128.data.resize_(out1.size()).copy_(out1)
        image_profile32.data.resize_(out2.size()).copy_(out2)
        image_profile64.data.resize_(out3.size()).copy_(out3)
        noise.data.resize_(n_batch, 100)
        noise.data.normal_(0, 1)
        image_fake, extra = G(image_profile128, image_profile32, image_profile64, noise)
        #image_fake, extra = G(image_profile128, noise)
        
        label.data.fill_(fake_label)
        output = D(image_fake.detach())
        loss_d_fake = bce_loss(output, label)
        loss_d_fake.backward()
        
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()
        
        #G
        G.zero_grad()
        optimizer_g.zero_grad()
        
        loss_pixel = l1_loss(image_fake, image_real)
        
        _ = image_fake.clone().data.cpu().numpy()
        _ = np.fliplr(_.reshape(-1, 128)).reshape(image_fake.size())
        image_fake_flip.data.resize_(image_fake.size()).copy_(torch.from_numpy(_.copy()))
        loss_sym = l1_loss(image_fake, image_fake_flip)
        
        label.data.fill_(real_label)
        output = D(image_fake)
        loss_adv = bce_loss(output, label)
        
        image_real_feat1, image_real_feat2 = light_cnn_model(image_real)
        image_fake_feat1, image_fake_feat2 = light_cnn_model(image_fake)
        image_real_feat1 = Variable(image_real_feat1.data.float(), requires_grad = False)
        image_real_feat2 = Variable(image_real_feat2.data.float(), requires_grad = False)
        image_fake_feat1 = Variable(image_fake_feat1.data.float(), requires_grad = False)
        image_fake_feat2 = Variable(image_fake_feat2.data.float(), requires_grad = False)
        
        loss_ip = l1_loss(image_real_feat1, image_fake_feat1) + l1_loss(image_real_feat2, image_fake_feat2)
        
        loss_tv = (torch.sum(torch.abs(image_fake[:, :, :, :-1] - image_fake[:, :, :, 1:])) \
            + torch.sum(torch.abs(image_fake[:, :, :-1, :] - image_fake[:, :, 1:, :]))) / n_batch
        
        image_label = Variable(image_label, requires_grad = False)
        if cuda:
            image_label = image_label.cuda()
        loss_cross_entropy = cross_entropy_loss(extra, image_label)
        
        loss_g = 10.0 * loss_pixel + lamda1 * loss_sym + lamda2 * loss_adv + lamda3 * loss_ip \
            + lamda4 * loss_tv \
            + alpha * loss_cross_entropy
        
        loss_g.backward()
        
        optimizer_g.step()
        
        current_loss_g = loss_g.data[0]
        current_loss_d = loss_d.data[0]
        
        if batch_id % 50 == 0:
            print('Epoch: [{}][{}/{}]\t loss_d: {} loss_g: {}\n'.format(epoch, batch_id, len(train_loader), 
                                                                              current_loss_d, current_loss_g))
            log.write('Epoch: [{}][{}/{}]\t loss_d: {} loss_g: {}\n'.format(epoch, batch_id, len(train_loader), 
                                                                              current_loss_d, current_loss_g))
            
    save_name = 'checkpoint/' + 'TP-GAN' + str(epoch+1) + '_checkpoint.pth'
    torch.save({
        'epoch': epoch + 1,
        'state_dict_d': D.state_dict(),
        'state_dict_g': G.state_dict()
    }, save_name)
    
log.close()

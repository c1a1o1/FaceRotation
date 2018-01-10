import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os

def recover_image(img):
    return (
        (
            img *
            np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)) +
            np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)

def test(_model, transform, test_list, save_root):
    _model.eval()
    
    with open(test_list, 'r') as file:
        count = 0
        for img_path in file.readlines():
            img_path = img_path.strip()
            image_profile = Image.open(img_path).convert('RGB')

            image_profile128 = transform(image_profile.resize((128, 128)))
            image_profile32 = transform(image_profile.resize((32, 32)))
            image_profile64 = transform(image_profile.resize((64, 64)))

            in1 = torch.FloatTensor(2, 3, 128, 128)
            in2 = torch.FloatTensor(2, 3, 32, 32)
            in3 = torch.FloatTensor(2, 3, 64, 64)
            in4 = torch.FloatTensor(2, 100)

            in1[0] = image_profile128
            in2[0] = image_profile32
            in3[0] = image_profile64
            in4 = in4.normal_(0, 1)

    #         #in1 = in1.cuda()
    #         #in2 = in2.cuda()
    #         #in3 = in3.cuda()
    #         #in4 = in4.cuda()

            in1 = torch.autograd.Variable(in1)
            in2 = torch.autograd.Variable(in2)
            in3 = torch.autograd.Variable(in3)
            in4 = torch.autograd.Variable(in4)
        
#         img_fake = transform_back(torch.Tensor(img_fake[0].data))
        
            count = count + 1
            img_fake, _ = _model(in1, in2, in3, in4)
            #img_fake, _ = model(in1, in4)
            img_fake = recover_image(img_fake.data.cpu().numpy())[0]
            img_fake = Image.fromarray(img_fake)
            img_fake.save(os.path.join(save_root, 'eval_' + str(count) + ".jpg"))
            
            #vutils.save_image(img_fake[0].data, os.path.join(save_root, 'eval_' + str(count) + ".jpg"), normalize=False)
            print("%d is done" % count)

def model():
    #load state_dict

    #G = Global(num_classes=1000)
    #G = torch.nn.DataParallel(G).cuda()
    G = Global()

    checkpoint_path = '/home/hezhenhao/TP-GAN144_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)

    #G.load_state_dict(checkpoint['state_dict_g'])

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict_g'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    #load params    
    G.load_state_dict(new_state_dict)
    return G

test(model(), transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
     test_list = '/home/hezhenhao/test_set.txt',
     save_root = '/home/hezhenhao/pic'
    )

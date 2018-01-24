import os
import argparse
from PIL import Image

import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from model.tp_gan import Global
from util import recover_image

def main(model_checkpoint, test_list, save_root):
    global args
    args = parser.parse_args()
    
    #get the value of the network setting
    model_checkpoint = args.checkpoint
    test_list = args.test_list
    save_root = args.save_root
    num_classes = args.num_classes

    #load state_dict
    model = Global(num_classes)
    if args.cuda:
        model = model.cuda()
    checkpoint = torch.load(model_checkpoint)
    #G.load_state_dict(checkpoint['state_dict_g'])

    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict_g'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #load params    
    G.load_state_dict(new_state_dict)
    """

    model.eval()
    
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

test(model(), transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
     test_list = '/home/hezhenhao/test_set.txt',
     save_root = '/home/hezhenhao/pic'
    )

if __name__ = '__main__':
    main()

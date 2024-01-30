import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from evaluator import Evaluator
from MedViT import MedViT_micro, MedViT_small, MedViT_base, MedViT_large
from tensorboardX import SummaryWriter
#from torchvision.models import resnet18, resnet50
from tqdm import trange
from dataloader import NEHOCTDataset, NEHOCTNoisyDataset
from focal_loss.focal_loss import FocalLoss

import random

random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)



def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, model_path, run): # resize , as_rgb
    lr = 0.0001  * (batch_size/128) ** 0.5
    gamma= 0.1
    milestones = [ ]

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image : image.convert('RGB')),
            #transforms.AugMix(),
            #transforms.TrivialAugmentWide(),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.8, fill= 0),
            transforms.RandomAffine(degrees=(-90, 90), shear=(-10, 10), translate=(0.2, 0.2), scale=(0.8, 1.2)), #, translate=(0.2, 0.2)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.8, 1.2),contrast = (0.8,1.2), saturation = (0.8, 1.2)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[.099], std=[.207]),
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image : image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.130], std=[.194]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda image : image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.130], std=[.194]),
        ]),
    }
     
    # If data_flag=ucsd change data directory
    train_dataset = NEHOCTNoisyDataset(args.data_file_train, args.data_root, args.class_name_column, args.image_column, data_transforms["train"])
    val_dataset = NEHOCTDataset(args.data_file_val, './NEH_UT_2021RetinalOCTDataset', args.class_name_column, args.image_column, data_transforms["validation"])
    test_dataset = NEHOCTDataset(args.data_file_test, './NEH_UT_2021RetinalOCTDataset', args.class_name_column, args.image_column, data_transforms["test"])
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8, pin_memory=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size, # batch_size=2*BATCH_SIZE
                                shuffle=False,
                                num_workers=8, pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8, pin_memory=True)

    print('==> Building and training model...')
    
    
    if model_flag == 'MedViT_micro':
        model =  MedViT_micro()
    elif model_flag == 'MedViT_small':
        model =  MedViT_small()
    elif model_flag == 'MedViT_base':
        model =  MedViT_base()
    elif model_flag == 'MedViT_large':
        model =  MedViT_large()
    else:
        raise NotImplementedError

    model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features= args.n_classes, bias=True) 
    model = model.to(device)

    train_evaluator = Evaluator(data_flag, args.task, 'train')
    val_evaluator = Evaluator(data_flag, args.task, 'val')
    test_evaluator = Evaluator(data_flag, args.task,  'test')

    criterion = FocalLoss(gamma=2)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, args.task, criterion, device, run, False, output_root)
        val_metrics = test(model, val_evaluator, val_loader, args.task, criterion, device, run, False, output_root)
        test_metrics = test(model, test_evaluator, test_loader, args.task, criterion, device, run, True, output_root)


        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0
    
    for epoch in trange(num_epochs):        
        # train_loss = train(model, train_loader, args.task, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, args.task, criterion, device, run, False)
        val_metrics = test(model, val_evaluator, val_loader, args.task, criterion, device, run, False)
        test_metrics = test(model, test_evaluator, test_loader, args.task, criterion, device, run, False)

        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        cur_acc = val_metrics[2]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', cur_auc)
            print('cur_best_epoch', best_epoch)
            print('cur_acc', cur_acc)
            print('test_acc', test_metrics[2])
            print('test_auc', test_metrics[1])


    path = os.path.join(output_root, 'best_model_checkpoint.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc
            }, path)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, args.task, criterion, device, run, False, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, args.task, criterion, device, run, False, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, args.task, criterion, device, run, True, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)

            m = torch.nn.Sigmoid()

            loss = criterion(m(outputs), targets)
        else:
            targets = targets.long().to(device)

            m = torch.nn.Softmax(dim=-1)

            loss = criterion(m(outputs), targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run, plot_conf_mat, save_folder=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)

                m = nn.Sigmoid()

                loss = criterion(m(outputs), targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.long().to(device)

                m = torch.nn.Softmax(dim=-1)

                loss = criterion(m(outputs), targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets),0)

        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if plot_conf_mat == True:
            if save_folder is not None:
                evaluator.save_results(y_true = y_true, y_score = y_score, task = task, outputpath = save_folder, threshold = 0.5)
        
        save_folder = None
        auc, acc = evaluator.evaluate(y_true, y_score, save_folder, run)

        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training model on UCSD/NEH')

    parser.add_argument('--data_flag',
                        default='NEH',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images to size 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='MedViT_micro',
                        help='choose backbone from MedViT_micro, MedViT_small, MedViT_base, MedViT_large',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument("--val_fold", type=str, default="split_0", help="Select the validation fold", choices=["fold_1", "fold_2", "fold_3"])
    
    parser.add_argument("--all_folds", default=["split_0", "split_1"], help="list of all folds available in data folder")

    parser.add_argument("--data_root", 
                default=None,
                help="Video data root with three subfolders (fold 1,2 and 3)")

    parser.add_argument("--data_file_train", 
                default=None,
                type=str)

    parser.add_argument("--data_file_val", 
                default=None,
                type=str)
    
    parser.add_argument("--data_file_test", 
                default=None,
                type=str)

    parser.add_argument("--class_name_column", 
                default=None,
                type=str)

    parser.add_argument("--image_column", 
                default=None,
                type=str)
    
    parser.add_argument("--n_classes", 
                default=None,
                type=int)
    
    parser.add_argument('--task',
                        default=None,
                        type=str)
    


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, model_path, run) # resize , as_rgb

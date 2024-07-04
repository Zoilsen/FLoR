import torch.nn as nn
from tensorboardX import SummaryWriter

from methods import backbone
import torch.nn.functional as F
import torch
from methods.protonet import ProtoNet


# --- conventional supervised training ---
class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, params, tf_path=None, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        
        self.params = params
        # feature encoder
        self.feature = model_func()
        
        self.feature.final_feat_dim = 512

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class, bias=False)

        elif loss_type == 'dist':
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_class = num_class
        #self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None



    def forward_loss(self, x, y, epoch):
        self.params.aux_container['epoch'] = epoch
        # forward feature extractor
        x = x.cuda()
        
        x_map = self.feature.forward(x, params=self.params) # [b, c, h, w]
        
        x = x_map.mean(dim=[2,3]) # [b, c]

        CLS_scores = self.classifier.forward(x) 

        # calculate loss
        y = y.cuda()
        loss_CLS = self.loss_fn(CLS_scores, y)

        return loss_CLS


    def train_loop(self, epoch, train_loader, optimizer, total_it):
        print_freq = len(train_loader) // 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            self.params.aux_container['batch_id'] = i
            optimizer.zero_grad()
            loss = self.forward_loss(x, y, epoch)
            
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()#data[0]

            #if (i + 1) % print_freq==0:
            #    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
            #if (total_it + 1) % 10 == 0:
            #    self.tf_writer.add_scalar('loss', loss.item(), total_it + 1)
            total_it += 1

        return total_it

    def test_loop(self, val_loader, params):
        #return -1 #no validation, just save model during iteration
        '''
        import subprocess as cmd
        cmd_reg = "python network_test.py -ckp_path %s/last_model.tar -stage pretrain -dataset %s -n_shot 5"
        acc_dict = {}
        novel_accs = []
        for d in params.eval_datasets:
            if d == 'ave':
                continue

            cmd_str = cmd_reg%(params.checkpoint_dir, d)
            output = cmd.getoutput(cmd_str)
            acc = float(output.split('Acc = ')[-1].split('%')[0])
            acc_dict[d] = acc

            if d != 'miniImagenet':
                novel_accs.append(acc)

        acc_dict['ave'] = sum(novel_accs) / len(novel_accs)
        '''
        
        import network_test
        params.ckp_path = params.checkpoint_dir + '/last_model.tar'
        train_dataset = params.dataset
        acc_dict = {}
        novel_accs = []
        for d in params.eval_datasets:
            if d == 'ave':
                continue

            params.dataset = d
            output = network_test.test_single_ckp(params)
            acc = float(output.split('Acc = ')[-1].split('%')[0])
            acc_dict[d] = acc

            if d != 'miniImagenet':
                novel_accs.append(acc)


        acc_dict['ave'] = sum(novel_accs) / len(novel_accs)

        params.dataset = train_dataset
        
        return acc_dict






















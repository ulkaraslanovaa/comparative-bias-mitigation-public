from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import torch.optim as optim

from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import EMA
import pandas as pd

class Learner(object):
    def __init__(self, args):
        data2model = {'cmnist': "MLP",
                       'cifar10c': "ResNet18",
                       'bffhq': "ResNet18",
                       'waterbirds': "ResNet18",
                       'waterbirds_noise': "ResNet18",
                       'fairface': "ResNet18"}

        data2batch_size = {'cmnist': 256,
                           'cifar10c': 256,
                           'bffhq': 64,
                           'waterbirds': 64,
                           'waterbirds_noise': 64,
                           'fairface': 64,}
        
        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True,
                           'waterbirds': True,
                           'waterbirds_noise': True,
                           'fairface': True,}

        if args.wandb:
            import wandb
            wandb.init(
                entity="debias-medimg",
                project=(
                    f"Learning_Debiased_Disentangled_{args.dataset}_ResNet18"
                ),
                id=None,
                resume=False,
                force=True,
                name=args.exp
            )

        run_name = args.exp
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')

        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args
        self.result_dir = self.log_dir

        self.train_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1,
            balanced=args.balanced
        )
        self.valid_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="test",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        train_target_attr = []

        if args.dataset == 'waterbirds' or args.dataset == 'waterbirds_noise' or args.dataset == 'fairface':
            train_target_attr = torch.LongTensor(self.train_dataset.labels)

        else:
            for data in self.train_dataset.data:
                train_target_attr.append(int(data.split('_')[-2]))
            train_target_attr = torch.LongTensor(train_target_attr)
        
        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.train_dataset = IdxDataset(self.train_dataset)

        seed = 42

        def seed_worker(worker_id):
            import random, numpy as np, torch
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g, # deterministic shuffling
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # define model and optimizer
        self.model_b = get_model(self.model, attr_dims[0]).to(self.device)
        self.model_d = get_model(self.model, attr_dims[0]).to(self.device)

        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        self.optimizer_d = torch.optim.Adam(
                self.model_d.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none') # loss is computed for each sample in the batch and returned as a tensor
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none')

        print(f'self.criterion: {self.criterion}')
        print(f'self.bias_criterion: {self.bias_criterion}')

        self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)

        print(f'alpha : {self.sample_loss_ema_d.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        self.best_valid_worst_acc_b, self.best_test_worst_acc_b = 0., 0.
        self.best_valid_worst_acc_d, self.best_test_worst_acc_d = 0., 0.
        print('finished model initialization....')


    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0] # in batch

        accs = total_correct/float(total_num)
        model.train()

        return accs

    # evaluation code for ours
    def evaluate_ours(self,model_b, model_l, data_loader, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0
        
        group_correct = torch.zeros(4, device=self.device, dtype=torch.long)
        group_count = torch.zeros(4, device=self.device, dtype=torch.long)

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0].to(self.device)
            bias  = attr[:, 1].to(self.device)
            data = data.to(self.device)

            group = (label.long() * 2 + bias.long())

            with torch.no_grad():
                if self.args.dataset == 'cmnist':
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                z_origin = torch.cat((z_l, z_b), dim=1)
                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin)
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
            total_correct += correct.sum()
            total_num += correct.shape[0]

            # per-group accuracy
            group_correct.scatter_add_(0, group, correct)
            group_count.scatter_add_(0, group, torch.ones_like(correct))


        avg_acc = (total_correct / float(total_num)).item()
        model_b.train()
        model_l.train()

        # worst-group accuracy (ignore empty groups)
        valid = group_count > 0
        group_acc = group_correct[valid].float() / group_count[valid].float()
        
        worst_group_acc, worst_idx_in_valid = group_acc.min(dim=0)
        worst_group_acc = worst_group_acc.item()
        print(f'Group accuracies: {group_acc.cpu().numpy()}, Worst group idx (in valid groups): {worst_idx_in_valid.item()}')
        return avg_acc, worst_group_acc

    def save_vanilla(self, epoch, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(epoch))
        state_dict = {
            'epoch': epoch,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{epoch} model saved ...')


    def save_ours(self, epoch, best=None, worst_group=False):
        if best and not worst_group:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        elif best and worst_group:
            model_path = os.path.join(self.result_dir, "best_worst_group_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(epoch))
        
        state_dict = {
            'epoch': epoch,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }

        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        
        if best and not worst_group:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        elif best and worst_group:
            model_path = os.path.join(self.result_dir, "best_worst_group_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(epoch))
            
        state_dict = {
            'epoch': epoch,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }

        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{epoch} model saved ...')

    def board_vanilla_loss(self, epoch, loss_b):
        if self.args.wandb:
            wandb.log({
                "loss_b_train": loss_b,
                "epoch": epoch
            })

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_b_train", loss_b, epoch)

    def board_ours_loss(self, epoch, loss_dis_conflict, loss_dis_align, loss_swap_conflict, loss_swap_align, lambda_swap):

        if self.args.wandb:
            wandb.log({
                "loss_dis_conflict":    loss_dis_conflict,
                "loss_dis_align":       loss_dis_align,
                "loss_swap_conflict":   loss_swap_conflict,
                "loss_swap_align":      loss_swap_align,
                "loss":                 (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align),
                "epoch":                epoch
            })

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_dis_conflict",  loss_dis_conflict, epoch)
            self.writer.add_scalar(f"loss/loss_dis_align",     loss_dis_align, epoch)
            self.writer.add_scalar(f"loss/loss_swap_conflict", loss_swap_conflict, epoch)
            self.writer.add_scalar(f"loss/loss_swap_align",    loss_swap_align, epoch)
            self.writer.add_scalar(f"loss/loss",               (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align), epoch)

    def board_vanilla_acc(self, epoch, inference=None):
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)

        print(f'epoch: {epoch}')

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b
            self.save_vanilla(epoch, best=True)
            # Save best validation accuracy to val_result.txt
            val_result_path = os.path.join(self.result_dir, "val_result.txt")
            with open(val_result_path, "w") as f:
                f.write(str(self.best_valid_acc_b.item()) + "\n")

        if self.args.wandb:
            wandb.log({
                "acc_b_valid": valid_accs_b,
                "best_acc_b_valid": self.best_valid_acc_b,
                "epoch": epoch
            })

        print(f'valid_b: {valid_accs_b}')

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, epoch)
            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, epoch)

        return valid_accs_b.item()


    def board_ours_acc(self, epoch, inference=None):
        avg_acc, worst_group_acc = self.evaluate_ours(self.model_b, self.model_l, self.valid_loader, model='label')

        if inference:
            print(f'valid avg acc: {avg_acc}, valid worst group acc: {worst_group_acc}')
            return

        # ----------------- CHANGED: use epoch instead of step -----------------
        if avg_acc >= self.best_valid_acc_d:
            self.best_valid_acc_d = avg_acc
            self.save_ours(epoch, best=True, worst_group=False)
            # Save best validation accuracy to val_avg_result.txt
            val_result_path = os.path.join(self.result_dir, "val_avg_result.txt")
            with open(val_result_path, "w") as f:
                f.write(str(self.best_valid_acc_d) + "\n")

        if worst_group_acc >= self.best_valid_worst_acc_d:
            self.best_valid_worst_acc_d = worst_group_acc
            # Save best worst-group validation accuracy to val_worst_result.txt
            self.save_ours(epoch, best=True, worst_group=True)
            val_worst_result_path = os.path.join(self.result_dir, "val_worst_result.txt")
            with open(val_worst_result_path, "w") as f:
                f.write(str(self.best_valid_worst_acc_d) + "\n")

        # ----------------- CHANGED: W&B logging uses epoch -----------------
        if self.args.wandb:
            wandb.log({
                "acc_d_valid": avg_acc,
                "best_acc_d_valid": self.best_valid_acc_d,
                "worst_group_acc_d_valid": worst_group_acc,
                "best_worst_group_acc_d_valid": self.best_valid_worst_acc_d,
                "epoch": epoch
            })

        # ----------------- CHANGED: tensorboard uses epoch -----------------
        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_d_valid", avg_acc, epoch)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, epoch)
            self.writer.add_scalar(f"acc/worst_group_acc_d_valid", worst_group_acc, epoch)
            self.writer.add_scalar(f"acc/best_worst_group_acc_d_valid", self.best_valid_worst_acc_d, epoch)

        print(f'valid_avg_acc: {avg_acc}, valid_worst_group_acc: {worst_group_acc}')
        return avg_acc, worst_group_acc
    
    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

    def train_vanilla(self, args):
        # training vanilla ...
        train_num = len(self.train_dataset.dataset)
        epoch = 0

        # ----------------- NEW: epoch-based loop -----------------
        for epoch in range(args.epochs):

            for index, data, attr, _ in tqdm(self.train_loader):

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]

                logit_b = self.model_b(data)
                loss_b_update = self.criterion(logit_b, label)
                loss = loss_b_update.mean()

                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()

            ##################################################
            #################### LOGGING #####################
            ##################################################

            # save checkpoint once per epoch
            self.save_vanilla(epoch)

            # log epoch-level loss
            self.board_vanilla_loss(epoch, loss_b=loss)

            # validation once per epoch
            self.board_vanilla_acc(epoch, epoch)

            print(f'finished epoch: {epoch}')


    def train_ours(self, args):
        epoch, cnt = 0, 0
        print('************** main training starts... ************** ')
        train_num = len(self.train_dataset)

        # self.model_l   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_l.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            if self.args.use_resnet20: # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.optimizer_l = torch.optim.Adam(
            self.model_l.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step, gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_step, gamma=args.lr_gamma)

        self.bias_criterion = GeneralizedCELoss(q=0.7)

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')

        # ---------------- NEW: epoch loop ----------------
        for epoch in range(args.epochs):

            for index, data, attr, image_path in tqdm(self.train_loader):

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx].to(self.device)

                # Feature extraction
                # Prediction by concatenating zero vectors (dummy vectors).
                # We do not use the prediction here.
                if args.dataset == 'cmnist':
                    z_l = self.model_l.extract(data)
                    z_b = self.model_b.extract(data)
                else:
                    z_b = []
                    # Use this only for reproducing CIFARC10 of LfF
                    if self.args.use_resnet20:
                        hook_fn = self.model_b.layer3.register_forward_hook(self.concat_dummy(z_b))
                        _ = self.model_b(data)
                        hook_fn.remove()
                        z_b = z_b[0]

                        z_l = []
                        hook_fn = self.model_l.layer3.register_forward_hook(self.concat_dummy(z_l))
                        _ = self.model_l(data)
                        hook_fn.remove()

                        z_l = z_l[0]

                    else:
                        hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                        _ = self.model_b(data)
                        hook_fn.remove()
                        z_b = z_b[0]

                        z_l = []
                        hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                        _ = self.model_l(data)
                        hook_fn.remove()

                        z_l = z_l[0]

                # z=[z_l, z_b]
                # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
                z_align = torch.cat((z_l.detach(), z_b), dim=1)

                # Prediction using z=[z_l, z_b]
                pred_conflict = self.model_l.fc(z_conflict)
                pred_align = self.model_b.fc(z_align)

                loss_dis_conflict = self.criterion(pred_conflict, label).detach()
                loss_dis_align = self.criterion(pred_align, label).detach()

                # EMA sample loss
                self.sample_loss_ema_d.update(loss_dis_conflict, index)
                self.sample_loss_ema_b.update(loss_dis_align, index)

                # class-wise normalize
                loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
                loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

                loss_dis_conflict = loss_dis_conflict.to(self.device)
                loss_dis_align = loss_dis_align.to(self.device)

                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0].to(self.device)
                    max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                    max_loss_align = self.sample_loss_ema_b.max_loss(c)
                    loss_dis_conflict[class_index] /= max_loss_conflict
                    loss_dis_align[class_index] /= max_loss_align

                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)                          # Eq.1 (reweighting module) in the main paper
                loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(self.device)              # Eq.2 W(z)CE(C_i(z),y)
                loss_dis_align = self.bias_criterion(pred_align, label)                                             # Eq.2 GCE(C_b(z),y)

                # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
                # ---------------- CHANGED: step → epoch ----------------
                if epoch >= args.curr_epoch:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = label[indices]     # y tilde

                    # Prediction using z_swap=[z_l, z_b tilde]
                    # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                    z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                    z_mix_align = torch.cat((z_l.detach(), z_b_swap), dim=1)

                    # Prediction using z_swap
                    pred_mix_conflict = self.model_l.fc(z_mix_conflict)
                    pred_mix_align = self.model_b.fc(z_mix_align)

                    loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(self.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
                    loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
                    lambda_swap = self.args.lambda_swap                                                             # Eq.3 lambda_swap_b

                else:
                    # before feature-level augmentation
                    loss_swap_conflict = torch.tensor([0]).float()
                    loss_swap_align = torch.tensor([0]).float()
                    lambda_swap = 0

                loss_dis  = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
                loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
                loss = loss_dis + lambda_swap * loss_swap                                                           # Eq.4 Total objective

                self.optimizer_l.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_l.step()
                self.optimizer_b.step()

            # ---------------- END OF EPOCH ----------------
            self.board_ours_loss(epoch,
                    loss_dis_conflict=loss_dis_conflict.mean(),
                    loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                    loss_swap_conflict=loss_swap_conflict.mean(),
                    loss_swap_align=args.lambda_swap_align * loss_swap_align.mean(),
                    lambda_swap=lambda_swap )
            self.board_ours_acc(epoch)

            if args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()
                
        self.save_ours(args.epochs - 1)


    def evaluate_to_csv(
        self,
        model_b,
        model_l,
        data_loader,
        out_csv: str,
    ):
        """
        Generic evaluation → CSV.

        Assumptions:
        - attr shape: (B, >=2)
            attr[:, 0] → target label (y_true)
            attr[:, 1] → spurious / protected attribute (attr_true)
        - Binary classification (2 logits)

        CSV columns:
            dataset, model, img_id,
            y_true, attr_true,
            y_pred,
            p_0, p_1, p_max,
            correct
        """

        model_b.eval()
        model_l.eval()

        rows = []

        for data, attr, img_path in tqdm(data_loader, leave=False):
            data = data.to(self.device)

            # labels & attributes
            y_true = attr[:, 0].to(self.device)
            attr_true = attr[:, 1].to(self.device)

            with torch.no_grad():
                # ----- feature extraction -----
                if self.args.dataset == "cmnist":
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []

                    hook = self.model_l.avgpool.register_forward_hook(
                        self.concat_dummy(z_l)
                    )
                    _ = self.model_l(data)
                    hook.remove()
                    z_l = z_l[0]

                    hook = self.model_b.avgpool.register_forward_hook(
                        self.concat_dummy(z_b)
                    )
                    _ = self.model_b(data)
                    hook.remove()
                    z_b = z_b[0]

                z = torch.cat((z_l, z_b), dim=1)
                logits = model_l.fc(z)
                probs = torch.softmax(logits, dim=1)

            # predictions & confidence
            y_pred = probs.argmax(dim=1)
            p_0 = probs[:, 0]
            p_1 = probs[:, 1]
            p_max = probs.max(dim=1).values
            correct = (y_pred == y_true)

            # write rows
            for i in range(len(y_true)):
                rows.append({
                    "img_path": str(img_path[i]),
                    "y_true": int(y_true[i]),
                    "attr_true": int(attr_true[i]),
                    "y_pred": int(y_pred[i]),
                    "p_0": float(p_0[i]),
                    "p_1": float(p_1[i]),
                    "p_max": float(p_max[i]),
                    "correct": int(correct[i]),
                })

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

        model_b.train()
        model_l.train()

        print(f"Saved evaluation CSV to {out_csv}")

    def test_ours(self, args):
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        if args.worst_group:
            self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_worst_group_model_l.th'))['state_dict'])
            self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_worst_group_model_b.th'))['state_dict'])
        
        else:
            self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_l.th'))['state_dict'])
            self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.evaluate_to_csv(self.model_b, self.model_l, self.test_loader, args.out_csv)

    def test_vanilla(self, args):
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model.th'))['state_dict'])
        test_accs_b = self.evaluate(self.model_b, self.test_loader)
        return test_accs_b.item()
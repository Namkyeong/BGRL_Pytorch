from torch_geometric.data import DataLoader


import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch import optim
from tensorboardX import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

torch.manual_seed(0)

import models
import utils
import data

import os.path as osp
import os
import sys


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._init()
        self.writer = SummaryWriter(log_dir="runs/BGRL_dataset({})_layers_({})".format(args.name, args.layers))

    def _init(self):
        args = self._args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._aug = utils.Augmentation(float(args.aug_params[0]),float(args.aug_params[1]),float(args.aug_params[2]),float(args.aug_params[3]))
        self._dataset = data.Dataset(root=args.root, name=args.name, num_parts=args.init_parts,
                                     final_parts=args.final_parts, augumentation=self._aug)
        self._loader = DataLoader(
            dataset=self._dataset)  # [self._dataset.data]
        print(f"Data: {self._dataset.data}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x1.shape[1]] + hidden_layers
        self._model = models.BGRL(layer_config=layers, pred_hid=args.pred_hid, dropout=args.dropout, epochs=args.epochs).to(self._device)
        print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay= 1e-5)
        # learning rate
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (self._args.epochs - 1000))) * 0.5
        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda = scheduler)

    def train(self):
        # get initial test results
        print("start training!")
        print("Initial Evaluation...")
        self.infer_embeddings()
        dev_best, dev_std_best, test_best, test_std_best = self.evaluate()
        self.writer.add_scalar("accs/val_acc", dev_best, 0)
        self.writer.add_scalar("accs/test_acc", test_best, 0)
        print("validation: {:.4f}, test: {:.4f}".format(dev_best, test_best))
        
        # start training
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                batch_data.to(self._device)
                v1_output, v2_output, loss = self._model(
                    x1=batch_data.x1, x2=batch_data.x2, edge_index_v1=batch_data.edge_index1, edge_index_v2=batch_data.edge_index2,
                    edge_weight_v1=batch_data.edge_attr1, edge_weight_v2=batch_data.edge_attr2)
                
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._scheduler.step()
                self._model.update_moving_average()
                sys.stdout.write('\rEpoch {}/{}, batch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, bc + 1,
                                                                                  self._dataset.final_parts, loss.data, self._optimizer.param_groups[0]['lr']))
                sys.stdout.flush()
            self.writer.add_scalar("loss/training_loss", loss, epoch)
            if (epoch + 1) % self._args.cache_step == 0:
                print("")
                print("\nEvaluating {}th epoch..".format(epoch + 1))
                path = osp.join(self._dataset.model_dir,
                                f"model.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
                self.infer_embeddings()
                dev_acc, dev_std, test_acc, test_std = self.evaluate()

                if dev_best < dev_acc:
                    dev_best = dev_acc
                    dev_std_best = dev_std
                    test_best = test_acc
                    test_std_best = test_std

                self.writer.add_scalar("stats/learning_rate", self._optimizer.param_groups[0]["lr"] , epoch + 1)
                self.writer.add_scalar("accs/val_acc", dev_acc, epoch + 1)
                self.writer.add_scalar("accs/test_acc", test_acc, epoch + 1)
                print("validation: {:.4f}, test: {:.4f} \n".format(dev_best, test_best))
        
        f = open("BGRL_dataset({}).txt".format(self._args.name), "a")
        f.write("best valid acc : {} best valid std : {} best test acc : {} best test std : {} \n".format(dev_best, dev_std_best, test_best, test_std_best))
        f.close()

        print()
        print("Training Done!")
        
    def infer_embeddings(self):
        
        self._model.train(False)
        self._embeddings = self._labels = None
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            v1_output, v2_output, _ = self._model(
                x1=batch_data.test_x, x2=batch_data.x2,
                edge_index_v1=batch_data.test_edge_index,
                edge_index_v2=batch_data.edge_index2,
                edge_weight_v1=batch_data.test_edge_attr,
                edge_weight_v2=batch_data.edge_attr2)
            emb = v1_output.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])
                
    
    def evaluate(self):
        """
        Used for producing the results of Experiment 3.2 in the BGRL paper. 
        """
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]
    
        dev_accs, test_accs = [], []
        
        for i in range(20):

            self._train_mask = self._dataset[0].train_mask[i]
            self._dev_mask = self._dataset[0].val_mask[i]
            if self._args.name == "WikiCS":
                self._test_mask = self._dataset[0].test_mask
            else :
                self._test_mask = self._dataset[0].test_mask[i]

            classifier = models.LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            
            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() / self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() / self._labels[self._test_mask].shape[0]).detach().cpu().numpy()
        
            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)
        
        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        return dev_acc, dev_std, test_acc, test_std

def train_eval(args):
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.writer.close()


def main():
    args = utils.parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()

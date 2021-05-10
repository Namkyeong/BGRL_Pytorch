from torch_geometric.data import DataLoader

import numpy as np

import torch

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
        self._model = models.BGRL(layer_config=layers, dropout=args.dropout, epochs=args.epochs).to(self._device)
        print(self._model)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(), lr=args.lr)

    def train(self):
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
                self._model.update_moving_average()
                sys.stdout.write('\rEpoch {}/{}, batch {}/{}, loss {:.4f}'.format(epoch + 1, self._args.epochs, bc + 1,
                                                                                  self._dataset.final_parts, loss.data))
                sys.stdout.flush()
            if (epoch + 1) % self._args.cache_step == 0:
                path = osp.join(self._dataset.model_dir,
                                f"model.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
        print("Training Done!")
        
    def infer_embeddings(self):
        outputs = []
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            v1_output, v2_output, _ = self._model(
                x1=batch_data.x1, x2=batch_data.x2,
                edge_index_v1=batch_data.edge_index1,
                edge_index_v2=batch_data.edge_index2,
                edge_weight_v1=batch_data.edge_attr1,
                edge_weight_v2=batch_data.edge_attr2)
            emb = torch.cat([v1_output, v2_output], dim=1).detach()
            y = batch_data.y.detach()
            trm = batch_data.train_mask.detach()
            dem = batch_data.dev_mask.detach()
            tem = batch_data.test_mask.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
                self._train_mask, self._dev_mask, self._test_mask = trm, dem, tem
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])
                self._train_mask = torch.cat([self._train_mask, trm])
                self._dev_mask = torch.cat([self._dev_mask, dem])
                self._test_mask = torch.cat([self._test_mask, tem])
    
    def evaluate(self):
        """
        Used for producing the results of Experiment 3.2 in the BGRL paper. 
        """
        print("Evaluating ...")
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]
        
        dev_accs, test_accs = [], []
        
        for i in range(50):
            classifier = models.LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

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
            print("Finished iteration {:02} of the logistic regression classifier. Validation accuracy {:.2f} test accuracy {:.2f}".format(i + 1, dev_acc, test_acc))

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)
        
        print('Average validation accuracy: {:.2f} with std: {}'.format(dev_accs.mean(), dev_accs.std()))
        print('Average test accuracy: {:.2f} with std: {:.2f}'.format(test_accs.mean(), test_accs.std()))
    

def train_eval(args):
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.infer_embeddings()
    trainer.evaluate()


def main():
    args = utils.parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()

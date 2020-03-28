import torch
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

IMG_LEN = 1024
TXT_LEN = 300
BATCH_SIZE = 512


class TopicsDecorator:
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, epochs=1, validation_data=None, weight_norm=False):
        x_img = X[0]
        x_txt = X[1]
        print('fit on ' + str(X[0].shape[0]) + ' objects')

        p = np.random.permutation(len(x_img))
        x_img = x_img[p]
        x_txt = x_txt[p]
        y = y[p]

        x_img_train_t = torch.tensor(x_img).float()
        x_txt_train_t = torch.tensor(x_txt).float()
        y_train_t = torch.tensor(y)
        train_ds = TensorDataset(x_img_train_t, x_txt_train_t, y_train_t)

        cur_batch_size = BATCH_SIZE + (len(x_img) % BATCH_SIZE) // ((len(x_img) // BATCH_SIZE)) + 1
        train_loader = DataLoader(train_ds, batch_size=cur_batch_size)

        if validation_data is not None:
            x_img_val_t = torch.tensor(validation_data[0][0]).float()
            x_txt_val_t = torch.tensor(validation_data[0][1]).float()
            y_val_t = torch.tensor(validation_data[1])
            val_ds = TensorDataset(x_img_val_t, x_txt_val_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        for epoch in range(epochs):
            self.model.train()

            loss_sum = 0.0
            loss_count = 0

            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.optimizer.zero_grad()
                output = self.model(x_img_cur, x_txt_cur)
                if not weight_norm:
                    loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                else:
                    loss_non_reducted = F.nll_loss(output, torch.argmax(y_cur, dim=1), reduction='none')

                    output_detached = output.detach()
                    predictions = torch.exp(output_detached)
                    weights = torch.t(torch.tensor(stats.entropy(torch.t(predictions))))

                    norm_loss = loss_non_reducted * weights
                    norm_loss /= sum(norm_loss) / sum(loss_non_reducted)

                    assert abs(sum(norm_loss) - sum(loss_non_reducted)) < 1e-2

                    loss = torch.mean(loss_non_reducted)

                loss.backward()
                loss_sum += loss

                loss_count += 1

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            print('epoch:', epoch, 'train_loss:', loss, 'average train loss', loss_sum / loss_count)

            if validation_data is not None:
                self.model.eval()

                correct = 0
                total = 0
                loss_sum = 0.0
                loss_count = 0

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_cur in val_loader:
                        output = self.model(x_img_cur, x_txt_cur)
                        loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                        loss_sum += loss
                        loss_count += 1
                        for idx, i in enumerate(output):
                            if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                                correct += 1
                            total += 1

                print('val_acc:', correct / total, 'val_avg_loss:', loss_sum / loss_count)

    def predict(self, X, with_dropout=False):

        if with_dropout:
            self.model.train()
        else:
            self.model.eval()

        x_img_t = torch.tensor(X[0])
        x_txt_t = torch.tensor(X[1])

        ds = TensorDataset(x_img_t, x_txt_t)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

        predictions = torch.tensor([])
        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                outputs = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions = torch.cat((predictions, outputs), 0)

        return predictions.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted)

    def evaluate(self, X, y, verbose=0):
        y_predicted = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        print('val acc', correct / total)

        return loss, correct / total

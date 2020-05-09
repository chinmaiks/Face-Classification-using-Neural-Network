from constants import *
import pickle
import os
import sys
import time

import numpy as np

import matplotlib.pyplot as plt

from torch import nn, optim, FloatTensor, LongTensor, no_grad, save
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import logging


log = logging.getLogger("my-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


class Data(Dataset):
    def __init__(self, data_type='train', data_aug=False, normalize=True):
        self.data_aug = data_aug
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        if normalize:
            tr_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "tr_face_data.pkl", "rb"))
            tr_non_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "tr_non_face_data.pkl", "rb"))

            te_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "te_face_data.pkl", "rb"))
            te_non_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "te_non_face_data.pkl", "rb"))
        else:
            tr_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "tr_face_data_not_norm.pkl", "rb"))
            tr_non_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "tr_non_face_data_not_norm.pkl", "rb"))

            te_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "te_face_data_not_norm.pkl", "rb"))
            te_non_face_data = pickle.load(open(DIR_PATH + OUTPUT_DIR + "te_non_face_data_not_norm.pkl", "rb"))

        tr_face_labels = pickle.load(open(DIR_PATH + OUTPUT_DIR + "tr_face_labels.pkl", "rb"))
        tr_non_face_labels = pickle.load(
            open(DIR_PATH + OUTPUT_DIR + "tr_non_face_labels.pkl", "rb"))
        te_face_labels = pickle.load(open(DIR_PATH + OUTPUT_DIR + "te_face_labels.pkl", "rb"))
        te_non_face_labels = pickle.load(
            open(DIR_PATH + OUTPUT_DIR + "te_non_face_labels.pkl", "rb"))

        X = np.concatenate((tr_face_data, tr_non_face_data))
        y = np.concatenate((tr_face_labels, tr_non_face_labels))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        if data_type == TRAIN_DATA_TYPE:
            self.data = np.array(list(zip(X_train, y_train)))
        elif data_type == TEST_DATA_TYPE:
            self.data = np.array(list(zip(np.concatenate((te_face_data, te_non_face_data)),
                                          np.concatenate((te_face_labels, te_non_face_labels)))))
        elif data_type == VALID_DATA_TYPE:
            self.data = np.array(list(zip(X_val, y_val)))

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.data_aug:
            x = self.transform(image)
        else:
            x = TF.to_tensor(image)
        return x, label

    def __len__(self):
        return len(self.data)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RunModel:
    def __init__(self, LEARNING_RATE, WEIGHT_DECAY, DATA_AUG_BOOL, NORMALIZE):
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.DATA_AUG_BOOL = DATA_AUG_BOOL
        self.NORMALIZE = NORMALIZE

        self.net = LeNet()
        log.info("Summary of the model:{}".format(self.net))

        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        os.makedirs(DIR_PATH + MODEL_OUTPUTS, exist_ok=True)

        fig_name = os.path.join(DIR_PATH + MODEL_OUTPUTS, "accuracy_lr_{}_decay_{}_aug_{}_norm_{}.png". \
                            format(self.LEARNING_RATE, self.WEIGHT_DECAY, self.DATA_AUG_BOOL, self.NORMALIZE))
        if os.path.isfile(fig_name):
            os.remove(fig_name)
            log.info("Removed existing accuracy curves")

        fig_name = os.path.join(DIR_PATH + MODEL_OUTPUTS, "loss_lr_{}_decay_{}_aug_{}_norm_{}.png". \
                            format(self.LEARNING_RATE, self.WEIGHT_DECAY, self.DATA_AUG_BOOL, self.NORMALIZE))
        if os.path.isfile(fig_name):
            os.remove(fig_name)
            log.info("Removed existing loss curves")

    def train(self, train_data_loader, val_data_loader):

        loss_func = nn.CrossEntropyLoss()
        optimization = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            epoch_training_loss = 0.0
            epoch_val_loss = 0.0
            num_batches_train = 0
            train_acc = 0.0

            for batch_num, training_batch in enumerate(train_data_loader):
                inputs, labels = training_batch
                inputs, labels = Variable(inputs).type(FloatTensor), Variable(labels).type(LongTensor)

                optimization.zero_grad()
                forward_output = self.net(inputs)

                predicted_val = forward_output.data.numpy()
                predicted_val = np.argmax(predicted_val, axis=1)
                train_acc += accuracy_score(labels, predicted_val)

                train_loss = loss_func(forward_output, labels)
                train_loss.backward()
                optimization.step()

                epoch_training_loss += train_loss.item()
                num_batches_train += 1
            train_acc /= num_batches_train
            self.training_accuracies.append(train_acc)
            self.training_losses.append(epoch_training_loss / num_batches_train)

            # calculate validation set accuracy
            val_acc = 0.0
            num_batches_val = 0
            best_valid_loss = float('inf')
            self.net.eval()
            with no_grad():
                for batch_num, validation_batch in enumerate(val_data_loader):
                    num_batches_val += 1
                    inputs, actual_val = validation_batch

                    forward_output = self.net(Variable(inputs)).type(FloatTensor)
                    predicted_val = forward_output.data.numpy()
                    predicted_val = np.argmax(predicted_val, axis=1)

                    val_loss = loss_func(forward_output, Variable(actual_val).type(LongTensor))

                    epoch_val_loss += val_loss.item()
                    val_acc += accuracy_score(actual_val, predicted_val)
                    if epoch_val_loss/num_batches_val < best_valid_loss:
                        best_valid_loss = epoch_val_loss/num_batches_val
                        save(self.net.state_dict(), os.path.join(DIR_PATH + MODEL_OUTPUTS, \
                        'model_lr_{}_decay_{}_aug_{}_norm_{}.pt'.format(self.LEARNING_RATE, \
                        self.WEIGHT_DECAY, self.DATA_AUG_BOOL, self.NORMALIZE)))

                val_acc /= num_batches_val
                self.validation_accuracies.append(val_acc)
                self.validation_losses.append(epoch_val_loss / num_batches_val)
            log.info("epoch: {}, train_loss: {:.5f}, train_accuracy: {:.5f}, val_loss: {:.5f}, val_accuracy: {:.5f}". \
                  format(epoch, epoch_training_loss / num_batches_train, train_acc, epoch_val_loss / num_batches_val, val_acc))

    def test(self, test_loader):
        self.net.eval()
        targets = []
        preds = []

        with no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = Variable(data).type(FloatTensor), Variable(target).type(LongTensor)
                output = self.net(data)

                pred = output.argmax(dim=1)

                targets += list(target.cpu().numpy())
                preds += list(pred.cpu().numpy())

        accuracy = accuracy_score(targets, preds)
        return accuracy, targets, preds

    def plot(self):
        plt.figure()
        plt.clf()
        plt.plot(list(range(EPOCHS)), self.training_losses, label='training loss')
        plt.plot(list(range(EPOCHS)), self.validation_losses, label='validation loss')

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='upper right')
        plt.title("Loss Curves")

        fig_name = os.path.join(DIR_PATH + MODEL_OUTPUTS, "loss_lr_{}_decay_{}_aug_{}_norm_{}.png". \
                        format(self.LEARNING_RATE, self.WEIGHT_DECAY, self.DATA_AUG_BOOL, self.NORMALIZE))

        plt.savefig(fig_name)
        plt.clf()
        log.info("Saved new loss curves")
        plt.close()

        plt.figure()
        plt.clf()
        plt.plot(list(range(EPOCHS)), self.training_accuracies, label='training accuracy')
        plt.plot(list(range(EPOCHS)), self.validation_accuracies, label='validation accuracy')

        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right')
        plt.title("Accuracy Curves")

        fig_name = os.path.join(DIR_PATH + MODEL_OUTPUTS, "accuracy_lr_{}_decay_{}_aug_{}_norm_{}.png". \
                        format(self.LEARNING_RATE, self.WEIGHT_DECAY, self.DATA_AUG_BOOL, self.NORMALIZE))

        plt.savefig(fig_name)
        plt.clf()
        log.info("Saved new accuracy curves")
        plt.close()


def run(LEARNING_RATE, WEIGHT_DECAY, DATA_AUG_BOOL, NORMALIZE):
    log.info("*******************Start*******************")
    run_model = RunModel(LEARNING_RATE, WEIGHT_DECAY, DATA_AUG_BOOL, NORMALIZE)
    train_data_obj = Data(data_type=TRAIN_DATA_TYPE, data_aug=DATA_AUG_BOOL, normalize=NORMALIZE)
    val_data_obj = Data(data_type=VALID_DATA_TYPE, data_aug=DATA_AUG_BOOL, normalize=NORMALIZE)
    test_data_obj = Data(data_type=TEST_DATA_TYPE, data_aug=DATA_AUG_BOOL, normalize=NORMALIZE)

    train_data_loader = DataLoader(train_data_obj, shuffle=True, batch_size=BATCH_SIZE)
    val_data_loader = DataLoader(val_data_obj, shuffle=True, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data_obj, shuffle=True, batch_size=BATCH_SIZE)
    log.info("Loaded train, val, test data")

    log.info("*******************Starting training process*******************")
    run_model.train(train_data_loader, val_data_loader)

    log.info("*******************Training process ended*******************")
    log.info("*******************Starting testing process*******************")
    accuracy, targets, preds = run_model.test(test_data_loader)
    log.info("Accuracy on test set is: {}".format(accuracy))
    log.info("*******************Testing process ended*******************")
    log.info("*******************Plotting loss and accuracy curves*******************")
    run_model.plot()
    log.info("*******************Loss and accuracy curves plotted*******************")
    log.info("*******************End*******************")
    return accuracy, targets, preds


if __name__ == '__main__':
    LEARNING_RATE = float(sys.argv[1])
    WEIGHT_DECAY = float(sys.argv[2])
    DATA_AUG_BOOL = "True" == sys.argv[3]
    NORMALIZE = "True" == sys.argv[4]
    log.info("Running with {} learning rate, {} weight decay, data augmentation as {} and data normalization as {}".\
             format(LEARNING_RATE, WEIGHT_DECAY, DATA_AUG_BOOL, NORMALIZE))
    run(LEARNING_RATE, WEIGHT_DECAY, DATA_AUG_BOOL, NORMALIZE)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
from torch.utils.data.sampler import SubsetRandomSampler

from VNN import VNN_utils as uf
from VNN.CustomizedLinear import CustomizedLinear


class Net(nn.Module):
    def __init__(self, layer_infos, matrix_list, device):
        super(Net, self).__init__()
        self.pc1 = CustomizedLinear(matrix_list[0].float().t(), device)
        self.pc2 = CustomizedLinear(matrix_list[1].float().t(), device)
        self.pc3 = CustomizedLinear(matrix_list[2].float().t(), device)
        self.pc1_bn = nn.BatchNorm1d(len(layer_infos[1]))
        self.pc2_bn = nn.BatchNorm1d(len(layer_infos[2]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = torch.tanh(self.pc1_bn(self.pc1(x)))
        out2 = torch.tanh(self.pc2_bn(self.pc2(out1)))
        out3 = self.pc3(out2)
        y_pred = self.sigmoid(out3)
        return y_pred


class Net_unmask(nn.Module):
    def __init__(self, layer_infos):
        super(Net_unmask, self).__init__()
        self.fc1 = nn.Linear(len(layer_infos[0]), len(layer_infos[1]))
        self.fc2 = nn.Linear(len(layer_infos[1]), len(layer_infos[2]))
        self.fc3 = nn.Linear(len(layer_infos[2]), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = torch.tanh(self.fc1(x))
        out2 = torch.tanh(self.fc2(out1))
        out3 = self.fc3(out2)
        y_pred = self.sigmoid(out3)
        return y_pred


def set_inputs(oversampling_ratio):
    relation_files = ['GE_MF', 'MF_BP', 'BP_PH']

    layer_info = uf.set_layer_info(relation_files)
    gene_index = layer_info[0]

    matrix_list = uf.set_mask_matrics(relation_files, layer_info)

    disease_drug = uf.get_diseas_drug()
    deg_drug_train = uf.get_DEG_drug(disease_drug)

    data_set = uf.set_input_data(deg_drug_train, gene_index, 'train_data')

    # oversampling
    data_set = uf.oversampling(data_set, oversampling_ratio)

    return [layer_info, matrix_list, data_set]


def set_network(layer_info, matrix_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # construct network

    net = Net(layer_info, matrix_list, device)
    net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())

    return [net, criterion, optimizer, device]


def train_epoch(model, optimizer, criterion, train_loader, test_loader, device, n_epochs):
    # number of epochs to train the model

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model #
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the batch loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * inputs.size(0)

        # Validating the model
        for i, data in enumerate(test_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the batch loss
            loss = criterion(output, labels)
            # update average validation loss
            valid_loss += loss.item() * inputs.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(test_loader.dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))


def get_trained_network(n_epochs, oversampling_ratio):
    if uf.manage_obj('trained_networks/VNN_%d_%.2f' % (n_epochs, oversampling_ratio)) is False:

        layer_info, matrix_list, data_set = set_inputs(oversampling_ratio)

        net, criterion, optimizer, device = set_network(layer_info, matrix_list)

        train_loader, test_loader = uf.get_train_test_valid_set(data_set, 0.3)

        train_loader = td.DataLoader(data_set, batch_size=50, shuffle=True, num_workers=10)

        train_epoch(net, optimizer, criterion, train_loader, test_loader, device, n_epochs)

        torch.save(net, '../data/objects/trained_networks/VNN_%d_%.2f.obj' % (n_epochs, oversampling_ratio))
        net.eval()
        return net
    else:
        net = torch.load('../data/objects/trained_networks/VNN_%d_%.2f.obj' % (n_epochs, oversampling_ratio))
        net.eval()
        return net


def get_auroc(n_epochs, cross_validation, oversampling_ratio):

    if cross_validation is False:

        layer_info, matrix_list, data_set = set_inputs(oversampling_ratio)

        net, criterion, optimizer, device = set_network(layer_info, matrix_list)

        train_loader, test_loader = uf.get_train_test_valid_set(data_set, 0.3)

        ########################################
        #########      TRAINING       ##########
        ########################################
        train_epoch(net, optimizer, criterion, train_loader, test_loader, device, n_epochs)
        ########################################

        net.eval()
        result = uf.get_sigmoid_auroc(test_loader, device, net)
        result = [n_epochs, result]
        return result
    else:
        layer_info, matrix_list, data_set = set_inputs(oversampling_ratio)
        cross_validation_idx = uf.cross_validation_index(data_set)
        result = list()
        for train_idx, test_idx in cross_validation_idx:

            net, criterion, optimizer, device = set_network(layer_info, matrix_list)

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            train_loader = td.DataLoader(data_set, batch_size=50, sampler=train_sampler, num_workers=10)
            test_loader = td.DataLoader(data_set, batch_size=50, sampler=test_sampler, num_workers=10)

            ########################################
            #########      TRAINING       ##########
            ########################################
            train_epoch(net, optimizer, criterion, train_loader, test_loader, device, n_epochs)
            ########################################

            net.eval()
            scores = uf.get_sigmoid_auroc(test_loader, device, net)
            net.train()

            scores = [n_epochs, scores]
            result.append(scores)
        return result


if __name__ == '__main__':
    print(get_auroc(30, False, 0.06))  # single_task(training:7, test:3)
    print(get_auroc(30, True, 0.06))    # 10_fold cross_validation

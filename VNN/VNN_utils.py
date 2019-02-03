import pickle
import os

import torch
import torch.utils.data as td
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import KFold


def read_file(file_path):
    result = list()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            result.append(line.split('\t'))

    return result


def write_tsv(result, output_file):
    output = open(output_file, 'w')
    for subset in result:
        line = ""
        # print subset
        for i in range(0, len(subset)):
            if i != len(subset) - 1:
                line += str(subset[i]) + "\t"
            else:
                line += str(subset[i]) + "\n"
        output.write(line)
    output.close()


def get_row_column(input_list, input_list_name):
    row = list()
    column = list()

    row_name = input_list_name + '_' + input_list_name.split('_')[1]
    column_name = input_list_name + '_' + input_list_name.split('_')[0]
    if manage_obj(row_name) is False or manage_obj(column_name) is False:
        for element in input_list:
            row.append(element[1])
            column.append(element[0])
        row = list(set(row))
        column = list(set(column))
        save_obj(row, row_name)
        save_obj(column, column_name)
    else:
        row = reload_obj(row_name)
        column = reload_obj(column_name)

    return [row, column]


def manage_obj(obj_name):
    if os.path.isfile('../data/objects/%s.obj' % obj_name):
        return True
    return False


def save_obj(obj, obj_name):
    with open('../data/objects/%s.obj' % obj_name, 'wb') as f:
        pickle.dump(obj, f)


def reload_obj(obj_name):
    with open('../data/objects/%s.obj' % obj_name, 'rb') as f:
        result = pickle.load(f)
    return result


def set_layer_info(relation_file_names):
    if manage_obj('network_layers/layer_infos') is False:
        layer_list = list()

        for relation in relation_file_names:

            relation_file = read_file('../data/network/VNN/%s.txt' % relation)
            column = sorted(list(set([element[0] for element in relation_file])))
            row = sorted(list(set([element[1] for element in relation_file])))
            if column not in layer_list:
                layer_list.append(column)
            if row not in layer_list:
                layer_list.append(row)

        save_obj(layer_list, 'network_layers/layer_info')

    else:
        layer_list = reload_obj('network_layers/layer_info')

    return layer_list


def get_diseas_drug():  # drug, disease name, phid
    disease_drug = read_file('../data/drug/target_phenotype/target_phenotype_drug_list.txt')
    disease_drug = [drug[0] for drug in disease_drug]
    return disease_drug


def get_DEG_drug(disease_drug):  # cell line, drug, [DEGs]
    result = read_file('../data/drug/target_phenotype/Drug_DEGs.txt')
    for line in result:

        line[1] = line[1].split("|")  # parsing DEGs

        if line[0] in disease_drug:  # append drug effect label: 1: effective, 0: non-effective
            line.append(1)
        else:
            line.append(0)

    return result


def convert_matrix(row, column, input_list):  # result: (row: previous layer, column: next layer), matrix name = 'row_column'
    result = list()

    for i in range(len(row)):

        new_row = [0]*len(column)

        relations = (j for j in range(len(column)) if [column[j], row[i]] in input_list)
        for relation in relations:
            new_row[relation] = 1

        result.append(new_row)

    return result


def set_mask_matrics(relation_files, layer_infos):
    if manage_obj('mask_matrix/matrix_list') is False:
        matrix_list = list()

        for i in range(len(relation_files)):
            matrix_file = read_file('../data/drug/target_phenotype/%s.txt' % relation_files[i])
            matrix = convert_matrix(layer_infos[i + 1], layer_infos[i], matrix_file)
            matrix_list.append(torch.tensor(matrix))

        save_obj(matrix_list, 'mask_matrix/matrix_list')
    else:
        matrix_list = reload_obj('mask_matrix/matrix_list')
    return matrix_list


def set_input_data(deg_drug, gene_layer, data_name):  # [[input],[label]]
    if manage_obj("input_data/%s" % data_name) is False:
        data_input = list()
        data_label = [relation[-1] for relation in deg_drug]    # label about effectiveness

        for line in deg_drug:
            degs = (deg for deg in line[2] if deg in gene_layer)
            input_feature = [0] * len(gene_layer)
            for deg in degs:
                input_feature[gene_layer.index(deg)] = 1    # check DEG loci
            data_input.append(input_feature)  # DEG vector

        trainData = td.TensorDataset(torch.FloatTensor(data_input), torch.FloatTensor(data_label))
        save_obj(trainData, "input_data/%s" % data_name)
    else:
        trainData = reload_obj("input_data/%s" % data_name)
    return trainData


def get_sigmoid_auroc(test_loader, device, net):
    ground_truth = list()
    predict = list()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            try:
                [ground_truth.append(int(num)) for num in labels.squeeze().cpu().tolist()]
            except TypeError as te:
                print(te)
                ground_truth.append(int(labels.squeeze().cpu().tolist()))
            try:
                [predict.append(num.item()) for num in outputs.cpu()]
            except TypeError as te:
                print(te)
                predict.append(outputs.cpu().item())
    print('Auroc: %f' % roc_auc_score(ground_truth, predict))
    return roc_auc_score(ground_truth, predict)


def oversampling(data_set, oversampleing_ratio):
    train_features = [row[0].numpy() for row in data_set]
    train_labels = [row[1].item() for row in data_set]

    print('Original dataset shape %s' % Counter(train_labels))

    ros = SMOTE(random_state=0, sampling_strategy=oversampleing_ratio)
    re_train_feature, re_train_labels = ros.fit_resample(train_features, train_labels)
    print('Resampled dataset shape %s' % Counter(re_train_labels))

    re_train_feature = re_train_feature.tolist()
    re_train_labels = re_train_labels.reshape(len(re_train_labels), 1).tolist()
    re_train_feature = torch.FloatTensor(re_train_feature)
    re_train_labels = torch.FloatTensor(re_train_labels)
    res_train_data_set = td.TensorDataset(re_train_feature, re_train_labels)
    return res_train_data_set


def oversampling_after_split(data_set, act_type):
    # oversampling after split
    train_size = int(0.7 * len(data_set))
    test_size = len(data_set) - train_size
    train_data, test_data = td.random_split(data_set, [train_size, test_size])

    train_size_v = int(0.7 * len(train_data))
    validation_size = len(train_data) - train_size_v
    train_data_v, validation_data = td.random_split(train_data, [train_size_v, validation_size])

    valid_loader = td.DataLoader(validation_data, batch_size=50, shuffle=True, num_workers=10)

    # train_loader = td.DataLoader(data_set, batch_size=50, shuffle=True, num_workers=10)  # whole data as a trainset

    train_data = oversampling(train_data, act_type)
    train_loader = td.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=10)

    test_data = oversampling(test_data, act_type)
    test_loader = td.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=10)

    return [train_loader, test_loader, valid_loader]


def get_train_test_valid_set(data_set, test_num):
    # For test
    num_data = len(data_set)
    indices_data = list(range(num_data))
    np.random.shuffle(indices_data)
    split_tt = int(np.floor(test_num * num_data))
    train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = td.DataLoader(data_set, batch_size=50, sampler=train_sampler, num_workers=10)
    test_loader = td.DataLoader(data_set, batch_size=50, sampler=test_sampler, num_workers=10)
    various_set = [train_loader, test_loader]

    return various_set


def cross_validation_index(data_set):
    num_data = len(data_set)
    indices_data = list(range(num_data))
    cv = KFold(10, shuffle=True, random_state=10)
    result = list()
    for train_idx, test_idx in cv.split(indices_data):
        result.append([train_idx, test_idx])
    return result


def count_pos_intersection(train_loader, test_loader):
    train = list()
    test = list()
    for data, target in train_loader:
        data = data.tolist()
        target = target.tolist()
        for i in range(len(target)):
            if target[i][0] == 1:
                train.append(data[i])
    for data, target in test_loader:
        data = data.tolist()
        target = target.tolist()
        for i in range(len(target)):
            if target[i][0] == 1:
                test.append(data[i])
    print(len(train))
    print(len(test))
    count = 0
    for data in train:
        if data in test:
            count += 1
    print(count)

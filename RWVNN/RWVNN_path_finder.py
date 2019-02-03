import os
from itertools import product

import networkx as nx
import torch
import torch.nn as nn

from VNN import VNN_main as vnn, VNN_utils as uf


def build_network(file_path: str) -> nx.DiGraph:
    result = nx.DiGraph()
    relation_list = uf.read_file(file_path)

    for relation in relation_list:
        result.add_edge(relation[0], relation[1], type=relation[2])

    return result


def path_edge_attribute(path, network):
    return [(u, v, network[u][v]['type']) for (u, v)in zip(path[0:], path[1:])]


def get_target_dict(is_comb) -> dict:
    result = dict()

    if not is_comb:

        path = "../data/drug/Single_Drug_Target.txt"
        target_file = uf.read_file(path)

        for line in target_file:
            result[line[0]] = line[1].split('|')

        return result

    elif is_comb:

        path = "../data/drug/Combination_Drug_Targets_From_DCDB.txt"
        target_file = uf.read_file(path)
        target_file.remove(target_file[0])

        for line in target_file:
            result[line[0]] = line[3].split('|')

        return result


def get_rw_score_dcit(is_comb, drug_name):
    result = dict()

    if is_comb:
        dir_path = "../RW/RW_combi_DEG_prediction_results/"
    else:
        dir_path = "../RW/RW_DEG_prediction_results/"

    raw_file = uf.read_file(''.join([dir_path, drug_name, "_RW_result.txt"]))

    for line in raw_file:
        result[line[0]] = float(line[1])

    return result


def get_rw_path(network: nx.DiGraph, targets: list, degs: list) -> list:
    result = list()
    for target, deg in product(targets, degs):
        target = target + "#P"
        deg = deg + "#P"
        if network.has_node(target) and network.has_node(deg) and nx.has_path(network, target, deg):
            shorest_paths = nx.all_shortest_paths(network, source=target, target=deg)
            for shorest_path in shorest_paths:
                shorest_path = path_edge_attribute(shorest_path, network)
                if len(shorest_path) == 0:
                    shorest_path = target
                result.append(shorest_path)
    return result


def get_sigmoid_network_params(net, degs, gene_index):
    result = list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_feature = [0] * len(gene_index)
    degs = (deg for deg in degs if deg in gene_index)
    for deg in degs:
        input_feature[gene_index.index(deg)] = 1

    input_feature = torch.FloatTensor(input_feature)
    input_feature = input_feature.view(1, -1)

    output = net(input_feature.to(device))

    params = net.parameters()

    BPs = nn.Sequential(*list(net.children())[:-4])(input_feature.to(device))
    MFs = nn.Sequential(*list(net.children())[:-5])(input_feature.to(device))

    result.append(output.cpu().squeeze().item())
    result.append(MFs.cpu().squeeze().tolist())
    result.append(BPs.cpu().squeeze().tolist())

    for i, param in enumerate(params):
        if i % 2 == 0:  # i = 0:GE-MF, 2:MF-BP, 4:BP-PH
            result.append(param.cpu().tolist())

    return result



def get_vnn_path(degs, net_params, layer_info, matrix_list):
    # dict and array of entities
    mf_dict = dict()  # { mf:score from activation function of network }
    mf_array = layer_info[1]
    bp_dict = dict()  # { bp:score from activation function of network }
    bp_array = layer_info[2]

    # parameter from trained network
    mf_score = net_params[1]
    bp_score = net_params[2]
    ge_mf = net_params[3]
    mf_bp = net_params[4]
    bp_ph = net_params[5]

    # make mf_dict
    for mf in mf_array:
        mf_dict[mf] = mf_score[mf_array.index(mf)]
    for bp in bp_array:
        bp_dict[bp] = bp_score[bp_array.index(bp)]

    # score function deg, mf, bp
    result = list()
    for deg, mf, bp in list(product(degs, mf_array, bp_array)):
        if matrix_list[0][mf_array.index(mf)][degs.index(deg)] == 0 or\
            matrix_list[1][bp_array.index(bp)][mf_array.index(mf)] == 0: continue
        line = list()
        line.append(deg)

        ge_mf_weight = ge_mf[mf_array.index(mf)][degs.index(deg)]
        line.append(ge_mf_weight)
        line.append(mf)
        line.append(mf_dict[mf])

        mf_bp_weight = mf_bp[bp_array.index(bp)][mf_array.index(mf)]
        line.append(mf_bp_weight)
        line.append(bp)
        line.append(bp_dict[bp])
        line.append(bp_ph[0][bp_array.index(bp)])  # use effect node
        line.append('Breast cancer')
        score = 1 * ge_mf[mf_array.index(mf)][degs.index(deg)] + \
                mf_dict[mf] * mf_bp[bp_array.index(bp)][mf_array.index(mf)] + \
                bp_dict[bp] * bp_ph[0][bp_array.index(bp)]
        line.append(score)
        result.append(line)

    return result


def get_total_path(RW_paths, gene_RW_dict, VNN_paths):
    result = list()

    max_length_rw = max([len(rw_path) for rw_path in RW_paths])

    header = ['Target protein', 'Relation']
    for i in range(max_length_rw - 1):
        header += ['Molecular #%d' % i, 'Molecular score #%d' % i, 'Relation']
    header += ['score']
    header += ['DEG', 'Weight', 'MF', 'MF_score', 'Weight', 'BP', 'BP_score', 'Weight', 'PH', 'Score']

    result.append(header)

    for rw_path, vnn_path in product(RW_paths, VNN_paths):

        output_frame = ['']*(max_length_rw*3)

        if type(rw_path) is list and rw_path[-1][1].split("#")[0] == vnn_path[0]:

            rw_score = 0
            count = 0

            for i, edge in enumerate(rw_path):  # edge information: [node1, node2, relation type]
                if i == 0:
                    output_frame[0] = edge[0]
                    output_frame[1] = edge[2]
                    output_frame[2] = edge[1]
                    continue

                output_frame[i * 3 - 1] = edge[0]
                output_frame[i * 3] = gene_RW_dict[edge[0]]
                output_frame[i * 3 + 1] = edge[2]

                rw_score += gene_RW_dict[edge[0]]

                if i < len(rw_path) - 1:
                    output_frame[i * 3 + 3] = edge[1]

                count += 1

            if count == 0:  # if, only single edge between a target and a deg
                count = 1

            output_frame[-1] = rw_score/count

            for entity in vnn_path:
                output_frame.append(entity)

            result.append(output_frame)

        elif type(rw_path) is not list and rw_path.split("#")[0] == vnn_path[0]:

            output_frame[0] = rw_path

            for entity in vnn_path:
                output_frame.append(entity)

            result.append(output_frame)
    return result


def framework(is_comb, drug_name, targets, degs, net):
    # get RW result
    network_path = '../data/network/RW/GP_GP_Four_Relation_Types.txt'
    network = build_network(network_path)
    rw_paths = get_rw_path(network, targets, degs)
    gene_RW_dict = get_rw_score_dcit(is_comb, drug_name)

    #get vnn result
    relation_files = ['GE_MF', 'MF_BP', 'BP_PH']
    layer_info = uf.set_layer_info(relation_files)
    matrix_list = uf.set_mask_matrics(relation_files, layer_info)
    gene_index = layer_info[0]
    params = get_sigmoid_network_params(net, degs, gene_index)  # [pred_score, BP, MF, net_paramters]
    vnn_paths = get_vnn_path(degs, params, layer_info, matrix_list)

    #write result
    result = get_total_path(rw_paths, gene_RW_dict, vnn_paths)
    save_path = 'path_result/%s_path.tsv' % drug_name
    uf.write_tsv(result, save_path)


def get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio, *drug_name):
    net = vnn.get_trained_network(n_epochs, oversampling_ratio)
    if is_comb:
        dir_path = "../RW/RW_combi_DEG_prediction_results"
    else:
        dir_path = "../RW/RW_DEG_prediction_results"

    filenames = os.listdir(dir_path)
    drug_target_dict = get_target_dict(is_comb)
    if not single_data:
        # for all drug
        for i, filename in enumerate(filenames):
            if not filename.endswith("_RW_result.txt"):
                drug_name = filename.split('.')[0]
            else:
                continue

            if drug_name not in drug_target_dict.keys():
                continue
            full_name = os.path.join(dir_path, filename)
            degs = list()
            for deg in uf.read_file(full_name):
                degs += deg

            framework(is_comb, drug_name, drug_target_dict[drug_name], degs, net)

    # for single drug
    elif single_data:
        degs = list()
        for deg in uf.read_file(os.path.join(dir_path, drug_name + ".txt")):
            degs += deg
        if drug_name not in drug_target_dict.keys():
            print('test fail')
            return 0
        framework(is_comb, drug_name, drug_target_dict[drug_name], degs, net)


if __name__ == '__main__':

    n_epochs = 30    # # of epochs of network
    oversampling_ratio = 0.06   # oversampling ratio of network (0 ,1]
    is_comb = False  # drug type: combination drug(True), single drug(False)
    single_data = False  # find path for single data(True) or multiple data(Fasle)

    get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio)

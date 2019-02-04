import os
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

WEIGHT_R_OPTIONS = [0.2, 1.0]
WEIGHT_E_OPTIONS = [0.2, 1.0]
WEIGHT_S_OPTIONS = [0.2, 1.0]
WEIGHT_U_OPTIONS = [0.2, 1.0]

TRAIN_DATA_RATIO = 0.8


def insert_into_dictionary(dict, key, value):
    if key in dict.keys():
        if value not in dict[key]:
            tmp = dict[key]
            tmp.append(value)
            dict[key] = tmp
    else:
        dict[key] = [value]


def make_network(network_file_path, weight_dict, output_file_path):
    network_file = open(network_file_path, 'r')

    index = 0
    node_dict = {}
    network_dict = {}
    while True:
        line = network_file.readline()
        if not line:
            break
        lt, rt, edge_type = line.strip().split("\t")

        if lt not in node_dict.keys():
            node_dict[lt] = index
            index += 1
        if rt not in node_dict.keys():
            node_dict[rt] = index
            index += 1

        if edge_type == 'Regulate':
            insert_into_dictionary(network_dict, node_dict[lt], str(node_dict[rt]) + "|R")
        elif edge_type == 'Express':
            insert_into_dictionary(network_dict, node_dict[lt], str(node_dict[rt]) + "|E")
        elif edge_type == 'Undirected Signaling':
            insert_into_dictionary(network_dict, node_dict[lt], str(node_dict[rt]) + "|U")
        elif edge_type == 'Directed Signaling':
            insert_into_dictionary(network_dict, node_dict[lt], str(node_dict[rt]) + "|S")

    output_file = open(output_file_path, 'w+')
    for entity in network_dict.keys():
        neighbors = network_dict[entity]
        tmp_total = 0.0
        for neighbor in neighbors:
            tmp_total += weight_dict[neighbor.split("|")[1]]

        for neighbor in neighbors:
            output_file.write(str(entity) + '\t' + neighbor.split("|")[0] + '\t' + str(weight_dict[neighbor.split("|")[1]] / tmp_total) + '\n')
    output_file.close()

    network_dict_file_path = output_file_path.split(".txt")[0] + '_dict.txt'
    network_dict_file = open(network_dict_file_path, 'w+')
    for node in node_dict.keys():
        network_dict_file.write(node + '\t' + str(node_dict[node]) + '\n')
    network_dict_file.close()


def make_weighted_network(weight_r_list, weight_e_list, weight_s_list, weight_u_list):
    index = 1
    weight_dict = {}
    if 'randomwalk_weight_training' not in os.listdir('./'):
        os.mkdir('randomwalk_weight_training', 511)
    for weight_r in weight_r_list:
        for weight_e in weight_e_list:
            for weight_s in weight_s_list:
                for weight_u in weight_u_list:
                    weight_dict = {'R': weight_r, 'E': 1.0, 'U': weight_s, 'S': weight_s}
                    if '%d' % index not in os.listdir('randomwalk_weight_training'):
                        os.mkdir('randomwalk_weight_training/%d' % index, 511)
                    else:
                        print('Already existing weight combination! %d' % index)
                        continue
                    make_network('../data/network/RW/original_network.txt', weight_dict, 'randomwalk_weight_training/%d/%d.txt' % (index, index))
                    index_file = open("randomwalk_weight_training/%d/%d_R:%f_E:%f_S:%f_U:%f.txt" % (index, index, weight_dict['R'], weight_dict['E'], weight_dict['S'], weight_dict['U']), 'w+')
                    index_file.close()
                    index += 1
                    print("%d(R:%f_E:%f_S:%f_U:%f) network is constructed!" % (index, weight_dict['R'], weight_dict['E'], weight_dict['S'], weight_dict['U']))


def split_train_test_drug_sets(train_ratio, drug_file_path):
    drug_file = open(drug_file_path, 'r')
    cnt = 1
    drug_dict = {}
    test_drug_dict = {}
    train_drug_dict = {}

    while True:
        line = drug_file.readline()
        if not line:
            break
        drugName, drugTargets, drugDEGs = line.strip().split("\t")
        drug_dict[drugName] = drugTargets + "__" + drugDEGs
        cnt += 1

    drug_file.close()

    for drug in drug_dict.keys():
        if len(train_drug_dict.keys()) > train_ratio * cnt:
            test_drug_dict[drug] = drug_dict[drug]
        elif len(test_drug_dict.keys()) > (1 - train_ratio) * cnt:
            train_drug_dict[drug] = drug_dict[drug]
        else:
            rand_int = random.randint(0, 100)
            if rand_int <= train_ratio * 100:
                train_drug_dict[drug] = drug_dict[drug]
            else:
                test_drug_dict[drug] = drug_dict[drug]
    print("# of Training drug samples: %d\n# of Validation drug samples: %d" % (len(train_drug_dict.keys()), len(test_drug_dict.keys())))

    return train_drug_dict, test_drug_dict


def get_optimal_threshold(label, score):
    optimal_th_dict = {}
    fpr, tpr, threshold = roc_curve(label, score)

    for i in range(len(fpr)):
        optimal_th_dict[threshold[i]] = abs(tpr[i] - (1 - fpr[i]))
    sorted_by_value = sorted(optimal_th_dict.items(), key=lambda kv: kv[1])
    optimal_threshold = sorted_by_value[0][0]

    return optimal_threshold


def randomwalk_network_training(train_drugs):
    train_auroc_dict = {}
    network_index = 1
    for network_folder in os.listdir('randomwalk_weight_training'):
        print("TOTAL NETOWRKS:", os.listdir('randomwalk_weight_training'))
        print("** %s network is being trained..." % network_folder)

        total_auroc = 0.0
        total_threshold = 0.0
        drug_index = 0
        if '%s_dict.txt' % network_folder in os.listdir('randomwalk_weight_training/%s' % network_folder):
            network_file = 'randomwalk_weight_training/%s/%s.txt' % (network_folder, network_folder)
            network_dict = 'randomwalk_weight_training/%s/%s_dict.txt' % (network_folder, network_folder)
            for drug in train_drugs.keys():
                target_list = train_drugs[drug].split("__")[0].split("|")
                DEG_list = train_drugs[drug].split("__")[1].split("|")

                if len(target_list) == 0:
                    print('-- %s drug has no targets')
                    continue
                else:
                    drug_index += 1
                    target_file_name = 'randomwalk_weight_training/%s/%s.txt' % (network_folder, drug)
                    target_file = open(target_file_name, 'w+')
                    for target in target_list:
                        target_file.write(drug + '\t' + target + '\n')
                    target_file.close()
                    
                    rw_result_file = 'randomwalk_weight_training/%s/%s_RW_results.txt' % (network_folder, drug)
                    randomwalk(network_file, network_dict, target_file_name, rw_result_file)
                    
                    os.remove(target_file_name)
                    
                    label_list = []
                    score_list = []
                    
                    rw_result = open(rw_result_file, 'r')
                    while True:
                        line = rw_result.readline()
                        if not line:
                            break
                        entity, rw_score = line.strip().split("\t")
                        if entity.split("#")[1] == 'P':
                            if entity in DEG_list:
                                label_list.append(1)
                                score_list.append(float(rw_score))
                            else:
                                label_list.append(0)
                                score_list.append(float(rw_score))
                    
                    rw_result.close()
                    os.remove(rw_result_file)
                                        
                    auroc_score = roc_auc_score(label_list, score_list)
                    optimal_threshold = get_optimal_threshold(label_list, score_list)
                    print('\n## Network: %s, drug(%d): %s' % (network_folder, drug_index, drug))
                    print('  AUROC:%f optimal threshold:%f' % (auroc_score, optimal_threshold))

                    # For average
                    total_auroc += auroc_score
                    total_threshold += optimal_threshold
                    print('  Average AUROC:%f' % (total_auroc / drug_index))
                    print('  Average optimal threshold:%f' % (total_threshold / drug_index))

        train_auroc_dict[network_folder] = str(total_auroc / drug_index ) + "|" + str(total_threshold / drug_index)
        network_index += 1

    print('** TRAINING IS OVER!! results below (network, auroc, optimal threshold)')
    training_result_file = open('randomwalk_weight_training/training_result.txt', 'w+')
    for net in train_auroc_dict.keys():
        print(net + '\t' + str(train_auroc_dict[net].split("|")[0]) + '\t' + str(train_auroc_dict[net].split("|")[1]))
        training_result_file.write(net + '\t' + str(train_auroc_dict[net].split("|")[0]) + '\t' + str(train_auroc_dict[net].split("|")[1]) + '\n')

    # Network and optimal threshold with HIGHEST performance
    highest_auroc = 0.0
    highest_network = ''
    highest_threshold = 0.0
    for network in train_auroc_dict.keys():
        auroc, optimal_ths = train_auroc_dict[network].split("|")
        if float(auroc) > highest_auroc:
            highest_network = str(network)
            highest_auroc = float(auroc)
            highest_threshold = float(optimal_ths)

    return highest_network, highest_threshold


def validation_randomwalk_network_training(opt_network, validation_drugs):
    if '%s_dict.txt' % opt_network in os.listdir('randomwalk_weight_training/%s' % opt_network):
        network_file = 'randomwalk_weight_training/%s/%s.txt' % (opt_network, opt_network)
        network_dict = 'randomwalk_weight_training/%s/%s_dict.txt' % (opt_network, opt_network)
        total_auroc = 0.0
        total_threshold = 0.0
        drug_index = 1
        
        for drug in validation_drugs.keys():
            target_list = validation_drugs[drug].split("__")[0].split("|")
            DEG_list = validation_drugs[drug].split("__")[1].split("|")

            if len(target_list) == 0:
                print('-- %s drug has no targets')
                continue
            else:
                target_file_name = 'randomwalk_weight_training/%s/%s.txt' % (opt_network, drug)
                target_file = open(target_file_name, 'w+')
                for target in target_list:
                    target_file.write(drug + '\t' + target + '\n')
                target_file.close()

                rw_result_file = 'randomwalk_weight_training/%s/%s_rw_results.txt' % (opt_network, drug)
                randomwalk(network_file, network_dict, target_file_name, rw_result_file)

                os.remove(target_file_name)

                label_list = []
                score_list = []

                rw_result = open(rw_result_file, 'r')
                while True:
                    line = rw_result.readline()
                    if not line:
                        break
                    entity, rw_score = line.strip().split("\t")
                    if entity.split("#")[1] == 'P':
                        if entity in DEG_list:
                            label_list.append(1)
                            score_list.append(float(rw_score))
                        else:
                            label_list.append(0)
                            score_list.append(float(rw_score))

                rw_result.close()
                os.remove(rw_result_file)

                auroc_score = roc_auc_score(label_list, score_list)
                optimal_threshold = get_optimal_threshold(label_list, score_list)
                print('\n## Network:%s, drug:%s' % (opt_network, drug))
                print('  AUROC:%f optimal threshold:%f' % (auroc_score, optimal_threshold))
                total_auroc += auroc_score
                total_threshold += optimal_threshold
                print('  Average AUROC:%f' % (total_auroc / drug_index))
                print('  Average optimal threshold:%f' % (total_threshold / drug_index))

    
def randomwalk(network_file, network_dict_file, drug_targets_file, output_file):
    os.system("Rscript RandomWalk.R " + network_file + " " + network_dict_file + " " + drug_targets_file + " " + output_file)


if __name__ == '__main__':
    make_weighted_network(WEIGHT_R_OPTIONS, WEIGHT_E_OPTIONS, WEIGHT_S_OPTIONS, WEIGHT_U_OPTIONS)
    
    train_drug, validation_drug = split_train_test_drug_sets(TRAIN_DATA_RATIO, '../data/drug/drug_targets_DEGs.txt')
    train_auroc_result = randomwalk_network_training(train_drug)
    validation_randomwalk_network_training(train_auroc_result, validation_drug)

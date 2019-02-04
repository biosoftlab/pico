import os
import sys

RANDOM_WALK_SCORE_CUT_OFF = 0.009862
NETWORK_FILE_PATH = 'randomwalk_weight_training/1/1.txt'
NETWORK_DICT_FILE_PATH = 'randomwalk_weight_training/1/1_dict.txt'


def deg_prediction():
    if_combi = sys.argv[1]
    drug_name_list = sys.argv[2:]
    drug_names = '|'.join(drug_name_list)

    # Make Drug Targets List
    drug_dict_file = open('../data/drug/drug_targets_DEGs.txt', 'r')
    drug_target_file_path = '%s_targets.txt' % drug_names
    drug_target_file = open(drug_target_file_path, 'w+')

    target_list = []
    while True:
        line = drug_dict_file.readline()
        if not line:
            break
        name, targets, DEGs = line.strip().split('\t')
        if name in drug_name_list:
            for target in targets.split('|'):
                if target not in target_list:
                    drug_target_file.write(drug_names + '\t' + target + '\n')
                    target_list.append(target)
    drug_dict_file.close()
    drug_target_file.close()

    if if_combi == 'combination':
        output_file_path = 'RW_combi_DEG_prediction_results/%s_RW_result.txt' % drug_names.lower()
        deg_prediction_file = 'RW_combi_DEG_prediction_results/%s.txt' % drug_names.lower()
    elif if_combi == 'single':
        output_file_path = 'RW_DEG_prediction_results/%s_RW_result.txt' % drug_names.lower()
        deg_prediction_file = 'RW_DEG_prediction_results/%s.txt' % drug_names.lower()
    else:
        print('Wrong input arguments!')
        sys.exit(-1)

    # Run RW
    os.system(
        "Rscript RandomWalk.R " + NETWORK_FILE_PATH + " " + NETWORK_DICT_FILE_PATH + " " + drug_target_file_path + " " + output_file_path)
    os.remove(drug_target_file_path)

    # Make DEG List
    rw_result_file = open(output_file_path, 'r')
    deg_prediction = open(deg_prediction_file, 'w+')

    while True:
        line = rw_result_file.readline()
        if not line:
            break
        entity, rw_score = line.strip().split("\t")
        if float(rw_score) >= RANDOM_WALK_SCORE_CUT_OFF:
            if entity.split("#")[1] == 'P':
                deg_prediction.write(entity.split("#")[0] + '\n')

    rw_result_file.close()
    deg_prediction.close()

if __name__ == '__main__':
    deg_prediction()

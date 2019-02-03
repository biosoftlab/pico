import os

from VNN import VNN_main as vnn, VNN_utils as uf
from RWVNN import RWVNN_path_finder as rpf


def get_vnn_prediction(degs, net):
    # get vnn prediction
    result = list()
    relation_files = ['GE_MF', 'MF_BP', 'BP_PH']
    gene_index = uf.set_layer_info(relation_files)[0]

    params = rpf.get_sigmoid_network_params(net, degs, gene_index)  # [pred_score, BP, MF, net_paramters]
    result.append(params[0])  # prediction score
    return result


def get_disease_comb_drug():
    path = "../data/drug/Combination_Drug_Targets_From_DCDB.txt"
    combi_file = uf.read_file(path)
    result = list()
    for combi in combi_file:
        if combi[1] == "P":
            result.append(combi[0])
    return result


def get_vnn_score(dirname, output_path, is_comb, net):
    if is_comb:
        disease_drug = get_disease_comb_drug()
    else:
        disease_drug = uf.get_diseas_drug()

    filenames = os.listdir(dirname)

    result = list()
    result.append(['drug_name', 'label', 'prediction'])

    for i, filename in enumerate(filenames):

        if not filename.endswith("_RW_result.txt"):
            drug_name = filename.split('.')[0]
        else:
            continue

        full_name = os.path.join(dirname, filename)

        degs = list()
        for deg in uf.read_file(full_name):
            degs += deg

        line = list()
        line.append(drug_name)

        if not is_comb:
            drug_name = drug_name.lower()

        if drug_name in disease_drug:
            line.append(1)
        else:
            line.append(0)

        line += get_vnn_prediction(degs, net)

        result.append(line)

    uf.write_tsv(result, output_path)


def get_drug_score(is_comb, n_epochs, oversampling_ratio):
    if is_comb:
        comb = "combi"
        dir_path = "../RW/RW_combi_DEG_prediction_result"
    else:
        comb = "single"
        dir_path = "../RW/RW_DEG_prediction_result"

    output = "score_result/%s_VNN_%d_epoch_%.2f.tsv" % (comb, n_epochs, oversampling_ratio)

    net = vnn.get_trained_network(n_epochs, oversampling_ratio)
    get_vnn_score(dir_path, output, is_comb, net)


if __name__ == '__main__':
    is_comb = True
    n_epochs = 30
    oversampling_ratio = 0.06
    get_drug_score(is_comb, n_epochs, oversampling_ratio)

from RW import RW_main as rwm
from RW import RW_training as rwt
from VNN import VNN_main as vm
from RWVNN import RWVNN_path_finder as rpf
from RWVNN import RWVNN_score as rs

if __name__ == '__main__':
    # RW
    """
    RW_training.py
    
    Train weight of network for better prediction using random-walk
    """

    # training weight options
    rwt.WEIGHT_R_OPTIONS = [0.2, 1.0]
    rwt.WEIGHT_E_OPTIONS = [0.2, 1.0]
    rwt.WEIGHT_S_OPTIONS = [0.2, 1.0]
    rwt.WEIGHT_U_OPTIONS = [0.2, 1.0]

    rwt.make_weighted_network(rwt.WEIGHT_R_OPTIONS, rwt.WEIGHT_E_OPTIONS, rwt.WEIGHT_S_OPTIONS, rwt.WEIGHT_U_OPTIONS)

    # split train/validation drug set
    rwt.TRAIN_DATA_RATIO = 0.8  # test:0.8  validation:0.2

    # train and validate
    DRUG_FILE = '../data/drug/drug_targets_DEGs.txt'
    train_drug, validation_drug = rwt.split_train_test_drug_sets(rwt.TRAIN_DATA_RATIO, DRUG_FILE)
    opt_network, opt_threshold = rwt.randomwalk_network_training(train_drug)
    rwt.validation_randomwalk_network_training(opt_network, validation_drug)


    """
    RW_main.py
    
    Get predicted DEGs from drug or drug combination
    """
    opt_threshold = 0.009  # If you start with training, please remove this line

    # run with trained results
    rwm.RANDOM_WALK_SCORE_CUT_OFF = opt_threshold
    rwm.NETWORK_FILE_PATH = 'randomwalk_weight_training/%d/%d.txt' % (opt_network, opt_network)
    rwm.NETWORK_DICT_FILE_PATH = 'randomwalk_weight_training/%d/%d_dict.txt' % (opt_network, opt_network)

    COMBI_OR_SINGLE = 'single'  # 'single' or 'combination'
    DRUG_NAME_LIST = ['aminoglutethimide']
    rwm.deg_prediction(COMBI_OR_SINGLE, DRUG_NAME_LIST)



    # VNN

    """
    VNN_main.py

    Get VNN auroc score from DEGs and effectiveness of drug
    """

    # input example - no cross-validation, 30 epochs, 0.06 oversampling_ratio
    cross_validation = False
    n_epochs = 30
    oversampling_ratio = 0.06

    # get auroc score from VNN model and input drug DEGs
    # no cross-validation
    auroc_result = list()  # auroc_result = [n_epochs, auroc score]
    auroc_result = vm.get_auroc(n_epochs, cross_validation, oversampling_ratio)

    # cross-validation
    cross_validation = True
    auroc_result = list()  # auroc_result = [[n_epochs, auroc score] <- 1 fold, [n_epochs, auroc score] <- 2fold, ...]
    auroc_result = vm.get_auroc(n_epochs, cross_validation, oversampling_ratio)

    """
    RWVNN_score.py

    Get VNN score of drugs from predicted DEGs from RW in .tsv file format under score_result folder
    """

    # input example - single drug, 30 epochs, 0.06 oversampling_ratio
    n_epochs = 30
    oversampling_ratio = 0.06
    is_comb = False

    # get prediction score for drugs of RWVNN from
    rs.get_drug_score(is_comb, n_epochs, oversampling_ratio)

    """
    RWVNN_path_finder.py

    Get total path with RW score and VNN path score in .tsv file format under path_result folder
    """

    # input example - single drug, 30 epochs, 0.06 oversampling_ratio, all data under RW prediction result folder
    n_epochs = 30
    oversampling_ratio = 0.06
    is_comb = False
    single_data = False
    rpf.get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio)

    # single data example with drug name under RW prediction result folder
    single_data = True
    drug_name = "lapatinib"
    rpf.get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio, "lapatinib")

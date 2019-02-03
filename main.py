from VNN import VNN_main as vm
from RWVNN import RWVNN_path_finder as rpf
from RWVNN import RWVNN_score as rs

if __name__ == '__main__':


    #VNN

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

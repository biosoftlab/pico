# pico

PICO predicts drug effects on a target disease and it interprets underlying paths between drug target proteins and the target disease.
This framework consists of two parts;
  1) RW: random walk based simulation of drug influence on molecules 
  2) VNN: visible neural network based prediction of drug influence on the target disease through cellular functions

Environment
virtual environment: conda 4.5.11
python version: 3.6.8
pytorch version: 1.0.0


Data setup for RW

It recommends every drug name in lowercase.
1. Drug-target-DEGs relations:
    Fill the file 'data/drug/drug_targets_DEGs.txt' with following format of drug-targets-DEG.
    Input data format: drug1    target1|target2 DEG1|DEG2|DEG3
                       ...

2. Molecular relations:
    Fill the file 'data/network/RW/original_network.txt' with following format of subject-object-predicate.
    There are two types of nodes for subject and object: #P (=protein), #D (=DNA)
    There are four types of predicates:Regulate(R), Express(E), Directed Signaling(S), and Undirected Signaling(U).
    Input data format: EGF#P	EGFR#P	Directed Signaling
                       ...


Code in RW:
1. RW_training.py:
        Train weights for three edge types (i.e. R, E, S/U). It generates weighted networks in 'randomwalk_weight_training' folder.
        Funiction:
            make_weighted_network(rwt.WEIGHT_R_OPTIONS, rwt.WEIGHT_E_OPTIONS, rwt.WEIGHT_S_OPTIONS, rwt.WEIGHT_U_OPTIONS)
        
        # training weight options
            rwt.WEIGHT_R_OPTIONS = [0.2, 1.0]
            rwt.WEIGHT_E_OPTIONS = [0.2, 1.0]
            rwt.WEIGHT_S_OPTIONS = [0.2, 1.0]
            rwt.WEIGHT_U_OPTIONS = [0.2, 1.0]
            rwt.make_weighted_network(rwt.WEIGHT_R_OPTIONS, rwt.WEIGHT_E_OPTIONS, rwt.WEIGHT_S_OPTIONS, rwt.WEIGHT_U_OPTIONS)
2. RW_main.py:
        From the optimal network and cut-off threshold from RW_training.py, it predicts DEGs from input drug name.
        
        # run with trained results
            rwm.RANDOM_WALK_SCORE_CUT_OFF = opt_threshold
            rwm.NETWORK_FILE_PATH = 'randomwalk_weight_training/%d/%d.txt' % (opt_network, opt_network)
            rwm.NETWORK_DICT_FILE_PATH = 'randomwalk_weight_training/%d/%d_dict.txt' % (opt_network, opt_network)
            
            COMBI_OR_SINGLE = 'single'  # 'single' or 'combination'
            DRUG_NAME_LIST = ['aminoglutethimide']
            rwm.deg_prediction(COMBI_OR_SINGLE, DRUG_NAME_LIST)


Data setup for VNN

It recommends every drug name in lowercase.
Every input files are wrote in tsv format.

1. Drug-DEG relations:
    Fill the file 'data/drug/target_phenotype/Drug_DEGs.txt' with following format of drug-DEG pair
    Input data format: drug1 DEG1|DEG2|DEG3
                       ...

2. Effective drugs for the target phenotype(=disease):
    Fill the file 'data/drug/target_phenotype/target_phenotype_drug_list.txt' with follwing format of drugs
    Input data format: drug1
                       drug2
                       drug3
                       ...

3. Relations of gene(GE), molecular function(MF), biological process(BP), and phenotype(PH):
    Fill the files under 'data/network/VNN'
    Files: GE_MF.txt, MF_BP.txt, BP_PH.txt

    Fill the files with follwing format example
    Input data format(GE_MF): gene1 'molecular function1'
                              gene1 'molecular function2'
                              ...


Code in VNN:
1. VNN_utils.py:
        It contains util functions, such as file read and write, for VNN.py and python files in RWVNN
2. CustomizedLinear.py:
        Make custom connection in the neural network.
        It is the modified version of https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
3. VNN_main:
        Get VNN auroc score.
        Function:
            get_auroc(n_epochs, cross_validation, oversampling_ratio)

        # Example - no cross-validation, 30 epochs, 0.06 oversampling_ratio
            cross_validation = False
            n_epochs = 30
            oversampling_ratio = 0.06
            auroc_result = get_auroc(n_epochs, cross_validation, oversampling_ratio)
            # auroc_result format: [n_epochs, auroc score]

        # Example - cross-validation, 30 epochs, 0.06 oversampling_ratio
            cross_validation = True
            n_epochs = 30
            oversampling_ratio = 0.06
            auroc_result = get_auroc(n_epochs, cross_validation, oversampling_ratio)
            # auroc_result format: [[n_epochs, auroc score] <- 1 fold, [n_epochs, auroc score] <- 2fold, ...]


Data setup for RWVNN(PICO)

It recommends every drug name in lower case.
Every input file is wrote in tsv format.

1. Drug_target information:
    It needs to fill the two files. Drug_Target.txt, Combination_Drug_Targets_From_DCDB.txt
        -Drug_Target.txt
            Fill the files with following format example
            Input data format: drug1 Target1|Target2|Target3
                               ...
        -Combination_Drug_Targets_From_DCDB.txt
            Fill the files with follwing format example
            This file needs header as below
               --> ID: DCDBID, label: P(=positive) or N(=nagative), name: names of drugs, targets: gene symbols of target proteins
            Input data format: ID   label   name    targets (header)
                               DCID1 P drug1|drug2 Target1|Target2|Target3
                               ...

2. Target phentype effective drug list:
    Same as VNN

3. Drug DEGs and RW score of all nodes in the network:
    Result from RW


Code in RWVNN

1.RWVNN_path_finder.py:
        Get total paths and their scores calculated with RW scores and VNN path scores (in .tsv file format under path_result folder)

        Function:
            get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio, *drug_name)

        # Example - single drug, 30 epochs, 0.06 oversampling_ratio, all data under RW prediction result folder
            n_epochs = 30
            oversampling_ratio = 0.06
            is_comb = False
            single_data = False
            get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio)

        # Example - single drug, 30 epochs, 0.06 oversampling_ratio, single data example with drug name under RW prediction result folder
            n_epochs = 30
            oversampling_ratio = 0.06
            is_comb = False
            single_data = True
            drug_name = "lapatinib"
            get_RWVNN_path(single_data, is_comb, n_epochs, oversampling_ratio, "lapatinib")

2.RWVNN_score.py:
        Get VNN scores for drugs using their predicted DEGs from RW (in .tsv file format under score_result folder)

        Function:
            get_drug_score(is_comb, n_epochs, oversampling_ratio)

        # Example - single drug, 30 epochs, 0.06 oversampling_ratio
            n_epochs = 30
            oversampling_ratio = 0.06
            is_comb = False
            get_drug_score(is_comb, n_epochs, oversampling_ratio)


Main.py control all functions, change inputs in Main.py will adopted to each function
It contains example of how to use functions in python files

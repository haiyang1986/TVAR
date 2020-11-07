cd ./input/
#python build_train_set3_pos_label.py -q 0.01 -s 0.7 -t 1500
#python build_train_set3_neg_label.py
cd ..
python rare_build_sets.py -i Heart_Left_Ventricle
python rare_build_sets.py -i Pancreas
python rare_build_sets.py -i Brain_Anterior_cingulate_cortex_BA24
python rare_build_sets.py -i Breast_Mammary_Tissue

#python TVar_gpu.py -m cv
#python TVar_cpu.py -m cv
#python gwas_build_sets.py -i Heart_Left_Ventricle
#python gwas_build_sets.py -i Pancreas
#python gwas_build_sets.py -i Brain_Anterior_cingulate_cortex_BA24
#python gwas_build_sets.py -i Breast_Mammary_Tissue
#python rare_build_sets.py -i Heart_Left_Ventricle
#python rare_build_sets.py -i Pancreas
#python rare_build_sets.py -i Brain_Anterior_cingulate_cortex_BA24
#python rare_build_sets.py -i Breast_Mammary_Tissue
python run_eval_all.py

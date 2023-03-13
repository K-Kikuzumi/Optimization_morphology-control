# ICM
python ./train.py -c ewalker_free_icm.json -o log/thesis_free_icm_1
# (ICM+)MDM
python ./train.py -c ewalker_free_icm+mdm.json -o log/thesis_free_icm+mdm_1
# MDM
python ./train.py -c ewalker_free_mdm.json -o log/thesis_free_mdm_1
# (MDM+)ICM
python ./train.py -c ewalker_free_mdm+icm.json -o log/thesis_free_mdm+icm_1

# ICM
python ./train.py -c ewalker_nofail_icm.json -o log/thesis_nofail_icm_1
# (ICM+)MDM
python ./train.py -c ewalker_nofail_icm+mdm.json -o log/thesis_nofail_icm+mdm_1
# MDM
python ./train.py -c ewalker_nofail_mdm.json -o log/thesis_nofail_mdm_1
# (MDM+)ICM
python ./train.py -c ewalker_nofail_mdm+icm.json -o log/thesis_nofail_mdm+icm_1

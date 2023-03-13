# ICM
python ./train.py -c ewalker_lock_icm.json -o log/thesis_lock_icm_1
# (ICM+)MDM
python ./train.py -c ewalker_lock_icm+mdm.json -o log/thesis_lock_icm+mdm_1
# MDM
python ./train.py -c ewalker_lock_mdm.json -o log/thesis_lock_mdm_1
# (MDM+)ICM
python ./train.py -c ewalker_lock_mdm+icm.json -o log/thesis_lock_mdm+icm_1

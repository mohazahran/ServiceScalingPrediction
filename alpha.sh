#!/bin/bash

mkdir -p log
python mm1k_all_cond_all_in.py  ${1} | tee log/mm1k_all_cond_all_${1}.log
python mmmmr_all_cond_all_in.py ${1} | tee log/mmmmr_all_cond_all_${1}.log
python lbwb_all_cond_all_in.py  ${1} | tee log/lbwb_all_cond_all_${1}.log
python cio_all_cond_all_in.py   ${1} | tee log/cio_all_cond_all_${1}.log
#!/usr/bin/env bash

KAGGLE_DATA_URL="btw78jt/additionaldataforfrenchmotorclaims"
PROJ_RAW_DATA_FOLDER="data/external/"
PROJ_DATA_LOG="logs/data.log"
NOW=$(date)

kaggle datasets download -d $KAGGLE_DATA_URL -p $PROJ_RAW_DATA_FOLDER --unzip
echo $NOW ': Fetched data from:' $KAGGLE_DATA_URL 'to: ' $PROJ_RAW_DATA_FOLDER >> $PROJ_DATA_LOG && \
echo "Data download complete"

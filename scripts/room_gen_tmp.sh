#! /bin/bash

python generate_museums.py
python generate_museums.py --fields author,type_semart --combined
python generate_museums.py --fields author,type_wikiart --combined
python generate_museums.py --fields style,type_semart --combined
python generate_museums.py --fields style,type_wikiart --combined
python generate_museums.py --fields timeframe,type_semart --combined
python generate_museums.py --fields timeframe,type_wikiart --combined
import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import numpy as np
from dask import delayed, compute
import time, os, sys, json, argparse, warnings

warnings.filterwarnings("ignore")


def read_csv(folder, file):

    print(folder + file)
    fecha, predio, seccion = file[:6], file[7:12], file[13:17]
    df = pd.read_csv(folder + file)
    cols = df.columns.tolist()
    df['FECHA'] = fecha
    df['PREDIO'] = predio
    df['SECCION'] = seccion
    ncols = ['FECHA', 'PREDIO', 'SECCION'] + cols

    return df[ncols]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Terrestrial LiDAR Sensor Forestry Processing Method')
    parser.add_argument('input_path', type=str, help="Input dir for shape files")
    parser.add_argument('prefix', type=str, help="Prefix for the output files")
    args = parser.parse_args()

    input_folder = args.input_path
    prefix = args.prefix

    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    frames = []

    for file in files:
    	df = read_csv(input_folder, file)
    	frames.append(df)

    frames = compute(*frames)

    result = pd.concat(frames)
    result.to_csv(input_folder + prefix + '.csv', sep=',',index=False)
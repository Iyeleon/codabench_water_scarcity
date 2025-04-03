import os
import tqdm
from utils import load_config

import pandas as pd
import geopandas as gpd
from src.utils.data_loader import (
    load_station_info,
    load_hydro_data,
    read_soil_data,
)   

def get_stations_list(area, data_dir, crs = 'epsg:4326'):
    # get train and eval stations
    stations_train = load_station_info(area, 'train', data_dir, crs = crs)
    stations_eval = load_station_info(area, 'eval', data_dir, crs = crs)

    # eval has more stations than train. label stations present in eval only or both
    # common stations appear in both data
    common_stations = set(stations_train.station_code).intersection(stations_eval.station_code)
    stations_train['eval_only'] = False
    stations_eval['eval_only'] = stations_eval.station_code.apply(lambda x: False if x in common_stations else True)
    
    return stations_eval
    
def get_hydro_data(area, data_dir, mini_data_dir = None, crs = 'epsg:4326'):
    # load the data
    hydro_data = load_hydro_data(area, data_dir)
    
    # load stations list
    if mini_data_dir is not None:
        stations = get_stations_list(area, mini_data_dir, crs = crs)
    else:
        stations = get_stations_list(area, data_dir, crs = crs)

    # reproject hydro data to same crs as stations
    for geo_scale in hydro_data.keys():
        hydro_data[geo_scale] = hydro_data[geo_scale].to_crs(crs)
        
    # filter the hydro data
    # 1 - get join info
    join_info = get_join_info(area)

    # loop through and update filter
    stations_ = stations.copy()
    for level, (col, lsuffix, rsuffix) in join_info.items():
        stations_ = gpd.sjoin(
            stations_,
            hydro_data[level][['geometry', col]],
            how = "left",
            predicate = "within",
            lsuffix = lsuffix,
            rsuffix = rsuffix
        ).rename(columns = {col:f'{col}{rsuffix}', f'index_{rsuffix}': f'id{rsuffix}'})

    for geo_scale in hydro_data.keys():
        hydro_data[geo_scale] = hydro_data[geo_scale].iloc[list(stations_[f'id_{geo_scale}'].unique())]
    
    # return filtered hydro data and stations list
    return stations_, hydro_data

def get_join_info(area):
    if area == "france":
            join_info = {
                "region":    ("CdRegionHydro",      "_stations",      "_region"),
                "sector":    ("CdSecteurHydro",     "_stations",      "_sector"),
                "sub_sector":("CdSousSecteurHydro", "_stations",      "_sub_sector"),
                "zone":      ("CdZoneHydro",        "_stations",      "_zone")
            }
    elif area == "brazil":
        join_info = {
            "region":    ("wts_pk", "_stations_region", "_region"),
            "sector":    ("wts_pk", "_stations_sector", "_sector"),
            "sub_sector":("wts_pk", "_stations_sub_sector", "_sub_sector"),
            "zone":      ("wts_pk", "_stations_zone", "_zone")
        }
    else:
        raise ValueError('area must be one of brazil or france')
    return join_info
    
def aggregate_meteo_data():
    pass

def aggregate_soil_data(area, data_dir, stations_gdf, buffer_scales = None):
    """ aggregates the data at specified buffer scale. 
    If buffer scale is not provided, it aggregates to local, field and watershed scales
    """
    stations_gdf = stations_gdf.copy()
    
    # load soil data
    soil_data = read_soil_data(area, data_dir)

    # rename data vars to remove area from variable names
    soil_data = soil_data.rename({var: var.replace(f"{area}_", "") for var in soil_data.data_vars})

    # set buffer scales
    if buffer_scales is None:
        buffer_scales = [1, 5, 25]

    # aggregate to buffer scales level
    print(f"Processing soil data for {area}")
    for idx, row in tqdm.tqdm(stations_gdf.iterrows(), total = len(stations_gdf)):
        for buffer in buffer_scales: # (km)
            geom = row.geometry.buffer(buffer / 111)
            for var in list(soil_data.data_vars):
                try:
                    clipped_bdod = soil_data[var].rio.clip([geom], stations_gdf.crs)
                    mean_val = float(clipped_bdod.mean().values)
                    std_val = float(clipped_bdod.std().values)
                except NoDataInBounds:
                    mean_val = np.nan
                    std_val = np.nan
                    print(f"No data in bounds for {var}_{buffer}km")
                stations_gdf.loc[idx, f"{var}_{buffer}km_mean"] = mean_val
                stations_gdf.loc[idx, f"{var}_{buffer}km_std"] = std_val
    
    return stations_gdf

def main(is_mini = False):    
    # load config
    config = config = load_config()
    
    # get constants
    RAW_DATA = config['raw_data']
    PROCESSED_DATA = config['processed_data']
    AREAS = config['areas']
    TYPES = config['types']
    BBOX = config['bbox']
    CRS = config['crs']
    if is_mini:
        MINI_DATA = config['mini_data']
    else:
        MINI_DATA = None

    hydro_data = {}
    stations = {}
    
    # process hydrological data and stations info
    for area in AREAS:
        stations[area], hydro_data[area] = get_hydro_data(area = area, data_dir = RAW_DATA, mini_data_dir = MINI_DATA)

    # get soil data
    for area in AREAS:
        stations[area] = aggregate_soil_data(area = area, data_dir = RAW_DATA, station_gdf = stations[area])

    # get waterflow data


    # get meterological data

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--is-mini', default = False, dtype = bool, required = False)
    args = parser.parse_args()
    main(is_mini = args.is_mini)
import os
import tqdm
import argparse
from utils import load_config

import pandas as pd
import geopandas as gpd
from src.utils.data_loader import (
    load_station_info,
    load_hydro_data,
    load_water_flows,
    load_meteo_data,
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
    
def get_hydro_data(area, data_dir, sec_data_dir, crs = 'epsg:4326'):
    # load the data
    hydro_data = load_hydro_data(area, data_dir)
    
    # load stations list
    stations = get_stations_list(area, sec_data_dir, crs = crs)

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

def get_water_flow_data(area, data_dir):
    print(f'Loading water flow data for {area}')
    train_water_flow = load_water_flows(area, 'train', data_dir)
    eval_water_flow = load_water_flows(area, 'eval', data_dir)
    return {'train' : train_water_flow, 'eval' : eval_water_flow}

def aggregate_meteo_data(data, stations_gdf, max_date, buffer_scales = None, crs = 'epsg:4326'):
    # get buffer scales
    if buffer_scales is None:
        buffer_scales = [50, 100]
        
    # define output df
    df = None

    # establish key variables
    key_vars = {
        'precipitations' : 'tp', 
        'temperatures' : 't2m',
        'soil_moisture' : 'swvl1',
        'evaporation' : 'e'
    }

    # loop through and aggregate at different scales for the four variables
    # loop 1 - variable loop
    for key, var in key_vars.items():
        keys_list = []
        data_var = data[key][var]  # extract variable
        data_var = data_var.rio.write_crs(crs) #project to crs
        # loop 2 - stations loop
        for idx, row in stations_gdf.iterrows():
            lat, lon = row.geometry.y, row.geometry.x
            # sample a single point (each point is sampled at 0.25 degrees approximately at 27km)
            sampled_values = data_var.sel(latitude=lat, longitude=lon, method="nearest").to_dataframe().reset_index()
            # Filter by date range
            sampled_values = sampled_values[sampled_values.valid_time <= max_date]
            sampled_values["station_code"] = row.station_code
            sampled_values = sampled_values[['station_code', 'valid_time', var]]
            # loop 3 - sample at different buffer scales
            for buffer in buffer_scales:
                # select data within buffer
                geom = row.geometry.buffer(buffer / 111) # convert kilometer to degrees
                clipped_data = data_var.rio.clip([geom])
                buffer_values = clipped_data.mean(dim = ['latitude', 'longitude'])
                buffer_values = buffer_values.to_dataframe().reset_index()
                buffer_values = buffer_values[buffer_values.valid_time <= max_date]
                buffer_values['station_code'] = buffer_values["station_code"] = row.station_code
                buffer_values = buffer_values.rename(columns={var: f'{var}_{buffer}km'})
    
                # Merge with point values using date
                sampled_values = sampled_values.merge(
                    buffer_values[["valid_time", 'station_code', f'{var}_{buffer}km']],
                    on=['station_code', "valid_time"],
                    how="left"
                )
        
            keys_list.append(sampled_values)

        # Combine all DataFrames
        df_keys = pd.concat(keys_list, ignore_index=True)
        if not isinstance(df, pd.DataFrame):
            df = df_keys
        else:
            df = df.merge(
                df_keys,
                on = ['station_code', "valid_time"],
                how = 'left'
            )
    return df

def load_and_aggregate_meteo_data(area, data_dir, stations_gdf, max_date, buffer_scales = None, crs = 'epsg:4326'):
    # load meteo data
    meteo_train = load_meteo_data(area, 'train', data_dir)
    meteo_eval = load_meteo_data(area, 'eval', data_dir)

    # aggregate data
    # train
    train_stations_gdf = stations_gdf[~stations_gdf.eval_only]
    meteo_train = aggregate_meteo_data(
        data = meteo_train, 
        stations_gdf = train_stations_gdf, 
        max_date = max_date, 
        buffer_scales = buffer_scales, 
        crs = crs
    )
    # eval
    meteo_eval = aggregate_meteo_data(
        data = meteo_eval, 
        stations_gdf = stations_gdf, 
        max_date = max_date, 
        buffer_scales = buffer_scales, 
        crs = crs
    )
    return {'train' : meteo_train, 'eval' : meteo_eval}

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
        SEC_DATA = config['mini_data']
    else:
        SEC_DATA = RAW_DATA

    hydro_data = {}
    stations = {}
    water_flows = {}
    meteo_data = {}
    max_dates = {}
    
    # process hydrological data and stations info
    for area in AREAS:
        stations[area], hydro_data[area] = get_hydro_data(area = area, data_dir = RAW_DATA, sec_data_dir = SEC_DATA)

    # get soil data
    for area in AREAS:
        stations[area] = aggregate_soil_data(area = area, data_dir = RAW_DATA, station_gdf = stations[area])

    # get altitude data

    # get stations relationship

    # get waterflow data
    for area in AREAS:
        water_flows[area] = get_water_flow_data(area, data_dir = SEC_DATA)
        max_date[area] = water_flows[area]['eval'].ObsDate.max() + pd.Timedelta(7, unit = 'days')

    # get meterological data
    for area in AREAS:
        meteo_data[area] = load_and_aggregate_meteo_data(
            area = area, 
            data_dir = SEC_DATA, 
            stations_gdf = stations[area], 
            max_date = max_date[area], 
            buffer_scales = None, 
            crs = CRS
        )

    # merge all data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--is-mini', default = False, dtype = bool, required = False)
    args = parser.parse_args()
    main(is_mini = args.is_mini)
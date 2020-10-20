import netCDF4
import numpy as np
import argparse
import pickle

##########################################
# Load the dataset
##########################################
"""
load precipitation first and find out the wet days
"""
def load_PRECL():
    nc_fid = netCDF4.Dataset('./data/cam5.1_amip_1d_002.cam2.h1.PRECL.19790101-20051231.nc', 'r')
    # latitute
    lat = nc_fid.variables['lat'][:]
    # longitue
    lon = nc_fid.variables['lon'][:]
    # level
    lev = nc_fid.variables['lev'][:]
    # time
    time = nc_fid.variables['time']
    time = netCDF4.num2date(time[:], time.units, time.calendar)
    idx = []
    for j, t in enumerate(time):
        if t.month in [12, 1, 2]:
            idx.append(j)

    # reshape the variable of interest & select the period of interest
    var_value = nc_fid.variables['PRECL']
    # only check data for Dec, Jan, and Feb
    var_value = var_value[idx]

    # change the unit from m/s to mm/day
    var_value *= 8.64e7
    # First of all, find out the wet days
    wet_day_idx = np.where(var_value > 1.0)

    wet_day_prepc = var_value[wet_day_idx]
    # For each location, pick out the wet days that exceeds the 95% quantile of the PREC
    masks = []
    PRECL = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if np.where(var_value[:,i,j])[0].shape[0]!=0:
                wet_days = var_value[np.where(var_value[:,i,j]),i,j]
                upper_qunatile = np.quantile(wet_days, 0.95)
                idx = np.where(var_value[:,i,j]>upper_qunatile)[0]
                for tup in zip(idx, [i]*idx.shape[0], [j]*idx.shape[0]):
                    masks.append(tup)
                    PRECL.append(var_value[tup])
    print("filtered %d observations." % len(masks))

    # save mask to numpy array
    with open('prec_mask.pickle', 'wb') as f:
        pickle.dump(masks, f)

    PRECL = np.array(PRECL).reshape((-1, 1))    
    np.save('PRECL.npy', PRECL)



def processing_data(var):
    nc_fid = netCDF4.Dataset('./data/cam5.1_amip_1d_002.cam2.h1.' + var + '.19790101-20051231.nc',
                             'r')
    with open('prec_mask.pickle', 'rb') as f:
        masks = pickle.load(f)
    var_value = nc_fid.variables[var]
    # del nc_fid

    filtered_var_value = []
    for tup in masks:
        filtered_var_value.append(var_value[tup[0],:,tup[1],tup[2]].data)
    
    filtered_var_value = np.stack(filtered_var_value)
    print(filtered_var_value.shape)
    np.save(var + '.npy', filtered_var_value)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocessing for CAM dataset')
    parser.add_argument('--var', default='OMEGA', choices=['Q', 'T', 'OMEGA'])
    args = parser.parse_args()

    processing_data(args.var)

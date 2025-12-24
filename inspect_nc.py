
import netCDF4
import os

files = [
    '/Users/zhongwangwei/Desktop/Github/pycama/data/sed/nextxy.nc',
    '/Users/zhongwangwei/Desktop/Github/pycama/data/sed/sedfrc_v2_final.nc',
    '/Users/zhongwangwei/Desktop/Github/pycama/data/sed/slope.nc'
]

for fpath in files:
    print(f"--- Inspecting {os.path.basename(fpath)} ---")
    try:
        with netCDF4.Dataset(fpath, 'r') as nc:
            print("Dimensions:")
            for d in nc.dimensions:
                print(f"  {d}: {len(nc.dimensions[d])}")
            print("Variables:")
            for v in nc.variables:
                print(f"  {v}: {nc.variables[v].dtype} {nc.variables[v].dimensions} {nc.variables[v].shape}")
                # Print attributes if needed
                # for attr in nc.variables[v].ncattrs():
                #     print(f"    {attr}: {getattr(nc.variables[v], attr)}")
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
    print()

"""
Tools for dam parameter calculation, equivalent to src_dam
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from .fortran_io import FortranBinary, read_params_txt

from decimal import Decimal, getcontext

class DamParamProcessor:
    """Handles dam parameter calculations"""

    def __init__(self, config):
        self.config = config
        self.imis = -9999
        self.rmis = 1.e20
        getcontext().prec = 50 # Set precision for Decimal

    def calc_annual_max_mean(self, map_dir, temp_dir):
        """
        Calculate mean and annual max discharge from naturalized simulations.
        Equivalent to src_dam/script/p01_get_annualmax_mean.py
        """
        print("=" * 60)
        print("DAM_PARAMS: Calculating mean and annual max discharge")
        print("=" * 60)

        syear = int(self.config.get('RiverMap_Gen', 'syear'))
        eyear = int(self.config.get('RiverMap_Gen', 'eyear'))
        dt = int(self.config.get('RiverMap_Gen', 'dt'))
        natsim_dir = self.config.get('RiverMap_Gen', 'natsim_dir')
        # GRanD_river.txt is intermediate output from dam allocation step
        dam_file = os.path.join(map_dir, 'GRanD_river.txt')

        # Get map dimensions
        params = read_params_txt(os.path.join(map_dir, 'params.txt'))
        nx, ny = params['nx'], params['ny']
        print(f'CaMa map dim (nx,ny): {nx}, {ny}')

        damcsv = pd.read_csv(dam_file, sep='\s+', header=0, skipinitialspace=True)
        ndams = len(damcsv)
        print(f'Number of dams: {ndams}')

        maxdays = 1  # Number of days to consider extreme values in a year
        max_outf = os.path.join(temp_dir, 'tmp_p01_AnnualMax.bin')
        mean_outf = os.path.join(temp_dir, 'tmp_p01_AnnualMean.bin')

        years = eyear - syear + 1
        max_finarray = np.zeros((years * maxdays, ndams), dtype=np.float32)
        mean_yeararray = np.zeros((years, ndams), dtype=np.float32)

        x_arr = damcsv['ix'].values - 1
        y_arr = damcsv['iy'].values - 1

        for i, year in enumerate(range(syear, eyear + 1)):
            print(f'\nRead natsim outflw: year={year}')
            outflw_file = os.path.join(natsim_dir, f'outflw{year}.bin')
            if not os.path.exists(outflw_file):
                print(f'  WARNING: {outflw_file} not found, skipping year.')
                continue

            outflw_all = np.fromfile(outflw_file, 'float32').reshape(-1, ny, nx)
            outflw_dam = outflw_all[:, y_arr, x_arr]
            print(f'  outflw_dam.shape: {outflw_dam.shape}')

            # Annual mean with standard numpy float
            mean_yeararray[i, :] = np.mean(outflw_dam, axis=0)

            print(f'  mean: {mean_yeararray[i, :5]}')

            # Annual maximum
            for j, row in damcsv.iterrows():
                outflw = outflw_dam[:, j]
                maxindex = argrelmax(outflw, order=8 * 7)[0]
                if len(maxindex) > 0:
                    maxarray = outflw[maxindex]
                    maxarray_sorted = np.sort(maxarray)[::-1]
                    max_finarray[i * maxdays:(i + 1) * maxdays, j] = maxarray_sorted[0:maxdays]
                else:
                    outflw_sorted = np.sort(outflw)[::-1]
                    max_finarray[i * maxdays:(i + 1) * maxdays, j] = outflw_sorted[0:maxdays]
            print(f'  max: {max_finarray[i * maxdays, :5]}')

        print('\nSave flood and mean discharge at dam grids')
        max_finarray.astype('float32').tofile(max_outf)
        
        mean_finarray = np.mean(mean_yeararray, axis=0)

        mean_finarray = np.where(mean_finarray < 1.E-10, 1.E-10, mean_finarray)
        mean_finarray.astype('float32').tofile(mean_outf)

        print(f'Output Plain Binary Files')
        print(f'-- flood discharge [{years} * {ndams}] {max_outf}')
        print(f'-- mean  discharge [{ndams}] {mean_outf}')
        print('###########################################\n')

    def calc_100yr_discharge(self, map_dir, temp_dir):
        """
        Calculate 100-year discharge from annual max discharge.
        Equivalent to src_dam/script/p02_get_100yrDischarge.py
        """
        print("=" * 60)
        print("DAM_PARAMS: Calculating 100-year discharge")
        print("=" * 60)

        syear = int(self.config.get('RiverMap_Gen', 'syear'))
        eyear = int(self.config.get('RiverMap_Gen', 'eyear'))
        # GRanD_river.txt is intermediate output from dam allocation step
        dam_file = os.path.join(map_dir, 'GRanD_river.txt')
        pyear = 100 # Return period

        df = pd.read_csv(dam_file, sep='\s+', header=0, skipinitialspace=True)
        ndams = len(df)
        years = eyear - syear + 1

        readdatapath = os.path.join(temp_dir, 'tmp_p01_AnnualMax.bin')
        print(f'Read annual max files: {readdatapath}')
        readdata = np.fromfile(str(readdatapath), 'float32').reshape(years, ndams)

        outputpath = os.path.join(temp_dir, f'tmp_p02_{pyear}year.bin')

        # Gumbel distribution functions
        def plotting_position(n, alpha=0.0):
            ii = np.arange(n) + 1
            pp = (ii - alpha) / (n + 1 - 2 * alpha)
            return pp

        def gum(xx, pp, pyear):
            def func_gum(xx):
                n = len(xx)
                b0 = np.sum(xx) / n
                j = np.arange(n)
                b1 = np.sum(j * xx) / n / (n - 1)
                lam1 = b0
                lam2 = 2 * b1 - b0
                aa = lam2 / np.log(2)
                cc = lam1 - 0.5772 * aa
                return aa, cc
            
            def est_gum(aa, cc, pp):
                return cc - aa * np.log(-np.log(pp))

            aa, cc = func_gum(xx)
            ye = est_gum(aa, cc, pp)
            rr = np.corrcoef(xx, ye)[0][1]
            prob = 1.0 - 1.0 / pyear
            yp = est_gum(aa, cc, prob)
            return yp

        finarray = np.zeros((ndams), dtype=np.float32)
        pps = plotting_position(years)
        for dam in range(ndams):
            site_arr = readdata[:, dam]
            if np.max(site_arr) >= 1e+20 or np.max(site_arr) == np.min(site_arr):
                finarray[dam] = np.nan
                continue

            site_arr = np.where(site_arr < 0, 0, site_arr)
            site_arr = np.sort(site_arr)
            yp = gum(site_arr, pps, pyear)

            if yp > 0:
                finarray[dam] = yp
            else:
                finarray[dam] = np.nan

        finarray.astype('float32').tofile(outputpath)

        print('Output Plain Binary Files')
        print(f'-- 100yr-discharge [{ndams}] {outputpath}')
        print('###########################################\n')

    def est_fldsto_surfacearea(self, map_dir, temp_dir):
        """
        Estimate dam storage parameter from GRSAD and ReGeom data.
        Equivalent to src_dam/script/p03_est_fldsto_surfacearea.py
        """
        print("=" * 60)
        print("DAM_PARAMS: Estimating flood storage and surface area")
        print("=" * 60)

        # GRanD_river.txt is intermediate output from dam allocation step
        dam_file = os.path.join(map_dir, 'GRanD_river.txt')
        grsad_dir = self.config.get('RiverMap_Gen', 'grsad_dir')
        regeom_dir = self.config.get('RiverMap_Gen', 'regeom_dir')
        output_file = os.path.join(temp_dir, 'tmp_p03_fldsto.csv')

        pc = 75  # Percentile for Normal Water Level

        grand = pd.read_csv(dam_file, sep='\s+', header=0, skipinitialspace=True)
        cols = ['ID', 'damname', 'ave_area', 'fldsto_mcm', 'totalsto_mcm']
        df_new = pd.DataFrame(columns=cols)

        for i in range(len(grand)):
            gr = grand.iloc[i:i + 1]
            damid = gr['ID'].values[0]
            damname = gr['damname'].values[0]
            totalsto = gr['cap_mcm'].values[0]

            grsadpath = os.path.join(grsad_dir, f'{damid}_intp')
            if not os.path.isfile(grsadpath):
                df_i = pd.DataFrame([[damid, damname, -998, -998, totalsto]], columns=cols)
                df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)
                continue

            df = pd.read_table(grsadpath, index_col=0, parse_dates=True)
            data = df.dropna()

            if np.max(df['3water_enh'].value_counts()) > 12:
                rm_df = df['3water_enh'].value_counts()
                rm_df = rm_df[rm_df > 12]
                rm_df = rm_df.index
                # Match original code: replace values one by one
                for j in range(len(rm_df)):
                    rm_val = rm_df[j]
                    data.loc[:, ('3water_enh')] = data['3water_enh'].replace(rm_val, np.nan)
                data = data.dropna()

            data = data['3water_enh']
            if len(data) < 2:
                df_i = pd.DataFrame([[damid, damname, -997, -997, totalsto]], columns=cols)
                df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)
                continue

            fld_area = np.percentile(data, pc)
            areamax = np.max(data)

            regeompath = os.path.join(regeom_dir, f'{damid}.csv')
            if not os.path.isfile(regeompath):
                df_i = pd.DataFrame([[damid, damname, -996, -996, totalsto]], columns=cols)
                df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)
                continue

            regeom = pd.read_csv(regeompath, header=7)
            regeom.columns = ['Depth', 'Area', 'Storage']
            if len(regeom) <= 1:
                df_i = pd.DataFrame([[damid, damname, -999, -999, totalsto]], columns=cols)
                df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)
                continue

            fld_area = fld_area * regeom['Area'].values[-1] / areamax

            fld_sto, sto_max = 0, 0
            for i_rg in range(len(regeom)):
                rg = regeom.iloc[i_rg:i_rg + 1]
                if rg['Area'].values[0] < fld_area:
                    continue
                elif rg['Area'].values[0] == fld_area:
                    use_sto = np.mean(regeom.query('Area == @fld_area')['Storage'])
                    sto_max = np.mean(regeom.query('Area == @fld_area')['Storage'])  # Match original code
                    use_sto = use_sto * totalsto / regeom['Storage'].values[-1]
                    fld_sto = totalsto - use_sto
                    break
                elif rg['Area'].values[0] > fld_area:
                    sto_max = rg['Storage'].values[0]
                    area_max = rg['Area'].values[0]
                    rg_p = regeom.iloc[i_rg - 1:i_rg]
                    sto_min = rg_p['Storage'].values[0]
                    area_min = rg_p['Area'].values[0]

                    if( area_max == area_min ):
                        use_sto = sto_min
                        fld_sto = totalsto - use_sto
                        break  # Match original code: break here

                    if( area_min <= fld_area ):
                        use_sto = sto_min + (sto_max - sto_min) * (fld_area - area_min) / (area_max - area_min)  ## linearly interporlate
                        use_sto = use_sto * totalsto / regeom['Storage'].values[-1]   ## adjustment to fit GranD original data
                        fld_sto = totalsto - use_sto

                        if( use_sto > totalsto ):
                            use_sto = totalsto
                            fld_sto = 0
                        # Match original code: NO break here - continues to next iteration

            if sto_max == 0:
                print(f'{damid} ERR: sto_max == 0')
                area_max = regeom['Area'].values[-1]
                use_sto = np.mean(regeom.query('Area == @area_max')['Storage'])
                use_sto = use_sto * totalsto / regeom['Storage'].values[-1]
                fld_sto = totalsto - use_sto
                print(fld_sto, totalsto)
                raise RuntimeError(f'Dam {damid}: sto_max == 0, terminating as in original code')

            if fld_sto == 0:
                print('error!')
                print(fld_area, rg['Area'].values[0])
                raise RuntimeError(f'Dam {damid}: fld_sto == 0, terminating as in original code')

            if fld_sto < 0:
                fld_sto = 0.0

            df_i = pd.DataFrame([[damid, damname, fld_area, fld_sto, totalsto]], columns=cols)
            df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)

        print(df_new)
        df_new.to_csv(output_file)
        print(output_file)
        print('##################################')

    def complete_dam_csv(self, map_dir, temp_dir):
        """
        Merge all temporary files and create the final dam_param.csv.
        Equivalent to src_dam/script/p04_complete_damcsv.py
        """
        print("=" * 60)
        print("DAM_PARAMS: Completing dam parameter CSV")
        print("=" * 60)

        min_uparea = int(self.config.get('RiverMap_Gen', 'min_uparea'))
        # GRanD_river.txt is intermediate output from dam allocation step
        dam_file = os.path.join(map_dir, 'GRanD_river.txt')
        output_file = os.path.join(temp_dir, 'tmp_p04_damparams.csv')

        qmean_file = os.path.join(temp_dir, 'tmp_p01_AnnualMean.bin')
        q100_file = os.path.join(temp_dir, 'tmp_p02_100year.bin')
        storage_file = os.path.join(temp_dir, 'tmp_p03_fldsto.csv')

        qn_all = np.fromfile(qmean_file, 'float32')
        q100_all = np.fromfile(q100_file, 'float32')

        damcsv = pd.read_csv(dam_file, sep='\s+', header=0, skipinitialspace=True)
        damcsv['Qn'] = qn_all
        damcsv['Qf'] = q100_all * 0.3

        stocsv = pd.read_csv(storage_file)

        fldsto_l, consto_l, qf_new = [], [], []
        for index, row in damcsv.iterrows():
            damid = row['ID']
            fldsto = stocsv.query('ID == @damid')['fldsto_mcm'].values[0]
            totalsto = stocsv.query('ID == @damid')['totalsto_mcm'].values[0]

            if pd.isna(fldsto) or fldsto < -99:
                fldsto = totalsto * 0.37
            
            consto = totalsto - fldsto
            fldsto_l.append(fldsto)
            consto_l.append(consto)

            qf, qn = row['Qf'], row['Qn']
            if qf < qn:
                qf_new.append(q100_all[index] * 0.4 if q100_all[index] * 0.4 >= qn else qn * 1.1)
            else:
                qf_new.append(qf)

        damcsv['fldsto_mcm'] = fldsto_l
        damcsv['consto_mcm'] = consto_l
        damcsv['Qf'] = qf_new

        damcsv = damcsv.query('area_CaMa >= @min_uparea').dropna()

        # Treat multiple dams in one grid
        cnt = damcsv.groupby(['ix', 'iy']).size().to_dict()
        damcsv2 = damcsv.copy()
        for k, v in cnt.items():
            if v > 1:
                ix, iy = k
                dams = damcsv.query('ix == @ix & iy == @iy')
                maxsto = dams['cap_mcm'].max()
                rmdams = dams[dams['cap_mcm'] != maxsto]
                if len(rmdams) == 0:
                    maxfsto = dams['fldsto_mcm'].max()
                    rmdams = dams[dams['fldsto_mcm'] != maxfsto]
                damcsv2.drop(index=rmdams.index, inplace=True)

        # Format and save final CSV
        damcsv2 = damcsv2.rename(columns={'ID': 'GRAND_ID', 'damname':'DamName', 'lon':'DamLon', 'lat':'DamLat', 'ix':'DamIX', 'iy':'DamIY', 'fldsto_mcm':'FldVol_mcm', 'consto_mcm':'ConVol_mcm', 'cap_mcm':'TotalVol_mcm'})
        final_cols = ['GRAND_ID', 'DamName', 'DamLat', 'DamLon', 'area_CaMa', 'DamIX', 'DamIY', 'FldVol_mcm', 'ConVol_mcm', 'TotalVol_mcm', 'Qn', 'Qf', 'year']
        damcsv2 = damcsv2[final_cols]
        damcsv2.to_csv(output_file, index=None)

        print('Final dam parameters saved to:', output_file)
        print('###########################################\\n')



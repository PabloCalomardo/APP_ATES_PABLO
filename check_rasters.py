import rasterio
import numpy as np

rasters = [
    'Verificador/CORRECTE_BOW_SUMMIT.tif',
    'Verificador/CORRECTE_CONNAUGHT.tif',
    'Verificador/EXPERIMENT_BOW_SUMMIT/Definitive_Layers/Ponderador_ATES.tif',
    'Verificador/EXPERIMENT_CONNAUGHT/Definitive_Layers/Ponderador_ATES.tif'
]

data = {}

for r in rasters:
    print(f'\\n--- {r} ---')
    with rasterio.open(r) as src:
        band = src.read(1)
        meta = {
            'shape': src.shape,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'crs': src.crs,
            'transform': src.transform,
            'res': src.res
        }
        data[r] = meta
        print(f"Shape: {meta['shape']}")
        print(f"Dtype: {meta['dtype']}")
        print(f"NoData: {meta['nodata']}")
        print(f"CRS: {meta['crs']}")
        print(f"Transform: {meta['transform']}")
        
        valid_data = band[band != src.nodata] if src.nodata is not None else band
        if valid_data.size > 0:
            u, c = np.unique(valid_data, return_counts=True)
            print("Unique values (value: count):")
            for val, count in zip(u, c):
                print(f"  {val}: {count}")
            print(f"Min: {np.min(valid_data)}, Max: {np.max(valid_data)}")
        else:
            print("No valid data found.")

def compare(r1, r2):
    print(f'\\n--- Comparison: {r1} vs {r2} ---')
    m1, m2 = data[r1], data[r2]
    match_shape = m1['shape'] == m2['shape']
    match_trans = m1['transform'] == m2['transform']
    match_crs = m1['crs'] == m2['crs']
    print(f"Shape match: {match_shape}")
    print(f"Transform match: {match_trans}")
    print(f"CRS match: {match_crs}")
    if match_shape and match_trans and match_crs:
        print("EXACT MATCH in geometry/metadata.")
    else:
        print("MISMATCH detected.")

compare('Verificador/CORRECTE_BOW_SUMMIT.tif', 'Verificador/EXPERIMENT_BOW_SUMMIT/Definitive_Layers/Ponderador_ATES.tif')
compare('Verificador/CORRECTE_CONNAUGHT.tif', 'Verificador/EXPERIMENT_CONNAUGHT/Definitive_Layers/Ponderador_ATES.tif')

import os
import numpy as np
import cv2
import csv
from pathlib import Path

def save_depth_map_npy(depth_map, path="depth_map.npy"):
	"""
	Salva la depth map corrente su disco in formato NumPy (.npy).
	Ritorna True se va a buon fine, False altrimenti.
	"""
	if depth_map is None:
		print("⚠ Nessuna depth_map da salvare (depth_map è None).")
		return False

	# Crea la cartella se serve
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

	# Salva come array float32
	np.save(path, depth_map.astype(np.float32))
	print(f"✓ Depth map salvata in: {path}")
	return True

def save_depth_map_png( depth_map ,path="depth_map.png"):
    """
    Salva la depth map come immagine 8-bit (0-255) in scala di grigi.
    """
    if depth_map is None:
        print("⚠ Nessuna depth_map da salvare.")
        return False

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    dm = depth_map.copy()
    # Normalizza in [0, 1] se non lo è già
    dm = np.nan_to_num(dm, nan=0.0)
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)
    dm_uint8 = (dm * 255).astype(np.uint8)

    cv2.imwrite(path, dm_uint8)
    print(f"✓ Depth map (immagine) salvata in: {path}")
    return True




def depth_map_to_csv(depth_map, filename='depth_map.csv', format_type='simple', step=5):
    """
    Converte depth map in CSV. [code_file:2][code_file:3][code_file:4]
    
    Args:
        depth_map: np.ndarray (H, W) float32 [0,1]
        filename: nome file CSV di output
        format_type: 'simple', 'pixel', 'subsampled'
        step: campionamento per 'subsampled' (default=5)
    
    Returns: True se salvato con successo
    """
    
    # Crea cartella se non esiste
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'simple':
        # 🔹 MATRICE DIRETTA (più veloce)
        np.savetxt(filename, depth_map, delimiter=',', fmt='%.6f')
        print(f'✓ SIMPLE: {filename} ({depth_map.shape})')
        
    elif format_type == 'pixel':
        # 🔹 OGNI PIXEL SU RIGA (per Excel/Pandas)
        h, w = depth_map.shape
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'depth'])
            for y in range(h):
                for x in range(w):
                    writer.writerow([x, y, f'{depth_map[y,x]:.6f}'])
        print(f'✓ PIXEL: {filename} ({h*w:,} righe)')
        
    elif format_type == 'subsampled':
        # 🔹 LEGGERO (96% compressione con step=5)
        h, w = depth_map.shape
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'depth'])
            for y in range(0, h, step):
                for x in range(0, w, step):
                    writer.writerow([x, y, f'{depth_map[y,x]:.6f}'])
        print(f'✓ SUBSAMPLED: {filename} (step={step})')
    
    else:
        print('❌ Formato non valido. Usa: simple, pixel, subsampled')
        return False
    
    return True
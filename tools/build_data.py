"""Converte o pacote de funcoes de transferencia da ISO 2631-5 (formato MATLAB/CSV)
para arquivos .npz compactos usados pelo aplicativo.

Uso:
    python build_data.py ISO-DIS_2631-5_Transferfunctions_used_in_the_software_MATLAB_format.zip saida/

Roda uma vez so. Depois disso o zip original nao e mais necessario.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

# nome do zip interno -> numero do grupo de postura usado pela norma
POSTURES = {
    1: 'group1_UefuBSK_mat.zip',
    2: 'group2_UefuBSM_mat.zip',
    3: 'group3_UefuBSK_mat.zip',
    4: 'forwarder_UefuBSK_mat.zip',
    5: 'harvester_UefuBSK_mat.zip',
    7: 'handsinlap-UefuBSK_mat.zip',
}

N_ANTHRO = 30   # (classe IMC, percentil de massa, subclasse)
N_LOC = 4       # excitacao: assento, pes, maos, encosto
N_DIR = 3       # direcao da excitacao: x, y, z
N_FD = 3        # direcao da forca: x, y, compressao
N_DISC = 6      # T12/L1 ... L5/S1
N_FREQ = 120    # 0,25 a 30 Hz


def parse_complex_csv(raw):
    txt = raw.decode('latin-1').strip()
    rows = [[complex(v.replace('i', 'j')) for v in ln.split(',')]
            for ln in txt.splitlines()]
    return np.array(rows)


def convert_posture(zf, member, out_path):
    with zf.open(member) as fh:
        inner = zipfile.ZipFile(io.BytesIO(fh.read()))

    H = np.zeros((N_ANTHRO, N_LOC, N_DIR, N_FD, N_DISC, N_FREQ), dtype=np.complex64)
    static = np.zeros((N_ANTHRO, N_LOC, N_DIR, N_FD, N_DISC), dtype=np.float64)
    freq = None

    for k in range(1, N_ANTHRO + 1):
        for el in range(1, N_LOC + 1):
            for ed in range(1, N_DIR + 1):
                arr = parse_complex_csv(inner.read(f'BSKUefuM_{k}_{el}_{ed}'))
                pages = arr.reshape(N_FD, N_DISC + 1, N_FREQ + 1)
                if freq is None:
                    freq = np.real(pages[0, 0, 1:]).copy()
                static[k - 1, el - 1, ed - 1] = np.real(pages[:, 1:, 0])
                H[k - 1, el - 1, ed - 1] = pages[:, 1:, 1:]
        print(f'    antropometria {k}/{N_ANTHRO}', end='\r', flush=True)

    np.savez_compressed(out_path, H=H, static=static, freq=freq)
    print(f'    -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)')


def main(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)

    names = set(zf.namelist())
    for posture, member in POSTURES.items():
        if member not in names:
            print(f'  postura {posture}: "{member}" nao encontrado no zip, pulando')
            continue
        print(f'  postura {posture} ({member})')
        convert_posture(zf, member, out_dir / f'posture{posture}.npz')

    # forcas estaticas: 7 posturas x 10 antropometrias x 6 discos, colunas x/y/z
    sf = np.loadtxt(io.BytesIO(zf.read('static_forces')), delimiter=',')
    np.savez_compressed(out_dir / 'static_forces.npz',
                        static=sf.reshape(7, 10, N_DISC, 3))
    print('  -> static_forces.npz')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])

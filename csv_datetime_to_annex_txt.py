"""Converte CSV "DateTime,AX,AY,AZ" (aceleracao em g) do firmware
VCI_nRF52840_accel_SD para o .txt ASCII de 4 colunas (tempo, ax, ay, az, em
m/s2, base de tempo uniforme) usado tanto pelo Anexo A quanto pela Clausula
5 / Anexo C (core.read_excitation le os dois do mesmo jeito).

Esse firmware amostra por hardware (deadline scheduling via micros(), nao
um loop livre) -- a taxa configurada no .ino (--rate) e exata, mais
confiavel que estimar a partir do timestamp DateTime, que só tem resolucao
de milissegundos (grosseiro demais pra medir um periodo de ~1,2ms a 833Hz).

Uso:
    python csv_datetime_to_annex_txt.py --rate 833 arquivo1.csv [arquivo2.csv ...]
    python csv_datetime_to_annex_txt.py --rate 833 pasta_com_csvs/
    python csv_datetime_to_annex_txt.py arquivo.csv          # sem --rate: infere da media do DateTime (menos preciso)
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iso26315 import fielddata as fd


def _parse_datetime(s):
    return datetime.strptime(s.strip(), '%Y-%m-%d %H:%M:%S.%f')


_META_RE = re.compile(r'rate_hz=([\d.]+)')


def _read_segments(path):
    """Le o CSV e separa em segmentos a cada linha de cabecalho repetida.

    Firmwares mais antigos geravam o nome do arquivo a partir da hora de
    COMPILACAO, nao da hora real de cada boot -- resetar a placa reabria o
    mesmo nome (em modo append), concatenando sessoes diferentes no mesmo
    arquivo, cada uma com sua propria contagem de tempo. Tratar isso como
    uma unica serie continua inventaria uma base de tempo que nunca
    existiu -- por isso ainda separamos por seguranca, mesmo que o
    firmware atual (VCI_nRF52840_accel_SD_v2.ino) ja nao tenha mais esse
    bug (nome de arquivo por indice de sessao persistente).

    Se o arquivo tiver uma linha "# rate_hz=<valor> ..." antes do
    cabecalho (escrita pelo firmware v2), a taxa vem dali -- ver
    metadata_rate no retorno de cada segmento.

    Retorna uma lista de (linhas, metadata_rate_ou_None).
    """
    path = Path(path)
    segments = []          # lista de listas de (datetime_str, [ax,ay,az])
    metadata_rates = []    # taxa lida do "# rate_hz=..." de cada segmento
    pending_rate = None
    idx_t = idxs = None

    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()

            if first.startswith('#'):
                m = _META_RE.search(first)
                if m:
                    pending_rate = float(m.group(1))
                continue

            if first == 'DateTime':
                header = [c.strip() for c in row]
                idx_t = header.index('DateTime')
                idxs = [header.index(c) for c in ('AX', 'AY', 'AZ')]
                segments.append([])
                metadata_rates.append(pending_rate)
                pending_rate = None
                continue

            segments[-1].append((row[idx_t], [float(row[i]) for i in idxs]))

    return [(seg, rate) for seg, rate in zip(segments, metadata_rates) if seg]


def convert(path, rate=None, out_dir=None):
    path = Path(path)
    segments = _read_segments(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    results = []

    for i, (rows, meta_rate) in enumerate(segments):
        n = len(rows)
        if n < 2:
            print(f'{path.name} segmento {i + 1}: so {n} amostra(s), pulando')
            continue

        # Prioridade: --rate explicito na linha de comando > metadado no
        # cabecalho do arquivo (firmware v2) > estimativa pelo DateTime
        # (ultimo recurso, resolucao de 1ms so, grosseira).
        seg_rate = rate
        if seg_rate is None and meta_rate is not None:
            seg_rate = meta_rate
            print(f'{path.name} segmento {i + 1}: taxa lida do cabecalho '
                  f'do arquivo: {seg_rate:.2f} Hz')
        if seg_rate is None:
            t0 = _parse_datetime(rows[0][0])
            t1 = _parse_datetime(rows[-1][0])
            span_s = (t1 - t0).total_seconds()
            seg_rate = (n - 1) / span_s
            print(f'{path.name} segmento {i + 1}: --rate nao informado e '
                  f'sem metadado no arquivo, taxa estimada pelo DateTime '
                  f'(resolucao de 1ms, aproximada): {seg_rate:.2f} Hz')

        dt = 1.0 / seg_rate
        accel_g = np.array([axes for _, axes in rows], dtype=np.float64)
        # Remove o offset estatico (gravidade + orientacao do sensor) por
        # eixo antes de escrever. O Anexo A/E mede vibracao DINAMICA -- sem
        # isso, um eixo proximo da vertical fica com RMS perto de 1g (9,8
        # m/s2) so pela gravidade, inflando artificialmente a dose/risco.
        # Mesma convencao do fielddata.blocks_si(remove_gravity=True).
        accel_g = accel_g - accel_g.mean(axis=0, keepdims=True)
        accel_ms2 = accel_g * fd.GRAVITY_BH

        suffix = f'_part{i + 1}' if len(segments) > 1 else ''
        out_path = out_dir / (path.stem + suffix + '.txt')
        with open(out_path, 'w') as f:
            for j, (ax, ay, az) in enumerate(accel_ms2):
                f.write(f'{j * dt:.6f} {ax:.6f} {ay:.6f} {az:.6f}\n')

        results.append((out_path, n, seg_rate))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inputs', nargs='+', help='arquivo(s) .csv ou pasta(s)')
    ap.add_argument('--rate', type=float, default=None,
                    help='taxa de amostragem em Hz (recomendado: use o '
                         'valor configurado no firmware, ex. 833)')
    ap.add_argument('--out-dir', default=None,
                    help='pasta de saida (padrao: mesma pasta do .csv)')
    args = ap.parse_args()

    paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob('*.csv')))
        elif p.is_file():
            paths.append(p)

    if not paths:
        print('Nenhum arquivo .csv encontrado.')
        return

    for p in paths:
        try:
            results = convert(p, rate=args.rate, out_dir=args.out_dir)
            if not results:
                print(f'{p.name}: nenhum segmento com dado suficiente')
                continue
            for out_path, n, seg_rate in results:
                print(f'{p.name} -> {out_path.name}  '
                     f'({n} amostras, {seg_rate:.2f} Hz)')
        except Exception as e:
            print(f'ERRO em {p.name}: {e}')


if __name__ == '__main__':
    main()

"""Parser para o formato binário gravado pelo firmware do Xiao nRF52840 Sense.

Formato (little-endian, structs empacotados sem padding — ver o firmware .ino):

Cabeçalho (14 bytes, uma vez no início de cada arquivo):
    uint32  magic         'VIB1' (0x31424956)
    uint16  ano
    uint8   mes, dia, hora, minuto, segundo
    uint16  sampleRateHz
    uint8   fullScaleG    (2, 4, 8 ou 16)

Pacote de dados (10 bytes, repetido N vezes após o cabeçalho):
    uint32  tempoMicros   elapsed desde o cabeçalho DESTE arquivo
    int16   x, y, z       contagem bruta do acelerômetro (LSM6DS3)

Um arquivo cobre no máximo ~1h de gravação (rotação no firmware), então
tempoMicros nunca chega perto do overflow de uint32 (~71,6 min).
"""
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

MAGIC = 0x31424956  # 'VIB1'

_HEADER_FMT = '<IHBBBBBHB'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 14 bytes

_PACKET_FMT = '<Ihhh'
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)  # 10 bytes

# mg/LSB por faixa de escala, conforme datasheet do LSM6DS3
_SENSITIVITY_MG_PER_LSB = {2: 0.061, 4: 0.122, 8: 0.244, 16: 0.488}

G_SI = 9.80665  # m/s^2 por g


@dataclass
class XiaoFile:
    path: Path
    start: datetime
    sample_rate_hz: int
    full_scale_g: int
    t: np.ndarray          # (N,) segundos desde o início do arquivo
    counts: np.ndarray     # (N, 3) int16, contagem bruta [x, y, z]

    def to_si(self) -> np.ndarray:
        """Converte contagem bruta para m/s^2 usando a sensibilidade da faixa gravada."""
        sens_g = _SENSITIVITY_MG_PER_LSB[self.full_scale_g] / 1000.0
        return self.counts.astype(np.float64) * sens_g * G_SI

    def timestamps(self) -> np.ndarray:
        """Datetime absoluto de cada amostra (início do arquivo + tempo relativo)."""
        return np.array([self.start + timedelta(seconds=float(s)) for s in self.t])

    def gaps(self, tol_factor: float = 2.0):
        """Índices onde o intervalo entre amostras consecutivas excede tol_factor / sample_rate_hz.

        O firmware não interpola nem descarta silenciosamente: qualquer atraso de
        escrita no SD aparece aqui como um salto real no timestamp.
        """
        if len(self.t) < 2:
            return []
        dt = np.diff(self.t)
        expected = 1.0 / self.sample_rate_hz
        bad = np.flatnonzero(dt > expected * tol_factor)
        return [(int(i) + 1, float(dt[i])) for i in bad]


def parse_file(path) -> XiaoFile:
    path = Path(path)
    raw = path.read_bytes()

    if len(raw) < _HEADER_SIZE:
        raise ValueError(f'{path.name}: arquivo menor que o cabeçalho ({len(raw)} bytes)')

    magic, ano, mes, dia, hora, minuto, segundo, sample_rate_hz, full_scale_g = \
        struct.unpack_from(_HEADER_FMT, raw, 0)

    if magic != MAGIC:
        raise ValueError(
            f'{path.name}: assinatura inválida (0x{magic:08X}), '
            'arquivo corrompido ou formato errado')

    start = datetime(ano, mes, dia, hora, minuto, segundo)

    payload = raw[_HEADER_SIZE:]
    n_packets, resto = divmod(len(payload), _PACKET_SIZE)
    if resto:
        # último pacote truncado (ex: reset no meio de uma escrita) -- ignora o resto
        payload = payload[:n_packets * _PACKET_SIZE]

    if n_packets == 0:
        return XiaoFile(path, start, sample_rate_hz, full_scale_g,
                         t=np.empty(0), counts=np.empty((0, 3), dtype=np.int16))

    arr = np.frombuffer(payload, dtype=np.dtype([
        ('t_us', '<u4'), ('x', '<i2'), ('y', '<i2'), ('z', '<i2'),
    ]))

    t = arr['t_us'].astype(np.float64) / 1e6
    counts = np.stack([arr['x'], arr['y'], arr['z']], axis=1)

    return XiaoFile(path, start, sample_rate_hz, full_scale_g, t=t, counts=counts)


def parse_session(directory, pattern='F*.DAT') -> list:
    """Carrega todos os arquivos de uma sessão (F00000.DAT, F00001.DAT, ...) em ordem,
    e avisa sobre gap entre o fim de um arquivo e o início do próximo."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    parsed = [parse_file(f) for f in files]

    for prev, cur in zip(parsed, parsed[1:]):
        if len(prev.t) == 0:
            continue
        fim_prev = prev.start + timedelta(seconds=float(prev.t[-1]))
        gap = (cur.start - fim_prev).total_seconds()
        if gap > 1.0:  # a rotação normal (fechar+abrir arquivo) já causa um gap pequeno
            print(f'Aviso: gap de {gap:.2f}s entre {prev.path.name} e {cur.path.name}')

    return parsed


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('uso: python xiao_parser.py <arquivo.dat ou diretório>')
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        sessao = parse_session(target)
        for xf in sessao:
            n_gaps = len(xf.gaps())
            print(f'{xf.path.name}: início {xf.start}, {len(xf.t)} amostras, '
                  f'{xf.sample_rate_hz} Hz, +-{xf.full_scale_g}g, {n_gaps} gap(s)')
    else:
        xf = parse_file(target)
        print(f'início: {xf.start}')
        print(f'amostras: {len(xf.t)}  taxa: {xf.sample_rate_hz} Hz  faixa: +-{xf.full_scale_g}g')
        gaps = xf.gaps()
        print(f'gaps (> 2x período esperado): {len(gaps)}')
        for idx, dur in gaps[:10]:
            print(f'  amostra {idx}: gap de {dur*1000:.1f} ms')

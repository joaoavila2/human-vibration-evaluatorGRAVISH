"""Conversor standalone: le os arquivos .DAT gravados pelo Xiao nRF52840 Sense
(ver iso26315/xiao_parser.py para o formato binario) e escreve um .txt legivel
ao lado de cada um.

Uso:
    - Arraste um ou mais arquivos .DAT (ou uma pasta) em cima do .exe, OU
    - Rode em linha de comando: xiao_to_txt.exe arquivo1.DAT [arquivo2.DAT ...]
    - Sem argumentos: abre uma janela pra escolher a pasta com os .DAT.

Independente do resto do software -- nao precisa do PySide6/Qt, so numpy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iso26315 import xiao_parser as xp


def converter_arquivo(caminho: Path) -> str:
    """Converte um .DAT em .txt ao lado dele. Retorna uma linha de resumo."""
    xf = xp.parse_file(caminho)
    saida = caminho.with_suffix('.txt')

    si = xf.to_si() if len(xf.t) else None
    gaps = xf.gaps()

    with open(saida, 'w', encoding='utf-8') as f:
        f.write(f'# arquivo_origem: {caminho.name}\n')
        f.write(f'# inicio: {xf.start.isoformat()}\n')
        f.write(f'# taxa_amostragem_hz: {xf.sample_rate_hz}\n')
        f.write(f'# fundo_de_escala_g: {xf.full_scale_g}\n')
        f.write(f'# n_amostras: {len(xf.t)}\n')
        f.write(f'# n_gaps (>2x periodo esperado): {len(gaps)}\n')
        for idx, dur in gaps:
            f.write(f'#   gap na amostra {idx}: {dur * 1000:.1f} ms\n')
        f.write('#\n')
        f.write('tempo_s\tx_m_s2\ty_m_s2\tz_m_s2\tx_raw\ty_raw\tz_raw\n')

        if si is not None:
            for i in range(len(xf.t)):
                f.write(
                    f'{xf.t[i]:.6f}\t{si[i, 0]:.6f}\t{si[i, 1]:.6f}\t{si[i, 2]:.6f}\t'
                    f'{xf.counts[i, 0]}\t{xf.counts[i, 1]}\t{xf.counts[i, 2]}\n'
                )

    return f'{caminho.name} -> {saida.name}  ({len(xf.t)} amostras, {len(gaps)} gap(s))'


def coletar_arquivos(alvos):
    """Expande argumentos (arquivos e/ou pastas) numa lista de .DAT, sem duplicar
    entradas em sistemas de arquivo case-insensitive (Windows)."""
    vistos = set()
    arquivos = []
    for alvo in alvos:
        p = Path(alvo)
        candidatos = sorted(p.glob('*.DAT')) + sorted(p.glob('*.dat')) if p.is_dir() else [p] if p.is_file() else []
        for c in candidatos:
            chave = c.resolve().as_posix().lower()
            if chave not in vistos:
                vistos.add(chave)
                arquivos.append(c)
    return arquivos


def rodar_com_log(alvos, log):
    arquivos = coletar_arquivos(alvos)
    if not arquivos:
        log('Nenhum arquivo .DAT encontrado.')
        return

    log(f'{len(arquivos)} arquivo(s) encontrado(s). Convertendo...\n')
    for caminho in arquivos:
        try:
            log(converter_arquivo(caminho))
        except Exception as e:
            log(f'ERRO em {caminho.name}: {e}')
    log('\nConcluido.')


def _modo_grafico():
    import tkinter as tk
    from tkinter import filedialog, scrolledtext

    root = tk.Tk()
    root.title('Xiao -> TXT')
    root.geometry('560x420')

    texto = scrolledtext.ScrolledText(root, wrap='word')
    texto.pack(fill='both', expand=True, padx=8, pady=8)

    def log(msg):
        texto.insert('end', msg + '\n')
        texto.see('end')
        root.update_idletasks()

    def escolher_pasta():
        pasta = filedialog.askdirectory(title='Selecione a pasta com os arquivos .DAT')
        if pasta:
            texto.delete('1.0', 'end')
            rodar_com_log([pasta], log)

    def escolher_arquivos():
        arquivos = filedialog.askopenfilenames(
            title='Selecione um ou mais arquivos .DAT',
            filetypes=[('Xiao data', '*.DAT *.dat'), ('Todos', '*.*')],
        )
        if arquivos:
            texto.delete('1.0', 'end')
            rodar_com_log(list(arquivos), log)

    botoes = tk.Frame(root)
    botoes.pack(fill='x', padx=8, pady=(8, 0))
    tk.Button(botoes, text='Selecionar pasta...', command=escolher_pasta).pack(side='left')
    tk.Button(botoes, text='Selecionar arquivo(s)...', command=escolher_arquivos).pack(side='left', padx=8)

    log('Selecione uma pasta ou arquivo(s) .DAT pra converter em .txt.')
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # chamado por linha de comando ou por arrastar-e-soltar no .exe
        rodar_com_log(sys.argv[1:], print)
    else:
        _modo_grafico()

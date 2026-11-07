# -*- mode: python ; coding: utf-8 -*-
# Build:  pyinstaller iso2631-5.spec
# Gera um executavel unico com os dados de funcao de transferencia embutidos.

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data', 'data'),
        ('iso26315/assets', 'assets'),
        (r'C:\Users\User\AppData\Local\Programs\Python\Python314\Lib\site-packages\scienceplots',
         'scienceplots'),
    ],
    hiddenimports=['xlrd', 'openpyxl', 'serial', 'serial.tools.list_ports',
                  'serial.tools.list_ports_windows',
                  'scipy.io', 'scipy.io.matlab', 'scienceplots'],
    excludes=['tkinter', 'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.Qt3D',
              'PySide6.QtWebEngineCore', 'PySide6.QtMultimedia',
              'PySide6.QtNetwork', 'PySide6.QtSql', 'PySide6.QtTest',
              'IPython', 'pytest', 'pandas', 'PIL.ImageQt'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='AvaliadorDeVibracaoHumana-v3.1.2',
    debug=False, strip=False, upx=True,
    runtime_tmpdir=None, console=False,
    icon='iso26315/assets/icon.ico',
)

# Segundo executavel: o construtor de planilhas. Compartilha a mesma analise,
# entao sai pequeno perto do primeiro.
b = Analysis(
    ['run_editor.py'],
    pathex=[], binaries=[], datas=[('iso26315/assets', 'assets')],
    hiddenimports=['openpyxl', 'xlrd'],
    excludes=a.excludes,
    noarchive=False,
)
pyz_b = PYZ(b.pure)
exe_editor = EXE(
    pyz_b, b.scripts, b.binaries, b.datas, [],
    name='AvaliadorDeVibracaoHumana-Planilhas',
    debug=False, strip=False, upx=True,
    runtime_tmpdir=None, console=False,
    icon='iso26315/assets/icon.ico',
)

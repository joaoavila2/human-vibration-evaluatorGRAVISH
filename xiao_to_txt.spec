# -*- mode: python ; coding: utf-8 -*-
# Build:  pyinstaller xiao_to_txt.spec
# Conversor standalone .DAT (Xiao nRF52840) -> .txt. Independente do app
# principal: nao usa PySide6/Qt, so numpy + tkinter (stdlib), entao sai bem menor.

a = Analysis(
    ['xiao_to_txt.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    excludes=['PySide6', 'shiboken6', 'matplotlib', 'scipy', 'pandas',
              'IPython', 'pytest', 'PIL'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='XiaoParaTxt',
    debug=False, strip=False, upx=True,
    runtime_tmpdir=None, console=False,
)

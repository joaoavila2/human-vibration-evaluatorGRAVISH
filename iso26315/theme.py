"""Paleta e folha de estilo compartilhadas pelos dois aplicativos.

As cores sao todas explicitas, de propósito: sem isso o Qt herda o tema do
sistema operacional e, em Windows no modo escuro, o texto claro do tema cai em
cima dos fundos claros da folha de estilo e some.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

ACCENT = '#1f6feb'
ACCENT_DARK = '#1a5fca'
OK_COLOR = '#1b7a33'
WARN_COLOR = '#b25c00'
BAD_COLOR = '#b3261e'

TEXT = '#1a1d21'          # texto principal
TEXT_MUTED = '#4a5157'    # notas e dicas: escuro o bastante para ler
BG = '#ffffff'
BG_ALT = '#f4f6f8'        # linhas alternadas, cabecalhos
BORDER = '#c2c8ce'
HIGHLIGHT = '#fdf3d2'     # linha do nivel critico

STYLE = f"""
QWidget {{
    font-size: 13px;
    color: {TEXT};
    background: {BG};
}}
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel#muted {{ color: {TEXT_MUTED}; background: transparent; }}
QLabel#warning {{
    background: #fdf3d2; border: 1px solid #e0cf96; border-radius: 5px;
    padding: 7px; color: {TEXT}; font-size: 12px;
}}
QLabel#verdict {{
    background: {BG_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px;
    font-size: 15px;
    color: {TEXT};
}}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {ACCENT};
    background: {BG};
}}
QPushButton {{
    background: {ACCENT};
    color: #ffffff;
    border: 1px solid {ACCENT_DARK};
    border-radius: 5px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {ACCENT_DARK}; }}
QPushButton:disabled {{
    background: #dfe3e7; color: #8d949b; border-color: {BORDER};
}}
QPushButton#secondary {{
    background: {BG_ALT}; color: {TEXT}; border: 1px solid {BORDER};
}}
QPushButton#secondary:hover {{ background: #e6eaee; }}
QLineEdit, QComboBox, QTextEdit {{
    background: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
/* QSpinBox e QDoubleSpinBox ficam deliberadamente fora da folha de estilo:
   qualquer regra de padding neles desloca as areas clicaveis das setas e
   elas param de responder. O Fusion mais a paleta explicita ja bastam.

   Os indicadores de QRadioButton e QCheckBox, ao contrario, sao desenhados
   aqui de forma explicita. Estilizar o widget sem estilizar o indicador faz
   o Qt trocar o desenho nativo pelo motor da folha de estilo, que sem
   imagem de indicador nao desenha nada. Circulo e quadrado feitos so com
   border-radius funcionam em qualquer plataforma, sem depender de imagens. */
QRadioButton::indicator {{
    width: 13px; height: 13px;
    border: 1px solid #7a828a;
    border-radius: 7px;
    background: {BG};
}}
QRadioButton::indicator:checked {{
    /* Gradiente radial em vez de borda larga: aumentar a espessura da borda
       estoura o border-radius e o circulo vira quadrado. */
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                fx:0.5, fy:0.5,
                                stop:0 {ACCENT}, stop:0.45 {ACCENT},
                                stop:0.5 {BG}, stop:1 {BG});
    border: 1px solid {ACCENT_DARK};
}}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid #7a828a;
    border-radius: 3px;
    background: {BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT_DARK};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: #eceff1; border-color: #bcc3ca;
}}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: #f0f2f4; color: #8d949b;
}}
QComboBox QAbstractItemView {{
    background: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QTableWidget {{
    background: {BG};
    alternate-background-color: {BG_ALT};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    selection-background-color: #cfe0fb;
    selection-color: {TEXT};
}}
QTableWidget::item {{ padding: 4px; color: {TEXT}; }}
QHeaderView::section {{
    background: #e7ebef;
    color: {TEXT};
    padding: 6px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; background: {BG}; }}
QTabBar::tab {{
    background: {BG_ALT};
    color: {TEXT_MUTED};
    padding: 8px 18px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {BG}; color: {ACCENT}; font-weight: 600;
}}
QCheckBox, QRadioButton {{ color: {TEXT}; background: transparent; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: #8d949b; }}
QProgressBar {{
    border: 1px solid {BORDER}; border-radius: 4px;
    background: {BG_ALT}; color: {TEXT}; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; }}
QStatusBar {{ background: {BG_ALT}; color: {TEXT}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {BG_ALT}; }}
QToolBar {{ background: {BG_ALT}; border: 1px solid {BORDER}; spacing: 2px; }}
QToolButton {{
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 3px;
}}
QToolButton:hover {{ background: #dce7f5; border-color: {BORDER}; }}
QToolButton:checked {{
    /* Pan e zoom do matplotlib sao botoes marcaveis; sem esta regra o
       estado ligado fica invisivel. */
    background: {ACCENT}; border-color: {ACCENT_DARK};
}}
QToolButton:checked:hover {{ background: {ACCENT_DARK}; }}
QToolTip {{
    background: #2b3036; color: #ffffff; border: none; padding: 4px;
}}
"""


def apply_theme(app):
    """Trava a paleta clara, independente do tema do sistema.

    O estilo Fusion e obrigatorio aqui: o estilo nativo do Windows ignora
    parte da paleta e, combinado com folha de estilo, deixa de desenhar os
    indicadores de QRadioButton e as setas de QSpinBox.
    """
    app.setStyle('Fusion')
    pal = QPalette()
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText,
                 QPalette.ToolTipText, QPalette.PlaceholderText):
        pal.setColor(role, QColor(TEXT))
    for role in (QPalette.Window, QPalette.Base, QPalette.Button,
                 QPalette.ToolTipBase):
        pal.setColor(role, QColor(BG))
    pal.setColor(QPalette.AlternateBase, QColor(BG_ALT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor('#8d949b'))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#8d949b'))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#8d949b'))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)

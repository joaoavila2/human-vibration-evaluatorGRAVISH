"""Protótipo experimental (v2.9): postura de cadeira de rodas para a
Clausula 5/Anexo C.

NAO faz parte da cadeia validada de annexc.py, e NAO deve ser usado para
laudo. E' um teste de engenharia para explorar como a postura de um usuario
de cadeira de rodas poderia entrar na Clausula 5, que na norma so' tem UMA
funcao de transferencia assento-coluna (Formula 1, zero + 6 polos), sem
parametro de postura -- diferente do Anexo A, que varre 7 grupos de postura.

Modelo-base: Kawai & Matsuoka (SAE 2000-01-0645) / Matsuoka, Kawai & Sato
(ASME J. Mech. Design 125:779-785, 2003) -- cadeia articulada de 5 segmentos
(canela/coxa/tronco inferior/tronco superior/cabeca) sobre uma cadeira de
rodas modelada como corpo rigido (2 GDL: translacao vertical q, rotacao
theta9), com 4 pontos de contato mola-amortecedor medidos por ensaio modal
real (assento->coluna, o analogo direto da Formula 1 da norma, mas
especifico de cadeira de rodas).

Duas correcoes aplicadas sobre os valores publicados, cada uma com
justificativa propria (nao sao ajuste cego):

1. KtD (rigidez rotacional lombar) = 46,9 N.m/rad, nao 6,9 como aparece no
   SAE2000/ASME2003 -- confirmado por Hatano & Takahashi (2016, J. Phys.
   Conf. Ser. 744:012221), mesma universidade (Keio) do Matsuoka, que
   reproduz a tabela original e diverge so' nesse valor (e no expoente de
   c3, que nao afeta frequencia natural). Testado numericamente: reduz o
   erro contra as 4 frequencias medidas (Tabela 1 dos artigos originais).

2. Iwc (momento de inercia da cadeira) foi medido "considerando a cadeira
   como um pendulo" (texto original) -- um ensaio de pendulo mede a inercia
   em torno do EIXO DE SUSPENSAO, nao do centro de massa. Corrigido pelo
   teorema dos eixos paralelos: Iwc_cg = Iwc_dado - mwc*l9^2.

Simplificacao: a canela (segmento 1) e' tratada como rigidamente presa a
cadeira (sem grau de liberdade rotacional proprio). Isso NAO e' uma
premissa arbitraria -- e' o resultado de uma investigacao numerica: com a
canela livre, ela forma um modo quase dissociado do resto do sistema (a
canela balanca sozinha contra uma mola de joelho fraca), abaixo de 1 Hz,
que nao aparece nos 4 modos que os artigos originais reportam como medidos
-- e a canela nao e' relevante para a resposta do tronco/lombar que o
Anexo C precisa. Reduz o sistema de 7 para 6 graus de liberdade
{q, theta2 (coxa), theta3 (tronco inferior), theta4 (tronco superior),
theta5 (cabeca), theta9 (cadeira)}.

Validacao: NAO reproduz exatamente as 4 frequencias da Tabela 1 dos artigos
(2,51/4,63/7,10/8,31 Hz) -- os dados publicados, mesmo apos as duas
correcoes acima, nao foram suficientes para isso sem acesso aos dados
originais dos autores. Em vez disso, validado contra a faixa de ressonancia
reportada de forma consistente por tres fontes independentes (Chwalik-
Pilszyk et al. 2022; Hatano & Takahashi 2016: "a faixa de ressonancia da
aceleracao vertical do tronco e' de 4 a 6 Hz, a da cadeira e' de 2 a 6 Hz";
Lariviere et al. 2021, revisao sistematica): o modelo corrigido tem 3 modos
entre 2 e 8 Hz (3,4 / 3,6 / 5,0 Hz), com o mais alto proximo da faixa
dominante de 4-6 Hz relatada.

Postura: nao ha' literatura de cadeira de rodas com um fator numerico
calibrado para efeito de postura (Matsuoka et al. 2003 mostra so' o
grafico qualitativo de que o quadril deslizado para frente aumenta a
transmissibilidade, sem valor). `posture_softening` fica exposto como
parametro livre (padrao 1.0 = sem correcao) multiplicando a rigidez/
amortecimento do contato assento-quadril (k3, c3) -- para exploracao, nao
uma correcao validada.
"""
import numpy as np

# ----------------------------------------------------------------------
# Parametros do sujeito padrao (Tabela 2 de Kawai & Matsuoka, identica nas
# 3 fontes lidas). Segmentos: 1=canela (rigida com a cadeira, sem GDL
# proprio), 2=coxa, 3=tronco inferior, 4=tronco superior, 5=cabeca+pescoco.
MASS = {1: 8.00, 2: 14.2, 3: 12.1, 4: 25.8, 5: 4.50}          # kg
INERTIA = {2: 0.154, 3: 0.070, 4: 0.316, 5: 0.017}            # kg.m^2 (sobre o proprio CG)
LENGTH = {1: 0.458, 2: 0.419, 3: 0.191, 4: 0.282}             # m (comprimento do elo)
CG_OFFSET = {1: 0.215, 2: 0.220, 3: 0.075, 4: 0.191, 5: 0.168}  # m
ANGLE = {1: 1.08, 2: 6.06, 3: 1.42, 4: 1.46, 5: 1.50}          # rad

# Rigidez/amortecimento rotacional das juntas (Tabela 3). KtB (joelho) nao
# entra' mais: a canela nao tem GDL proprio nesta versao simplificada.
KT = {'C': 6.69, 'D': 46.9, 'E': 1.20}       # N.m/rad -- D corrigido (ver docstring)
CT = {'C': 3.84e-2, 'D': 2.67e-1, 'E': 6.84e-3}  # N.m.s/rad

# Cadeira (Tabela 4). Iwc corrigido pelo teorema dos eixos paralelos.
MASS_WC = 18.4                                 # kg
L_WC = {6: 0.385, 7: 0.405, '8a': 0.212, 9: 0.505}   # m
W_WC = {6: 0.870, 7: 6.220, 8: 1.470, 9: 0.371}      # rad
IWC_MEASURED = 8.074           # kg.m^2, medido por pendulo (em torno do pivo)
IWC_CG = IWC_MEASURED - MASS_WC * L_WC[9] ** 2        # corrigido para o CG

# Pontos de contato mola-amortecedor (Tabela 5, ajuste SAE2000 -- deu menor
# erro contra a Tabela 1 nos testes numericos que fizemos). k1,c1 = roda
# dianteira-solo (ponto F); k2,c2 = roda traseira-solo (ponto H);
# k3,c3 = quadril-assento (ponto C ao ponto G da cadeira);
# k4,c4 = tronco-encosto (ponto D ao ponto I da cadeira).
K_CONTACT = {1: 7.74e4, 2: 3.80e4, 3: 3.88e4, 4: 1.88e4}   # N/m
C_CONTACT = {1: 441.0, 2: 217.0, 3: 221.0, 4: 6.61}        # N.s/m -- c4 confirmado em 3 fontes

OUTPUT_SEGMENTS = {2: 'Coxa', 3: 'Tronco inferior (proximo a lombar)',
                   4: 'Tronco superior', 5: 'Cabeca'}

N_DOF = 6   # p = [q, theta2, theta3, theta4, theta5, theta9]
_IDX = {'q': 0, 2: 1, 3: 2, 4: 3, 5: 4, 9: 5}


def _d(w):
    return np.array([np.cos(w), np.sin(w)])


def _vec(d=None, **kw):
    v = np.zeros(N_DOF)
    if d:
        for k, val in d.items():
            v[_IDX[k]] = val
    for k, val in kw.items():
        v[_IDX[k]] = val
    return v


def _system_matrices(posture_softening=1.0):
    """M, C, K (6x6) e as posicoes (dy,dz) de cada ponto de interesse,
    como combinacoes lineares das coordenadas generalizadas -- ver
    docstring do modulo quanto a derivacao. posture_softening escala
    k3,c3 (contato quadril-assento)."""
    l1, w1 = LENGTH[1], ANGLE[1]
    l2, l2a, w2 = LENGTH[2], CG_OFFSET[2], ANGLE[2]
    l3, l3a, w3 = LENGTH[3], CG_OFFSET[3], ANGLE[3]
    l4, l4a, w4 = LENGTH[4], CG_OFFSET[4], ANGLE[4]
    l5a, w5 = CG_OFFSET[5], ANGLE[5]

    dzB, dyB = _vec(q=1), _vec()
    dz1, dy1 = _vec(q=1), _vec()          # canela: rigida com a cadeira

    dzC = dzB + _vec({2: l2 * np.cos(w2)}); dyC = dyB + _vec({2: -l2 * np.sin(w2)})
    dz2 = dzB + _vec({2: l2a * np.cos(w2)}); dy2 = dyB + _vec({2: -l2a * np.sin(w2)})

    dzD = dzC + _vec({3: l3 * np.cos(w3)}); dyD = dyC + _vec({3: -l3 * np.sin(w3)})
    dz3 = dzC + _vec({3: l3a * np.cos(w3)}); dy3 = dyC + _vec({3: -l3a * np.sin(w3)})

    dzE = dzD + _vec({4: l4 * np.cos(w4)}); dyE = dyD + _vec({4: -l4 * np.sin(w4)})
    dz4 = dzD + _vec({4: l4a * np.cos(w4)}); dy4 = dyD + _vec({4: -l4a * np.sin(w4)})

    dz5 = dzE + _vec({5: l5a * np.cos(w5)}); dy5 = dyE + _vec({5: -l5a * np.sin(w5)})

    w6, w7, w8, w9 = W_WC[6], W_WC[7], W_WC[8], W_WC[9]
    l6, l7, l8a, l9 = L_WC[6], L_WC[7], L_WC['8a'], L_WC[9]
    dzF = _vec(q=1, d={9: l6 * np.cos(w6)})
    dyF = _vec({9: -l6 * np.sin(w6)})
    dzH = _vec(q=1, d={9: l6 * np.cos(w6) + l7 * np.cos(w7)})
    dyH = _vec({9: -(l6 * np.sin(w6) + l7 * np.sin(w7))})
    dzI = _vec(q=1, d={9: l6 * np.cos(w6) + l7 * np.cos(w7) + l8a * np.cos(w8)})
    dyI = _vec({9: -(l6 * np.sin(w6) + l7 * np.sin(w7) + l8a * np.sin(w8))})
    dzG = _vec(q=1, d={9: l9 * np.cos(w9)})
    dyG = _vec({9: -l9 * np.sin(w9)})

    M = np.zeros((N_DOF, N_DOF))
    for dy, dz, m in [(dy1, dz1, MASS[1]), (dy2, dz2, MASS[2]), (dy3, dz3, MASS[3]),
                     (dy4, dz4, MASS[4]), (dy5, dz5, MASS[5]), (dyG, dzG, MASS_WC)]:
        M += m * (np.outer(dy, dy) + np.outer(dz, dz))
    for key, I in [(2, INERTIA[2]), (3, INERTIA[3]), (4, INERTIA[4]), (5, INERTIA[5])]:
        M[_IDX[key], _IDX[key]] += I
    M[_IDX[9], _IDX[9]] += IWC_CG

    def joint(a, b, Kt):
        e = np.zeros(N_DOF); e[_IDX[a]] = -1; e[_IDX[b]] = 1
        return Kt * np.outer(e, e)

    K = joint(2, 3, KT['C']) + joint(3, 4, KT['D']) + joint(4, 5, KT['E'])
    C = joint(2, 3, CT['C']) + joint(3, 4, CT['D']) + joint(4, 5, CT['E'])

    k3 = K_CONTACT[3] * posture_softening
    c3 = C_CONTACT[3] * posture_softening
    dCG_z = dzC - dzG
    dDI_z = dzD - dzI
    K += K_CONTACT[1] * np.outer(dzF, dzF) + K_CONTACT[2] * np.outer(dzH, dzH)
    K += k3 * np.outer(dCG_z, dCG_z) + K_CONTACT[4] * np.outer(dDI_z, dDI_z)
    C += C_CONTACT[1] * np.outer(dzF, dzF) + C_CONTACT[2] * np.outer(dzH, dzH)
    C += c3 * np.outer(dCG_z, dCG_z) + C_CONTACT[4] * np.outer(dDI_z, dDI_z)

    outputs = {2: (dy2, dz2), 3: (dy3, dz3), 4: (dy4, dz4), 5: (dy5, dz5)}
    return M, C, K, outputs, dzF, dzH


def natural_frequencies(posture_softening=1.0):
    """Frequencias naturais nao amortecidas (Hz) -- so' para conferencia,
    nao entra na cadeia de calculo."""
    M, _, K, _, _, _ = _system_matrices(posture_softening)
    from scipy.linalg import eigh
    w2, _ = eigh(K, M)
    return np.sort(np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi))


def transfer_function(freq, output=3, posture_softening=1.0):
    """H(omega) = aceleracao vertical do segmento `output` (2=coxa,
    3=tronco inferior/perto da lombar, 4=tronco superior, 5=cabeca) por
    aceleracao do assento -- mesma interface de annexc.transfer_function,
    para substituir a Formula (1) em spine_response_wheelchair. Default
    output=3 (tronco inferior): o segmento mais proximo da lombar, o
    analogo mais direto do que a Formula (1) representa."""
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    M, C, K, outputs, dzF, dzH = _system_matrices(posture_softening)
    dy_out, dz_out = outputs[output]
    H = np.zeros(len(freq), dtype=complex)
    for i, f in enumerate(freq):
        w = 2 * np.pi * f
        if w == 0:
            H[i] = 1.0
            continue
        A = -w ** 2 * M + 1j * w * C + K
        # excitacao pela base (assento/piso), mesmo sinal z0 nos dois pontos
        # de contato com o solo (F e H) -- ver nota do modulo.
        F = ((1j * w * C_CONTACT[1] + K_CONTACT[1]) * dzF +
             (1j * w * C_CONTACT[2] + K_CONTACT[2]) * dzH).astype(complex)
        Z = np.linalg.solve(A, F)
        H[i] = dz_out @ Z
    return H


def spine_response_wheelchair(az, dt, output=3, posture_softening=1.0):
    """Como annexc.spine_response, mas com a funcao de transferencia deste
    modulo em vez da Formula (1)."""
    n = len(az)
    A = np.fft.rfft(az)
    f = np.fft.rfftfreq(n, dt)
    return np.fft.irfft(A * transfer_function(f, output, posture_softening), n)

"""Testes unitarios de core.anthropometry() nas fronteiras das classes de massa.

Bordas DIN SPEC 45697 / ISO/CD 2631-5:2014 (Schust et al. 2015, Tab. 1):
  IMC <= 26,1:  <65 | 65-72 | 72-80 | 80-88 | >=88 kg  (classes 1..5)
  IMC >  26,1:  <80 | 80-87 | 87-96 | 96-109 | >=109 kg (classes 1..5)

Convencao: o valor NA fronteira pertence a classe SUPERIOR (side='right').
  Ex.: 65 kg => classe 2 (nao classe 1); 88 kg => classe 5.

Alturas usadas:
  h=1.90 m => IMC(70kg)=19.4 <= 26,1  (classe de IMC 1)
  h=1.60 m => IMC(70kg)=27.3 >  26,1  (classe de IMC 2)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iso26315.core import anthropometry

H_LOW  = 1.90   # garante IMC <= 26,1 para as massas testadas (ate ~94 kg)
H_HIGH = 1.60   # garante IMC >  26,1 para as massas testadas (a partir ~67 kg)


def check(mass, height, expected_percentile_class, expected_bmi_class):
    k, bmi_per = anthropometry(mass, height)
    bmi_class_got = (bmi_per - 1) // 5 + 1
    perc_class_got = (bmi_per - 1) % 5 + 1
    assert perc_class_got == expected_percentile_class, (
        f"mass={mass} h={height}: esperado perc-class {expected_percentile_class}, "
        f"obtido {perc_class_got} (bmi_per={bmi_per})"
    )
    assert bmi_class_got == expected_bmi_class, (
        f"mass={mass} h={height}: esperado bmi-class {expected_bmi_class}, "
        f"obtido {bmi_class_got}"
    )


def test_bmi_class1_boundaries():
    # IMC <= 26,1: bordas em 65 / 72 / 80 / 88 kg
    check(64.9, H_LOW, 1, 1)   # abaixo de 65 -> classe 1
    check(65.0, H_LOW, 2, 1)   # na fronteira 65 -> classe 2 (side='right')
    check(65.1, H_LOW, 2, 1)
    check(71.9, H_LOW, 2, 1)
    check(72.0, H_LOW, 3, 1)   # na fronteira 72 -> classe 3
    check(72.1, H_LOW, 3, 1)
    check(79.9, H_LOW, 3, 1)
    check(80.0, H_LOW, 4, 1)   # na fronteira 80 -> classe 4
    check(80.1, H_LOW, 4, 1)
    check(87.9, H_LOW, 4, 1)
    check(88.0, H_LOW, 5, 1)   # na fronteira 88 -> classe 5
    check(88.1, H_LOW, 5, 1)
    check(93.0, H_LOW, 5, 1)   # 93 kg, IMC=25.8 <= 26.1 -> ainda classe IMC 1, massa-class 5


def test_bmi_class2_boundaries():
    # IMC > 26,1: bordas em 80 / 87 / 96 / 109 kg
    check(79.9, H_HIGH, 1, 2)
    check(80.0, H_HIGH, 2, 2)   # na fronteira 80 -> classe 2
    check(80.1, H_HIGH, 2, 2)
    check(86.9, H_HIGH, 2, 2)
    check(87.0, H_HIGH, 3, 2)   # na fronteira 87 -> classe 3
    check(87.1, H_HIGH, 3, 2)
    check(95.9, H_HIGH, 3, 2)
    check(96.0, H_HIGH, 4, 2)   # na fronteira 96 -> classe 4
    check(96.1, H_HIGH, 4, 2)
    check(108.9, H_HIGH, 4, 2)
    check(109.0, H_HIGH, 5, 2)  # na fronteira 109 -> classe 5 (texto da norma)
    check(109.1, H_HIGH, 5, 2)
    check(200.0, H_HIGH, 5, 2)


def test_reference_cases_unchanged():
    """Casos do pacote de exemplos (70/80/90/100/110 kg, h=1.85 m).

    Esses valores nao estao nas fronteiras, entao as classes devem ser identicas
    antes e depois da correcao das bordas. IMC de todos: 20.4..32.2 kg/m2.
    """
    h = 1.85
    # 70 kg, IMC=20.4 -> classe IMC 1, massa-class 1 (< 72, depois da correcao < 72)
    check(70.0, h, 2, 1)   # 65<=70<72 -> classe 2
    # 80 kg, IMC=23.4 -> classe IMC 1, 80 == fronteira -> classe 4
    check(80.0, h, 4, 1)
    # 90 kg, IMC=26.3 -> IMC>26.1 => classe IMC 2; 87<=90<96 -> massa-class 3
    check(90.0, h, 3, 2)
    # 100 kg, IMC=29.2 -> classe IMC 2; 96<=100<109 -> massa-class 4
    check(100.0, h, 4, 2)
    # 110 kg, IMC=32.1 -> classe IMC 2; 110>109 -> massa-class 5
    check(110.0, h, 5, 2)


if __name__ == '__main__':
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'OK  {t.__name__}')
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            traceback.print_exc()
            failed += 1
    print(f'\n{len(tests) - failed}/{len(tests)} passaram')
    raise SystemExit(failed)

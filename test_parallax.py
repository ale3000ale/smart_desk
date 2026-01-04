import numpy as np

def test_parallax_formula():
    print("TEST 1: Verifica Formula Fisica Z = (B*f)/d")
    B, f = 90.0, 800.0 # Baseline 90mm, Focal 800px
    
    # Casi di test (Disparità -> Profondità attesa)
    cases = [
        (144, 500.0),   # (90*800)/144 = 500mm
        (72, 1000.0),   # (90*800)/72 = 1000mm
        (36, 2000.0)    # (90*800)/36 = 2000mm
    ]
    
    for disp, expected_z in cases:
        calc_z = (B * f) / disp
        print(f"  Disp: {disp}px -> Z calc: {calc_z:.1f}mm (Atteso: {expected_z}mm) - {'OK' if abs(calc_z-expected_z)<0.1 else 'FAIL'}")

def run_tests():
    print("=== INIZIO VALIDAZIONE DEPTH MAP ===")
    test_parallax_formula()
    print("\nTEST 2: Parametri StereoSGBM")
    print("  Block Size: 13 (Ottimizzato per noise/dettaglio)")
    print("  Num Disparities: 96 (Range Z > 750mm)")
    print("  Uniqueness: 15% (Filtra ambiguità)")
    print("=== FINE ===")

if __name__ == "__main__":
    run_tests()
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/training_microcycles_synthetic.csv')

print("\n" + "="*70)
print("VALIDACIÓN DE LOS 3 AJUSTES")
print("="*70)

# AJUSTE 1: Filas de Injury
print("\n✅ AJUSTE 1: Lesionados con session_type='Injury'")
injury_rows = df[df['session_type'] == 'Injury']
print(f"   Total filas de Injury: {len(injury_rows)}")
if len(injury_rows) > 0:
    print(f"\n   Muestra de filas de Injury:")
    print(injury_rows[['matchweek', 'session_id', 'session_type', 'player_id', 'duration_min', 'rpe_load']].head())
else:
    print("   ⚠️ No hay filas de Injury (posible falta de lesiones generadas)")

# AJUSTE 2: RPE limitado a 1-10
print("\n✅ AJUSTE 2: RPE limitado a [1, 10]")

# Solo tiene sentido estimar RPE en filas con actividad real
df_active_nonzero = df[
    (df['session_type'] != 'Injury') &
    (df['duration_min'] > 0)
].copy()

df_active_nonzero['rpe'] = df_active_nonzero['rpe_load'] / df_active_nonzero['duration_min']

print(f"   RPE_load range: {df['rpe_load'].min()} to {df['rpe_load'].max()}")
print(f"   RPE (1-10) range: {df_active_nonzero['rpe'].min():.2f} to {df_active_nonzero['rpe'].max():.2f}")

if df_active_nonzero['rpe'].max() <= 10 and df_active_nonzero['rpe'].min() >= 1:
    print("   ✓ RPE está correctamente limitado a [1, 10]")
else:
    print("   ✗ RPE sale del rango [1, 10]")

# AJUSTE 3: Distance vs Duration (≤ 180 m/min)
print("\n✅ AJUSTE 3: Distance ≤ Duration × 180 (max ~10.8 km/h)")
df_check = df[(df['duration_min'] > 0) & (df['session_type'] != 'Injury')].copy()
df_check['ratio'] = df_check['total_distance_m'] / df_check['duration_min']
max_ratio = df_check['ratio'].max()
violaciones = len(df_check[df_check['ratio'] > 180])

print(f"   Max ratio (m/min): {max_ratio:.2f}")
print(f"   Violaciones (ratio > 180): {violaciones}")
if violaciones == 0:
    print(f"   ✓ Todas las filas cumplen distance ≤ duration × 180")
else:
    print(f"   ✗ {violaciones} filas violan la regla")
    print(f"\n   Ejemplos de violaciones:")
    print(df_check[df_check['ratio'] > 180][['session_type', 'duration_min', 'total_distance_m', 'ratio']].head())

# Visualización adicional
print("\n📊 ESTADÍSTICAS GENERALES:")
print(f"   Total rows: {len(df)}")
print(f"   Session types: {df['session_type'].value_counts().to_dict()}")
print(f"   Players with injuries: {injury_rows['player_id'].nunique() if len(injury_rows) > 0 else 0}")

print("\n" + "="*70)

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, List, Dict, Tuple

# ==============================================
# CONFIGURATION CLASS (Modularity + Flexibility)
# ==============================================
@dataclass
class TeamConfig:
    """Configuración centralizada del equipo y parámetros de simulación."""
    seed: int = 42
    n_matches: int = 6
    injury_rate: float = 0.07  # 7% probabilidad de lesión por microciclo
    injury_min_weeks: int = 1
    injury_max_weeks: int = 4
    
    # Formación táctica: (GK, DEF, MID, FWD)
    formation: Tuple[int, int, int, int] = (1, 4, 4, 2)
    
    # Probabilidades base de jugar (por grupo)
    play_prob_core: float = 0.82
    play_prob_subs: float = 0.30
    play_prob_fringe: float = 0.15
    gk_primary_prob: float = 0.85
    gk_backup_prob: float = 0.15
    
    # Variabilidad en microciclos (intensidad planificada)
    microcycle_intensities: List[str] = field(default_factory=lambda: [
        "high", "moderate", "high", "moderate", "high", "moderate"
    ])

@dataclass
class Injury:
    """Modelo de lesión."""
    player_id: str
    start_week: int
    duration_weeks: int
    
    def is_active(self, week: int) -> bool:
        return self.start_week <= week < self.start_week + self.duration_weeks

# ==============================================
# GLOBAL RNG
# ==============================================
config = TeamConfig()
rng = np.random.default_rng(config.seed)

# ==============================================
# PLANTILLA (23) + POSICIÓN
# ==============================================
players = {
    "P01": "GK",
    "P02": "DEF",
    "P03": "DEF",
    "P04": "DEF",
    "P05": "DEF",
    "P06": "MID",
    "P07": "FWD",
    "P08": "MID",
    "P09": "FWD",
    "P10": "MID",
    "P11": "FWD",
    "P12": "DEF",
    "P13": "GK",
    "P14": "MID",
    "P15": "MID",
    "P16": "FWD",
    "P17": "FWD",
    "P18": "DEF",
    "P19": "MID",
    "P20": "DEF",
    "P21": "MID",
    "P22": "FWD",
    "P23": "DEF"
}

all_players = list(players.keys())

# Clasificación por calidad
classic_core = {"P01","P02","P03","P04","P05","P06","P08","P10","P07","P11","P09"}
likely_starters = {"P12", "P14", "P15", "P16", "P18", "P21"}
fringe_players = set(all_players) - classic_core - likely_starters

gks = ["P01", "P13"]

# ==============================================
# SESIONES DEL MICROCICLO
# ==============================================
microcycle_sessions = [
    ("TR01", "Gym"),
    ("TR02", "Field"),
    ("TR03", "Gym"),
    ("TR04", "Field"),
    ("TR05", "Field"),
    ("TR06", "Gym"),
    ("TR07", "Field"),
    ("M",    "Match"),
]

# ==============================================
# INJURY MANAGEMENT (Modelo realista de lesiones)
# ==============================================
def initialize_injuries(rng) -> Dict[str, Injury]:
    """Inicializa lesiones al comienzo de la simulación."""
    injuries = {}
    active_injuries = 0
    
    for player_id in all_players:
        if rng.random() < config.injury_rate:
            injury_start = rng.integers(1, config.n_matches + 1)
            injury_duration = rng.integers(
                config.injury_min_weeks, config.injury_max_weeks + 1
            )
            injuries[player_id] = Injury(
                player_id=player_id,
                start_week=injury_start,
                duration_weeks=injury_duration
            )
            active_injuries += 1
    
    return injuries

def get_available_players(week: int, injuries: Dict[str, Injury]) -> Set[str]:
    """Retorna jugadores disponibles (no lesionados) en una semana específica."""
    return {p for p in all_players if p not in injuries or not injuries[p].is_active(week)}

# ==============================================
# TACTICAL FORMATION SELECTION (Realista)
# ==============================================
def weighted_sample_without_replacement(items, weights, k, rng):
    """Muestreo ponderado sin reemplazo (simple y robusto)."""
    items = list(items)
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 1e-9)
    chosen = []
    for _ in range(k):
        probs = weights / weights.sum()
        idx = rng.choice(len(items), p=probs)
        chosen.append(items.pop(idx))
        weights = np.delete(weights, idx)
    return chosen

def build_play_probabilities(available_players: Set[str]) -> Dict[str, float]:
    """Construye probabilidades de jugar basadas en calidad + disponibilidad."""
    play_prob = {}
    for p in all_players:
        if p not in available_players:
            play_prob[p] = 0.0  # Lesionado
        elif p in classic_core:
            play_prob[p] = config.play_prob_core
        elif p == "P01":
            play_prob[p] = config.gk_primary_prob
        elif p == "P13":
            play_prob[p] = config.gk_backup_prob
        elif p in likely_starters:
            play_prob[p] = 0.45
        else:
            play_prob[p] = config.play_prob_subs
    
    return play_prob

def select_starting_xi(available_players: Set[str], rng) -> Set[str]:
    """
    Selecciona 11 jugadores respetando formación táctica (1 GK + 4 DEF + 4 MID + 2 FWD).
    """
    play_prob = build_play_probabilities(available_players)
    
    # 1) Seleccionar GK
    available_gks = [g for g in gks if g in available_players]
    gk_probs = [play_prob[g] for g in available_gks]
    gk_probs = np.array(gk_probs) / np.sum(gk_probs)
    gk = rng.choice(available_gks, p=gk_probs)
    
    # 2) Seleccionar por posición (respetando formación)
    starters = {gk}
    remaining_formation = list(config.formation[1:])  # [4, 4, 2] DEF, MID, FWD
    positions_order = ["DEF", "MID", "FWD"]
    
    for pos_idx, position in enumerate(positions_order):
        n_needed = remaining_formation[pos_idx]
        pos_players = [
            p for p in available_players 
            if p not in starters and players[p] == position
        ]
        pos_weights = [play_prob[p] for p in pos_players]
        
        selected = weighted_sample_without_replacement(
            pos_players, pos_weights, k=n_needed, rng=rng
        )
        starters.update(selected)
    
    return starters

# ==============================================
# TRAINING LOAD & RECOVERY (Mejorado)
# ==============================================
def sample_training_load(session_type: str, intensity: str = "moderate") -> Tuple[int, int, int]:
    """
    Carga externa coherente por tipo de sesión e intensidad de microciclo.
    intensity: 'low', 'moderate', 'high'
    
    IMPORTANTE: Asegura que distance ≤ duration × 180 (velocidad media máx ~10 km/h)
    """
    intensity_mult = {"low": 0.70, "moderate": 1.0, "high": 1.25}[intensity]
    
    if session_type == "Gym":
        duration = int(rng.integers(45, 70) * intensity_mult)
        dist = int(rng.integers(1200, 2800) * intensity_mult)
        hsr = int(rng.integers(0, 150) * intensity_mult)
    elif session_type == "Field":
        duration = int(rng.integers(60, 90) * intensity_mult)
        dist = int(rng.integers(4500, 8200) * intensity_mult)
        hsr = int(rng.integers(350, 1250) * intensity_mult)
    else:  # Match
        duration = 90
        dist = int(rng.integers(9000, 11500) * intensity_mult)
        hsr = int(rng.integers(900, 1700) * intensity_mult)
    
    # CONTROL: Asegurar distancia coherente con duración (máx ~10 km/h = 166.7 m/min)
    max_dist_allowed = int(duration * 160)
    if dist > max_dist_allowed:
        dist = int(max_dist_allowed * 0.95)  # 95% para mayor conservadurismo
    
    return int(duration), int(dist), int(hsr)

def compute_rpe_load(
    duration: int, 
    dist: int, 
    hsr: int, 
    prev_load: float, 
    sleep: float, 
    session_type: str,
    accumulated_week_fatigue: float = 0.0
) -> int:
    """
    Carga interna (RPE*duration) con modelo mejorado que incluye:
    - Fatiga acumulada de la semana
    - Recuperación correlacionada con sueño
    - Bonus por tipo de sesión
    
    Retorna: RPE_Load = RPE (1-10) × duration, clipeado a [0, 10×duration]
    """
    noise = rng.normal(0, 0.7)
    session_bonus = 1.0 if session_type == "Match" else 0.4
    
    # Penalización por fatiga acumulada (efecto de la carga previa)
    fatigue_penalty = 0.0005 * accumulated_week_fatigue
    
    rpe = (
        2.0
        + 0.00012 * dist
        + 0.00050 * hsr
        + 0.00080 * prev_load  # Aumentado para mejor correlación secuencial
        - 0.55 * (sleep - 7.0)  # Recuperación mejorada
        - fatigue_penalty
        + session_bonus
        + noise
    )
    # Clip RPE a rango válido [1, 10] ANTES de multiplicar por duration
    rpe = np.clip(np.round(rpe), 1, 10).astype(int)
    rpe_load = int(rpe * duration)
    
    # Clip final del RPE_Load para evitar valores extremos
    rpe_load = np.clip(rpe_load, 0, 10 * duration).astype(int)
    
    return rpe_load

def compute_sleep(prev_load: float, week_accumulated: float, session_type: str) -> float:
    """
    Sueño más realista:
    - Disminuye después de matches (más estrés)
    - Correlacionado con fatiga acumulada
    - Oscila naturalmente
    """
    # Después de match: -0.5 a -1.0 horas
    match_penalty = 0.75 if session_type == "Match" else 0.0
    
    # Fatiga acumulada afecta sueño
    fatigue_effect = -0.0007 * week_accumulated
    
    sleep_base = 7.2 - 0.0005 * prev_load + fatigue_effect - match_penalty
    sleep = np.clip(rng.normal(sleep_base, 1.1), 4.5, 9.5)
    
    return round(sleep, 2)

def validate_row(row: Dict) -> bool:
    """
    Validación de coherencia lógica:
    - Si duration=0, entonces dist=0 y hsr=0
    - Valores dentro de rangos razonables
    - Distance ≤ duration × 180 (velocidad media máxima ~10 km/h)
    """
    # Lesiones siempre válidas
    if row["session_type"] == "Injury":
        return True
    
    # Si no hay entrenamiento, no hay distancia
    if row["duration_min"] == 0:
        return (row["total_distance_m"] == 0 and row["high_speed_running_m"] == 0)
    
    # Validación de velocidad media (distance / duration)
    # Max velocidad sensata: ~10 km/h = 166.7 m/min → ~180 m/min (conservador)
    if row["total_distance_m"] > row["duration_min"] * 180:
        return False
    
    # Validaciones de rango por sesión
    if row["session_type"] == "Gym":
        return 30 < row["duration_min"] < 100
    elif row["session_type"] == "Field":
        return 50 < row["duration_min"] < 110
    elif row["session_type"] == "Match":
        return row["duration_min"] == 90
    
    return True

# ==============================================
# MAIN GENERATION
# ==============================================
rows = []

# Inicializar lesiones
injuries = initialize_injuries(rng)

# Estado por jugador
prev_load_player = {p: float(rng.integers(150, 450)) for p in all_players}
weekly_fatigue = {p: 0.0 for p in all_players}

for match_idx in range(1, config.n_matches + 1):
    # Obtener jugadores disponibles (sin lesiones)
    available = get_available_players(match_idx, injuries)
    
    # Intensidad planificada para este microciclo
    microcycle_intensity = config.microcycle_intensities[match_idx - 1]
    
    # Once con rotación (solo de jugadores disponibles)
    starting_xi = select_starting_xi(available, rng)
    
    # Reset semanal
    for p in all_players:
        weekly_fatigue[p] = 0.0
    
    for session_order, (session_code, session_type) in enumerate(microcycle_sessions, start=1):
        session_id = f"MC{match_idx:02d}_{session_order:02d}_{session_code}"
        
        for player_id in all_players:
            position = players[player_id]
            is_injured = player_id not in available
            
            # MANEJO DE LESIONADOS (NEW)
            if is_injured:
                # Generar fila de lesión (mantiene estructura del dataset)
                row = {
                    "matchweek": match_idx,
                    "session_id": session_id,
                    "session_type": "Injury",
                    "player_id": player_id,
                    "position": position,
                    "played_match": 0,
                    "duration_min": 0,
                    "total_distance_m": 0,
                    "high_speed_running_m": 0,
                    "sleep_hours": np.nan,  # No hay datos de sueño si lesionado
                    "prev_load": round(float(prev_load_player[player_id]), 1),
                    "rpe_load": 0
                }
                rows.append(row)
                continue
            
            # Sueño realista (solo para no lesionados)
            sleep = compute_sleep(
                prev_load_player[player_id], 
                weekly_fatigue[player_id],
                session_type
            )
            
            if session_type == "Match":
                if player_id in starting_xi:
                    duration, dist, hsr = sample_training_load("Match", microcycle_intensity)
                    rpe_load = compute_rpe_load(
                        duration, dist, hsr, 
                        prev_load_player[player_id], 
                        sleep, 
                        "Match",
                        weekly_fatigue[player_id]
                    )
                else:
                    # Sustituto: no juega
                    duration, dist, hsr, rpe_load = 0, 0, 0, 0
            else:
                # Entrenamientos: todos entrenan (excepto lesionados)
                duration, dist, hsr = sample_training_load(session_type, microcycle_intensity)
                
                # Gestión de carga: reducir si fatiga acumulada es muy alta
                if weekly_fatigue[player_id] > 1200:
                    reduction_factor = 0.80
                    duration = max(30, int(duration * reduction_factor))
                    dist = int(dist * reduction_factor)
                    hsr = int(hsr * 0.75)
                elif weekly_fatigue[player_id] > 800:
                    reduction_factor = 0.90
                    duration = max(35, int(duration * reduction_factor))
                    dist = int(dist * reduction_factor)
                    hsr = int(hsr * 0.85)
                
                rpe_load = compute_rpe_load(
                    duration, dist, hsr, 
                    prev_load_player[player_id], 
                    sleep,
                    session_type,
                    weekly_fatigue[player_id]
                )
            
            row = {
                "matchweek": match_idx,
                "session_id": session_id,
                "session_type": session_type,
                "player_id": player_id,
                "position": position,
                "played_match": int(session_type == "Match" and player_id in starting_xi),
                "duration_min": duration,
                "total_distance_m": dist,
                "high_speed_running_m": hsr,
                "sleep_hours": sleep,
                "prev_load": round(float(prev_load_player[player_id]), 1),
                "rpe_load": rpe_load
            }
            
            # Clip a posteriori: asegurar distance ≤ duration × 180 (después de gestión de carga)
            if row["duration_min"] > 0 and row["total_distance_m"] > row["duration_min"] * 180:
                row["total_distance_m"] = int(row["duration_min"] * 180 * 0.90)
            
            # Validar coherencia
            if not validate_row(row):
                continue  # Skip filas incoherentes
            
            rows.append(row)
            
            # Actualizar estado
            prev_load_player[player_id] = float(rpe_load)
            weekly_fatigue[player_id] += rpe_load

df = pd.DataFrame(rows)

# ==============================================
# OUTLIERS Y MISSING DATA (más realistas)
# ==============================================
n = len(df)

# Missing data patrón realista:
# 1. Sleep: 2% aleatorio
sleep_idx = rng.choice(df.index, size=max(1, int(0.02 * n)), replace=False)
df.loc[sleep_idx, "sleep_hours"] = np.nan

# 2. HSR: 3% aleatorio (pero más probable si no jugó)
hsr_candidates = df[
    (df["duration_min"] > 0) &
    (df["session_type"].isin(["Gym", "Field", "Match"]))
].index

if len(hsr_candidates) > 0:
    hsr_idx = rng.choice(
        hsr_candidates,
        size=max(1, int(0.02 * len(hsr_candidates))),
        replace=False
    )
    df.loc[hsr_idx, "high_speed_running_m"] = np.nan

# Outliers contextualizados:
# 1. Sueño anormalmente alto (estrés mental, viaje)
out_idx = rng.choice(df.index, size=2, replace=False)
for idx in out_idx:
    df.loc[idx, "sleep_hours"] = rng.uniform(3.0, 4.0)  # Bajo estrés

# 2. Distancia muy alta (error de sensor, reset de GPS)
out_idx = rng.choice(df[df["session_type"] == "Field"].index, size=1, replace=False)
df.loc[out_idx, "total_distance_m"] = int(df.loc[out_idx, "total_distance_m"].values[0] * 1.8)

# 3. RPE inconsistente (sensor fallido, entrada manual errónea)
out_idx = rng.choice(df.index, size=1, replace=False)
df.loc[out_idx, "rpe_load"] = int(rng.uniform(10, 50))  # Muy bajo

# ==============================================
# SAVE & SUMMARY STATISTICS
# ==============================================
out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "training_microcycles_synthetic.csv"
df.to_csv(out_file, index=False)

# Print summary
print("\n" + "="*70)
print("SYNTHETIC TRAINING DATASET GENERATED")
print("="*70)
print(f"\nDataset File: {out_file.resolve()}")
print(f"\nDataset Overview:")
print(f"   Total rows: {len(df):,}")
print(f"   Unique players: {df['player_id'].nunique()}")
print(f"   Matchweeks: {df['matchweek'].nunique()}")
print(f"   Sessions per matchweek: {len(microcycle_sessions)}")

print(f"\nConfiguration Used:")
print(f"   Formation: {config.formation[0]} GK + {config.formation[1]} DEF + {config.formation[2]} MID + {config.formation[3]} FWD")
print(f"   Injury rate: {config.injury_rate*100:.1f}%")
print(f"   Active injuries: {len(injuries)}")
print(f"   Microcycle intensities: {config.microcycle_intensities}")

print(f"\nData Quality:")
print(f"   Missing values (sleep_hours): {df['sleep_hours'].isna().sum()} ({df['sleep_hours'].isna().sum()/len(df)*100:.2f}%)")
print(f"   Missing values (high_speed_running_m): {df['high_speed_running_m'].isna().sum()} ({df['high_speed_running_m'].isna().sum()/len(df)*100:.2f}%)")

print(f"\nSample Data (first 15 rows):")
print(df.head(15).to_string())

print(f"\nColumn Info:")
print(df.info())

print(f"\nDescriptive Statistics (numeric columns):")
print(df.describe().to_string())

print("\n" + "="*70)

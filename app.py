import streamlit as st
import pandas as pd
import re
import unicodedata
from io import StringIO
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="NBA AI Chatbot", page_icon="🏀", layout="wide")

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('nba_cleaned_final.csv')
        df = df.fillna(0)
        
        # Conversión de Tipos
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(int)
        df['Player'] = df['Player'].astype(str).str.strip()
        df['Team'] = df['Team'].astype(str)
        df['Pos'] = df['Pos'].astype(str)
        
        if 'Colleges' in df.columns: 
            df['Colleges'] = df['Colleges'].astype(str).fillna("Unknown")
        else: 
            df['Colleges'] = "Unknown"
            
        # Identificador único
        df['Player_ID'] = df['Player'] + " (" + df['Colleges'].astype(str) + ")"
        
        # Limpieza base
        df = df[df['Team'] != 'TOT']
        # Eliminar duplicados exactos
        df = df.drop_duplicates(subset=['Player', 'Year', 'Team'])

        # Asegurar columnas numéricas
        numeric_cols = ['G', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'MVP', 'W', 'L']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Normalización de Totales vs Promedios
        is_avg = df['PTS'].max() < 100 
        stats_base = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA']
        
        for col in stats_base:
            if col in df.columns and 'G' in df.columns:
                if is_avg:
                    df[f'{col}_Total'] = df[col] * df['G'] # Crear Total
                else:
                    df[f'{col}_Total'] = df[col] # Ya es Total
                    df[col] = df[col] / df['G'].replace(0, 1) # Crear Promedio
        
        # Tiros de 2
        df['2P_Total'] = df['FG_Total'] - df['3P_Total']
        
        # Métricas de Equipo
        if 'team_PTS_per_game' not in df.columns and 'PS/G' in df.columns:
             df['team_PTS_per_game'] = df['PS/G']
             
        return df
    except Exception as e:
        st.error(f"Error crítico cargando datos: {e}")
        return pd.DataFrame()

df = load_data()

# --- 2. MODELO DE INTENCIÓN ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

INTENT_PROTOTYPES = {
    "HISTORY_LEADER": [
        "quien es el maximo anotador de la historia", 
        "jugador con mas mvps de la historia",
        "top asistentes de siempre"
    ],
    "YEARLY_LEADER": [
        "quien anoto mas puntos en 2010", 
        "mvp del año 2000",
        "top 5 jugadores mas anotadores de 2010"
    ],
    "DECADE_LEADER": [
        "alero con mas triples en la decada de los 90", 
        "maximo anotador decada 80"
    ],
    "COMPARE_PLAYERS": [
        "comparar lebron james y pau gasol",
        "quien tiene mejor porcentaje",
        "quien anoto mas puntos lebron o kobe"
    ],
    "COMPARE_TEAMS": [
        "quien gano mas partidos lakers o celtics", 
        "comparar victorias"
    ],
    "PLAYER_STATS": [
        "estadisticas de la carrera de lebron james", 
        "cuantos puntos hizo curry en 2010",
        "asistencias de pau gasol 2010",
        "promedio de puntos de lebron"
    ],
    "TEAM_STATS": [
        "puntos de boston celtics en 2010", 
        "plantilla de lakers",
        "jugadores de cleveland"
    ],
    "TEAM_RANKING": [
        "top 5 equipos mas anotadores", 
        "equipo con mas derrotas"
    ],
    "COLLEGE_STATS": [
        "jugadores de duke", 
        "mejor jugador de north carolina"
    ],
    "COUNT_STATS": [
        "cuantos escoltas han ganado el mvp", 
        "numero de bases mvp"
    ],
    "PLOT_COMPARE": [
        "muestrame en una grafica la comparativa de puntos",
        "grafica de los triples tiros libres y tiros de dos de Luka",
        "muestrame una grafica de los puntos de kobe y lebron",
        "grafica promedio puntos marc y pau decada 2000"
    ],
    "PLOT_EVOLUTION": [
        "muestrame en una grafica la evolucion", 
        "grafica de asistencias totales entre decadas 80 90 y 2000",
        "promedio de puntos de los clippers historia",
        "grafica de victorias de los clippers"
    ]
}

intent_embeddings = {k: model.encode(v) for k, v in INTENT_PROTOTYPES.items()}

def predict_intent(query):
    query_embedding = model.encode(query)
    best_intent, max_score = None, -1
    for intent, prototypes in intent_embeddings.items():
        scores = util.cos_sim(query_embedding, prototypes)
        current_max = float(np.max(scores.numpy()))
        if current_max > max_score:
            max_score = current_max
            best_intent = intent
    return best_intent

# --- 3. UTILIDADES ---

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower().strip()) if unicodedata.category(c) != 'Mn')

def extract_year(query):
    years = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', query)
    return int(years[0]) if years else None

def extract_decades(query):
    matches = re.findall(r'(?:decada|los|anos)\s+(?:de\s+)?(?:los\s+)?(\d{2,4})', normalize(query))
    decades = []
    for m in matches:
        val = int(m)
        if val < 100: val = 1900 + val if val >= 40 else 2000 + val
        decades.append(val)
    return sorted(list(set(decades)))

def extract_number(query, default=1):
    q_norm = normalize(query)
    match_top = re.search(r'top\s+(\d+)', q_norm)
    if match_top: return int(match_top.group(1))
    if "top" in q_norm: return 5
    return default

def get_ordinal_index(query):
    mapping = {'segundo': 1, '2do': 1, 'tercer': 2, '3er': 2, 'cuarto': 3, 'quinto': 4}
    for k, v in mapping.items():
        if k in query.lower(): return v
    return 0 

def extract_position(query):
    q = normalize(query)
    if "ala pivot" in q: return "PF", "Ala-Pívots"
    if "base" in q: return "PG", "Bases"
    if "escolta" in q: return "SG", "Escoltas"
    if "alero" in q: return "SF", "Aleros"
    if "pivot" in q: return "C", "Pívots"
    return None, ""

def extract_college(query):
    q = normalize(query)
    colleges = ["duke", "north carolina", "unc", "kentucky", "ucla", "kansas", "georgetown", "wake forest"]
    for c in colleges:
        if c in q: return "North Carolina" if c == "unc" else c.title()
    return None

def extract_entities(query):
    q_norm = normalize(query)
    q_norm = q_norm.replace("washinton", "washington").replace("clipers", "clippers").replace("cleveland cavaliers", "cleveland cavaliers")
    
    found_teams, found_players = [], []
    
    # 1. Equipos
    all_teams = sorted(df['Team'].unique(), key=len, reverse=True)
    for t in all_teams:
        t_norm = normalize(t)
        if t_norm in q_norm or t_norm.split()[-1] in q_norm: 
             if t_norm in q_norm:
                found_teams.append(t)
                q_norm = q_norm.replace(t_norm, "")

    # 2. Jugadores
    blocked = ["base", "escolta", "alero", "pivot", "decada", "top", "mvp", "mejor", "puntos", "total", "carrera", "duke", "college", "promedio", "grafica", "grafico", "evolucion", "comparativa", "historia", "historico", "anotador", "asistente", "triplista", "reboteador", "equipo", "equipos", "jugadores"]
    
    all_players = df['Player'].unique()
    
    words = [w for w in q_norm.split() if len(w) > 3 and w not in blocked]
    potential_players = []
    
    for p in all_players:
        if normalize(p) in q_norm:
            potential_players.append(p)
            
    if not potential_players:
        for w in words:
            for p in all_players:
                p_norm = normalize(p)
                if w in p_norm.split():
                    potential_players.append(p)
                    
    found_players = list(set(potential_players))
    
    unique_teams = []
    for t in found_teams:
        if not any(t in other and t != other for other in found_teams):
            unique_teams.append(t)
            
    return unique_teams, found_players

def resolve_stat_columns(query):
    q_norm = normalize(query)
    stats = []
    mapping = {
        "asisten": ("AST", "Asistencias"),
        "rebot": ("TRB", "Rebotes"),
        "tripl": ("3P", "Triples"),
        "tiros de tres": ("3P", "Triples"),
        "tiros de 2": ("2P", "Tiros de 2"),
        "tiros de dos": ("2P", "Tiros de 2"),
        "tiros libre": ("FT", "Tiros Libres"),
        "libre": ("FT", "Tiros Libres"),
        "robo": ("STL", "Robos"),
        "tapon": ("BLK", "Tapones"),
        "victoria": ("W", "Victorias"),
        "gano": ("W", "Victorias"), 
        "derrota": ("L", "Derrotas"),
        "puntos": ("PTS", "Puntos"),
        "anota": ("PTS", "Puntos"),
        "porcentaje de tiro": ("FG", "Porcentaje TC"),
        "porcentaje": ("FG", "Porcentaje")
    }
    
    if "porcentaje" in q_norm or "efectividad" in q_norm:
        if "tres" in q_norm or "triple" in q_norm: return [("3P_PCT", "% Triples")]
        if "libre" in q_norm: return [("FT_PCT", "% Libres")]
        return [("FG_PCT", "% Tiro")]

    if "victoria" in q_norm or "gano" in q_norm: return [("W", "Victorias")]
    if "derrota" in q_norm: return [("L", "Derrotas")]

    for key, val in mapping.items():
        if key in q_norm:
            if not any(val[0] == s[0] for s in stats):
                stats.append(val)
    
    if not stats: stats.append(("PTS", "Puntos"))
    return stats

def is_average_requested(query):
    return any(x in normalize(query) for x in ['promedio', 'media', 'por partido', 'ppg'])

# --- 4. LÓGICA DE RESPUESTA ---

def get_answer(query):
    intent = predict_intent(query)
    year = extract_year(query)
    decades = extract_decades(query) 
    decade = decades[0] if decades else None
    
    teams, players = extract_entities(query)
    college = extract_college(query)
    pos_code, pos_name = extract_position(query)
    limit = extract_number(query, 1) 
    rank_idx = get_ordinal_index(query)
    stats_requested = resolve_stat_columns(query)
    use_avg = is_average_requested(query)
    q_norm = normalize(query)

    # --- REGLAS DE INTENCIÓN ---
    if "grafica" in q_norm or "grafico" in q_norm:
        if "evolucion" in q_norm or "historia" in q_norm or len(decades) > 1: intent = "PLOT_EVOLUTION"
        else: intent = "PLOT_COMPARE"
    elif "equipo" in q_norm and ("top" in q_norm or "mas" in q_norm or "menos" in q_norm): intent = "TEAM_RANKING"
    elif "mvp" in q_norm and year and "cuantos" not in q_norm: intent = "YEARLY_LEADER"
    elif year and ("maximo" in q_norm or "lider" in q_norm or "top" in q_norm): intent = "YEARLY_LEADER"
    elif ("historia" in q_norm or "siempre" in q_norm) and ("maximo" in q_norm or "lider" in q_norm or "mas" in q_norm): intent = "HISTORY_LEADER"
    elif len(teams) >= 2 and intent != "PLOT_COMPARE": intent = "COMPARE_TEAMS"
    elif len(players) >= 2 and ("mejor" in q_norm or "mas" in q_norm or "entre" in q_norm): intent = "COMPARE_PLAYERS"
    elif "cuantos" in q_norm and "mvp" in q_norm and pos_code: intent = "COUNT_STATS"
    elif college: intent = "COLLEGE_STATS"
    elif len(players) >= 1 and intent == "COMPARE_PLAYERS": intent = "PLAYER_STATS" 

    # --- J. GRÁFICAS (MEJORADO: Filtro Década y Barras para Single Player) ---
    if intent == "PLOT_COMPARE":
        cols_to_plot = []
        for code, name in stats_requested:
            if "_PCT" in code: code = code.replace("_PCT", "") 
            suffix = "_Total" if not use_avg and code not in ['W', 'L', 'FG_PCT', '3P_PCT', 'FT_PCT'] else ""
            cols_to_plot.append(f"{code}{suffix}")
            
        index_col = 'Age' if "edad" in q_norm else 'Year'
        
        if players:
            df_plot = df[df['Player'].isin(players)].copy()
            # Aplicar filtro de década también a jugadores
            if decade: 
                df_plot = df_plot[(df_plot['Year'] >= decade) & (df_plot['Year'] <= decade+9)]
                
            if df_plot.empty: return "❌ No encontré datos para esos jugadores en ese periodo.", None, None
            
            # Si es 1 jugador y múltiples métricas (ej: Luka triples, libres, 2p) -> BARRAS
            if len(players) == 1 and len(cols_to_plot) > 1:
                pivot_df = df_plot.pivot_table(index=index_col, columns='Player', values=cols_to_plot, aggfunc='sum').fillna(0)
                pivot_df.columns = [f"{s[1]}" for s in stats_requested]
                return f"📊 Estadísticas de **{players[0]}** por {index_col}:", pivot_df, 'bar'
            
            # Comparativa estándar
            pivot_df = df_plot.pivot_table(index=index_col, columns='Player', values=cols_to_plot[0], aggfunc='sum').fillna(0)
            return f"📊 Comparativa de **{stats_requested[0][1]}** entre {', '.join(players)}:", pivot_df, 'line'
            
        elif teams:
            df_plot = df[df['Team'].isin(teams)].copy()
            if cols_to_plot[0] in ['W', 'L']: df_plot = df_plot.drop_duplicates(subset=['Year', 'Team'])
            if decade: df_plot = df_plot[(df_plot['Year'] >= decade) & (df_plot['Year'] <= decade+9)]
            pivot_df = df_plot.pivot_table(index='Year', columns='Team', values=cols_to_plot[0], aggfunc='sum').fillna(0)
            return f"📊 Comparativa de **{stats_requested[0][1]}** entre {', '.join(teams)}:", pivot_df, 'line'

    elif intent == "PLOT_EVOLUTION":
        stat_code, stat_lbl = stats_requested[0]
        if "_PCT" in stat_code: stat_code = stat_code.replace("_PCT", "")
        is_team_metric = stat_code in ['W', 'L']
        col_data = stat_code if is_team_metric else f"{stat_code}_Total"
        if use_avg and not is_team_metric: col_data = stat_code 
        
        if len(decades) > 1:
            df_dec = df[df['Year'].apply(lambda y: any(y >= d and y <= d+9 for d in decades))].copy()
            df_dec['Decade'] = (df_dec['Year'] // 10) * 10
            evol_df = df_dec.groupby('Decade')[col_data].sum()
            return f"📈 Comparativa de **{stat_lbl}** entre décadas:", evol_df, 'bar'

        target_df = df
        agg_method = 'sum'
        if teams:
            target_df = target_df[target_df['Team'].isin(teams)]
            if is_team_metric: agg_method = 'max'
            elif use_avg and stat_code == 'PTS': agg_method = 'sum' 
            else: agg_method = 'sum' 
            
        evol_df = target_df.groupby('Year')[col_data].agg(agg_method)
        ctx = teams[0] if teams else "la NBA"
        return f"📈 Evolución de **{stat_lbl}** de {ctx}:", evol_df, 'line'

    # --- G. ESTADÍSTICAS JUGADOR (MEJORADO Total + Promedio) ---
    elif intent == "PLAYER_STATS":
        if not players: return "❌ No encontré al jugador.", None, None
        
        # CASO 1: Año específico
        if year:
            res = ""
            found_any = False
            for p in players:
                p_data = df[(df['Player'] == p) & (df['Year'] == year)]
                if p_data.empty: continue
                found_any = True
                
                details = []
                for code, name in stats_requested:
                    if code.endswith("_PCT"): # Porcentajes (Sin promedio extra)
                        base = code.replace("_PCT", "")
                        m, a = p_data[f"{base}_Total"].sum(), p_data[f"{base}A_Total"].sum()
                        val = (m/a*100) if a>0 else 0
                        details.append(f"**{val:.1f}%** {name}")
                    else: # Totales
                        if use_avg:
                            # Si pide promedio explícitamente, solo promedio
                            val = p_data[code].sum()
                            details.append(f"**{val:.1f}** {name} (prom)")
                        else:
                            # Si pide Total, dar Total + (Promedio)
                            val_tot = p_data[f"{code}_Total"].sum()
                            games = p_data['G'].sum()
                            val_avg = val_tot / games if games > 0 else 0
                            details.append(f"**{int(val_tot):,}** {name} ({val_avg:.1f} pp)")
                
                res += f"📌 **{p}** ({year}): {', '.join(details)}\n"
            
            if found_any: return res, None, None
            return f"❌ No encontré datos de {', '.join(players)} en {year}.", None, None

        # CASO 2: Carrera (Sin cambios, ya muestra todo)
        final_res = ""
        for p in players:
            p_data = df[df['Player'] == p]
            if p_data.empty: continue
            
            pts = p_data['PTS_Total'].sum()
            ast = p_data['AST_Total'].sum()
            trb = p_data['TRB_Total'].sum()
            stl = p_data['STL_Total'].sum()
            blk = p_data['BLK_Total'].sum()
            g = p_data['G'].sum()
            
            fg_pct = (p_data['FG_Total'].sum() / p_data['FGA_Total'].sum() * 100) if p_data['FGA_Total'].sum() > 0 else 0
            p3_pct = (p_data['3P_Total'].sum() / p_data['3PA_Total'].sum() * 100) if p_data['3PA_Total'].sum() > 0 else 0
            ft_pct = (p_data['FT_Total'].sum() / p_data['FTA_Total'].sum() * 100) if p_data['FTA_Total'].sum() > 0 else 0

            final_res += f"• **Estadísticas de carrera - {p}:**\n"
            final_res += f"• **Partidos:** {int(g)}\n"
            final_res += f"• **Puntos:** {int(pts):,} ({pts/g:.1f} ppp)\n"
            final_res += f"• **Rebotes:** {int(trb):,} ({trb/g:.1f} rpp)\n"
            final_res += f"• **Asistencias:** {int(ast):,} ({ast/g:.1f} app)\n"
            final_res += f"• **Robos:** {int(stl):,} ({stl/g:.1f} rpp)\n"
            final_res += f"• **Tapones:** {int(blk):,} ({blk/g:.1f} tpp)\n"
            final_res += f"• **% TC:** {fg_pct:.1f}% | **% 3P:** {p3_pct:.1f}% | **% TL:** {ft_pct:.1f}%\n\n"
            
        return final_res, None, None

    # --- COMPARATIVAS (MANTENIDO V29) ---
    elif intent == "COMPARE_PLAYERS":
        df_comp = df[df['Player'].isin(players)]
        if year: df_comp = df_comp[df_comp['Year'] == year]
        if df_comp.empty: return "❌ Sin datos para comparar.", None, None
        
        code, name = stats_requested[0]
        results = []
        
        for p in players:
            p_rows = df_comp[df_comp['Player'] == p]
            if p_rows.empty: continue
            val = 0
            if code.endswith("_PCT"):
                base = code.replace("_PCT", "")
                m, a = p_rows[f"{base}_Total"].sum(), p_rows[f"{base}A_Total"].sum()
                val = (m/a*100) if a>0 else 0
            else:
                col = code if use_avg else f"{code}_Total"
                val = p_rows[col].sum()
            results.append((val, p))
            
        results.sort(reverse=True)
        details_str = " vs ".join([f"{r[1]} ({r[0]:.1f}{'%' if '_PCT' in code else ''})" for r in results])
        winner = results[0][1]
        return f"Entre {details_str}, tiene mejor registro **{winner}**.", None, None

    # --- EQUIPOS (MANTENIDO V29) ---
    elif intent == "TEAM_STATS" or intent == "TEAM_RANKING":
        if "top" in q_norm or "mas" in q_norm and not players:
            stat_code, stat_lbl = stats_requested[0]
            if "derrota" in q_norm: col_sort = 'L'
            elif "victoria" in q_norm: col_sort = 'W'
            else: col_sort = 'team_PTS_per_game'
            
            u_teams = df[df['Year'] == year].drop_duplicates('Team')
            top = u_teams.sort_values(col_sort, ascending=False).head(limit)
            res = f"Top {limit} equipos - {stat_lbl} ({year}):\n"
            for i, (_, r) in enumerate(top.iterrows()):
                val = r[col_sort]
                if col_sort == 'team_PTS_per_game': val = val * 82 
                res += f"{i+1}. **{r['Team']}**: {int(val)}\n"
            return res, None, None
        
        if teams and year:
            t = teams[0]
            t_data = df[(df['Team'] == t) & (df['Year'] == year)]
            
            if any(x in q_norm for x in ["plantilla", "jugadores", "roster", "quienes"]):
                res = f"🏀 **Plantilla {t} ({year}):**\n"
                sorted_roster = t_data.sort_values('PTS_Total', ascending=False)
                for _, r in sorted_roster.iterrows():
                    res += f"* {r['Player']} ({r['Pos']})\n"
                return res, None, None
                
            if "puntos" in q_norm or "anoto" in q_norm:
                total_pts = t_data['PTS_Total'].sum()
                return f"El equipo **{t}** anotó un total de **{int(total_pts):,} puntos** en {year}.", None, None
            return f"Datos de **{t}** en {year}: {int(t_data['W'].iloc[0])} victorias.", None, None

    elif intent == "COUNT_STATS":
        if "mvp" in q_norm and pos_code:
            filtered = df[(df['Pos'].str.contains(pos_code, case=False, na=False)) & (df['MVP'] > 0)]
            filtered = filtered.sort_values('Year')
            unique_players = filtered['Player'].unique()
            total_awards = len(filtered)
            details = []
            for _, r in filtered.iterrows():
                details.append(f"{r['Year']}: {r['Player']} ({r['Team']})")
            res_list = "\n".join(details)
            return f"En total, **{total_awards}** premios MVP han sido ganados por **{len(unique_players)}** {pos_name} distintos:\n\n{res_list}", None, None

    # --- HISTÓRICOS Y OTROS (MANTENIDO V29) ---
    elif intent == "HISTORY_LEADER":
        if "mvp" in q_norm:
            grouped = df.groupby('Player_ID')['MVP'].sum().reset_index().sort_values('MVP', ascending=False)
            top = grouped.iloc[0]
            name = top['Player_ID'].split('(')[0]
            return f"El jugador con más MVPs de la historia es **{name}** con **{int(top['MVP'])}** galardones.", None, None
            
        stat_code, stat_lbl = stats_requested[0]
        col_sort = f"{stat_code}_Total"
        target_df = df
        if teams: target_df = target_df[target_df['Team'].isin(teams)]
        grouped = target_df.groupby('Player_ID')[col_sort].sum().reset_index().sort_values(col_sort, ascending=False).head(limit)
        idx = 1 if "segundo" in q_norm else (2 if "tercer" in q_norm else 0)
        
        if limit > 1 and idx == 0:
            res = f"Top {limit} máximos {stat_lbl} de la historia:\n"
            for i, (_, r) in enumerate(grouped.iterrows()):
                name = r['Player_ID'].split('(')[0]
                res += f"{i+1}. **{name}**: {int(r[col_sort]):,}\n"
            return res, None, None
        if len(grouped) > idx:
            top = grouped.iloc[idx]
            name = top['Player_ID'].split('(')[0]
            return f"El {'máximo' if idx==0 else str(idx+1)+'º máximo'} {stat_lbl} es **{name}** con {int(top[col_sort]):,}.", None, None

    elif intent == "YEARLY_LEADER":
        if not year: return "📅 Indícame el año.", None, None
        target_df = df[df['Year'] == year]
        if teams: target_df = target_df[target_df['Team'].isin(teams)]
        if pos_code: target_df = target_df[target_df['Pos'].str.contains(pos_code, case=False, na=False)]
        
        stat_code, stat_lbl = stats_requested[0]
        if "mvp" in q_norm:
             mvp = target_df.sort_values('MVP', ascending=False).iloc[0]
             if mvp['MVP'] == 0: mvp = target_df.sort_values('Share', ascending=False).iloc[0]
             return f"El MVP de {year} fue **{mvp['Player']}** ({mvp['Team']}).", None, None

        col_sort = stat_code if use_avg else f"{stat_code}_Total"
        sorted_df = target_df.sort_values(col_sort, ascending=False)
        leader = sorted_df.iloc[rank_idx] if len(sorted_df) > rank_idx else sorted_df.iloc[0]
        
        if limit > 1 and rank_idx == 0:
             res = f"Top {limit} {stat_lbl} ({year}):\n"
             for i, (_, r) in enumerate(sorted_df.head(limit).iterrows()):
                 res += f"{i+1}. **{r['Player']}**: {int(r[col_sort])}\n"
             return res, None, None
        return f"Líder en {stat_lbl} de {year}: **{leader['Player']}** ({int(leader[col_sort]):,}).", None, None

    elif intent == "DECADE_LEADER":
         if not decade: return "⚠️ Especifica la década.", None, None
         end = decade + 9
         d = df[(df['Year'] >= decade) & (df['Year'] <= end)]
         stat_code = stats_requested[0][0]
         col = f"{stat_code}_Total"
         if pos_code: d = d[d['Pos'].str.contains(pos_code, case=False, na=False)]
         if "equipo" in q_norm and not players:
             top = d.groupby('Team')[col].sum().reset_index().sort_values(col, ascending=False)
             leader = top.iloc[rank_idx]
             return f"El equipo líder en {stats_requested[0][1]} de los {decade} es **{leader['Team']}**.", None, None
         top = d.groupby('Player_ID')[col].sum().reset_index().sort_values(col, ascending=False)
         leader = top.iloc[rank_idx]
         name = leader['Player_ID'].split('(')[0]
         return f"El líder en {stats_requested[0][1]} de los {decade} es **{name}** ({int(leader[col]):,}).", None, None

    elif intent == "COMPARE_TEAMS":
        if not year: return "📅 Necesito el año.", None, None
        t_data = df[(df['Team'].isin(teams)) & (df['Year'] == year)].drop_duplicates('Team')
        if t_data.empty: return "❌ No encontré datos.", None, None
        if any(x in q_norm for x in ['gano', 'ganar', 'victorias', 'partidos']):
            sorted_teams = t_data.sort_values('W', ascending=False)
            w, l = sorted_teams.iloc[0], sorted_teams.iloc[-1]
            return f"En {year}, **{w['Team']}** ganó más ({int(w['W'])}) que **{l['Team']}** ({int(l['W'])}).", None, None
        else:
            sorted_teams = t_data.sort_values('team_PTS_per_game', ascending=False)
            w, l = sorted_teams.iloc[0], sorted_teams.iloc[-1]
            val_w = int(w['team_PTS_per_game']*82)
            val_l = int(l['team_PTS_per_game']*82)
            return f"En {year}, **{w['Team']}** anotó más ({val_w:,}) que **{l['Team']}** ({val_l:,}).", None, None

    elif intent == "COLLEGE_STATS":
        if not college or not year: return "❌ Faltan datos (College + Año).", None, None
        u_df = df[(df['Colleges'].str.contains(college, case=False)) & (df['Year'] == year)]
        if "maximo" in q_norm:
             top = u_df.sort_values('PTS_Total', ascending=False).iloc[0]
             return f"Líder de {college}: **{top['Player']}**.", None, None
        res = f"Jugadores de **{college}** ({year}):\n"
        for _, r in u_df.iterrows(): res += f"• {r['Player']} ({int(r['PTS_Total'])} pts)\n"
        return res, None, None

    return "❌ No entendí bien la pregunta. Prueba reformulándola.", None, None

# --- MODO LOTES ---
def batch_process(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    results = ""
    for line in stringio:
        q = line.strip()
        if q: 
            resp, _, _ = get_answer(q)
            results += f"P: {q}\nR: {resp}\n" + "-"*30 + "\n"
    return results

# --- INTERFAZ ---
st.title("📊 NBA Stats AI")

with st.sidebar:
    st.header("Modo Lotes")
    uploaded_file = st.file_uploader("Sube tu .txt", type="txt")
    if uploaded_file and st.button("Procesar"):
        st.download_button("Descargar Resultados", batch_process(uploaded_file), "respuestas.txt")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "chart_data" in m and m["chart_data"] is not None:
            if m["chart_type"] == 'bar': st.bar_chart(m["chart_data"])
            else: st.line_chart(m["chart_data"])

if prompt := st.chat_input("Pregunta sobre la NBA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    resp_text, chart_data, chart_type = get_answer(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(resp_text)
        if chart_data is not None:
            if chart_type == 'bar': st.bar_chart(chart_data)
            else: st.line_chart(chart_data)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": resp_text, 
        "chart_data": chart_data, 
        "chart_type": chart_type
    })
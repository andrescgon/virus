# -*- coding: utf-8 -*-
"""
ABM de propagación de virus en Mesa 3.x con visualización Bokeh/Panel

Cambios clave de esta versión:
- Partículas: el plot ahora retorna tres ColumnDataSource (S, I, R) y en
  refresh_plots() se actualizan esos tres directamente (¡ya se mueven!).
- Eje X dinámico (se extiende con los pasos).
- Eliminada la gráfica Rt(t).
- Callbacks seguros con _next_tick para evitar race conditions.
"""

import enum
from typing import Dict, List

import numpy as np
import pandas as pd

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from bokeh.models import ColumnDataSource, Div, Range1d
from bokeh.plotting import figure
from bokeh.palettes import Category10

import panel as pn


# -----------------------------
# Estados y Agente
# -----------------------------

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2


class Person(Agent):
    def __init__(self, model, *, masked=False, distancer=False,
                 mobility_factor=1.0, isolation_compliant=False,
                 vaccinated=False):
        super().__init__(model)
        self.state = State.REMOVED if vaccinated else State.SUSCEPTIBLE
        self.infection_time = 0
        self.recovery_time = 0
        self.masked = masked
        self.distancer = distancer
        self.mobility_factor = mobility_factor
        self.isolation_compliant = isolation_compliant
        self.quarantined = False

    def move(self):
        if self.quarantined:
            return
        if self.random.random() > self.mobility_factor:
            return
        neigh = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if neigh:
            self.model.grid.move_agent(self, self.random.choice(neigh))

    def status(self):
        if self.state == State.INFECTED:
            dr = self.model.death_rate
            if dr > 0 and np.random.choice([0, 1], p=[dr, 1 - dr]) == 0:
                # Muere
                try:
                    self.model.grid.remove_agent(self)
                except Exception:
                    pass
                try:
                    self.model.agents.remove(self)
                except ValueError:
                    pass
                self.model.dead_agents += 1
                return
            t = self.model.steps - self.infection_time
            if t >= self.recovery_time:
                self.state = State.REMOVED
                self.quarantined = False
            else:
                if (not self.quarantined and self.isolation_compliant and
                        t >= self.model.detection_delay):
                    self.quarantined = True

    def contact(self):
        if self.distancer and self.model.random.random() < self.model.distancing_effect:
            return
        mates = self.model.grid.get_cell_list_contents([self.pos])
        if len(mates) <= 1:
            return
        for other in mates:
            if other is self:
                continue
            if self.state == State.INFECTED and other.state == State.SUSCEPTIBLE:
                p = self.model.ptrans
                if self.masked:
                    p *= (1.0 - self.model.mask_effect)
                if other.masked:
                    p *= (1.0 - self.model.mask_effect)
                if other.distancer and self.model.random.random() < self.model.distancing_effect:
                    continue
                if self.quarantined:
                    continue
                if self.random.random() <= max(0.0, min(1.0, p)):
                    other.state = State.INFECTED
                    other.infection_time = self.model.steps
                    other.recovery_time = self.model.get_recovery_time()
                    self.model.new_infections_this_step += 1

    def step(self):
        self.status()
        if self in self.model.agents:
            self.move()
            self.contact()


# -----------------------------
# Modelo
# -----------------------------

class InfectionModel(Model):
    def __init__(
        self,
        N=400, width=20, height=20,
        ptrans=0.25, death_rate=0.01,
        recovery_days=21, recovery_sd=7,
        vaccination_rate=0.2, mask_compliance=0.7, mask_effect=0.5,
        distancing_compliance=0.5, distancing_effect=0.5,
        mobility_factor=0.7,
        isolation_compliance=0.6, detection_delay=3,
    ):
        super().__init__()
        self.steps = 0
        self.N = N
        self.grid = MultiGrid(width, height, torus=True)
        self.ptrans = ptrans
        self.death_rate = death_rate
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.vaccination_rate = float(np.clip(vaccination_rate, 0.0, 1.0))
        self.mask_compliance = float(np.clip(mask_compliance, 0.0, 1.0))
        self.mask_effect = float(np.clip(mask_effect, 0.0, 1.0))
        self.distancing_compliance = float(np.clip(distancing_compliance, 0.0, 1.0))
        self.distancing_effect = float(np.clip(distancing_effect, 0.0, 1.0))
        self.mobility_factor = float(np.clip(mobility_factor, 0.0, 1.0))
        self.isolation_compliance = float(np.clip(isolation_compliance, 0.0, 1.0))
        self.detection_delay = int(max(0, detection_delay))

        self.running = True
        self.dead_agents = 0
        self.new_infections_history: List[int] = []
        self.infected_history: List[int] = []
        self.new_infections_this_step = 0

        # Crear agentes
        for _ in range(self.N):
            masked = (self.random.random() < self.mask_compliance)
            distancer = (self.random.random() < self.distancing_compliance)
            isolation_c = (self.random.random() < self.isolation_compliance)
            vaccinated = (self.random.random() < self.vaccination_rate)
            a = Person(
                self,
                masked=masked, distancer=distancer,
                mobility_factor=self.mobility_factor,
                isolation_compliant=isolation_c,
                vaccinated=vaccinated,
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        susceptibles = [ag for ag in self.agents if ag.state == State.SUSCEPTIBLE]
        k = max(1, int(0.02 * len(susceptibles)))
        if k > 0 and susceptibles:
            for a in self.random.sample(susceptibles, k):
                a.state = State.INFECTED
                a.infection_time = 0
                a.recovery_time = self.get_recovery_time()

        self.datacollector = DataCollector(agent_reporters={"State": lambda a: a.state})
        self.datacollector.collect(self)
        self._push_counters()

    def get_recovery_time(self) -> int:
        return int(max(1, self.random.normalvariate(self.recovery_days, self.recovery_sd)))

    def _push_counters(self):
        df = self.datacollector.get_agent_vars_dataframe().reset_index()
        if df.empty:
            self.infected_history.append(0)
        else:
            cur = df["Step"].max()
            snap = df[df["Step"] == cur]
            self.infected_history.append(int((snap["State"] == int(State.INFECTED)).sum()))
        self.new_infections_history.append(int(self.new_infections_this_step))
        self.new_infections_this_step = 0

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        self.steps += 1
        self._push_counters()
        if len(self.agents) == 0:
            self.running = False


# -----------------------------
# Datos y métricas
# -----------------------------

def states_timeseries(model: InfectionModel) -> pd.DataFrame:
    agent_state = model.datacollector.get_agent_vars_dataframe()
    if agent_state.empty:
        return pd.DataFrame({"Step": [0], "Susceptible": [model.N], "Infected": [0], "Removed": [0]})
    df = agent_state.reset_index()
    counts = df.groupby(["Step", "State"]).size().unstack(fill_value=0)
    out = pd.DataFrame({"Step": counts.index})
    # counts.get(...) puede devolver int; convertir a array para que siempre funcione
    out["Susceptible"] = np.asarray(counts.get(int(State.SUSCEPTIBLE), 0)).tolist()
    out["Infected"]    = np.asarray(counts.get(int(State.INFECTED), 0)).tolist()
    out["Removed"]     = np.asarray(counts.get(int(State.REMOVED), 0)).tolist()
    return out.reset_index(drop=True)


def composition_now(model: InfectionModel) -> Dict[str, int]:
    df = model.datacollector.get_agent_vars_dataframe().reset_index()
    cur = df["Step"].max()
    snap = df[df["Step"] == cur]
    S = int((snap["State"] == int(State.SUSCEPTIBLE)).sum())
    I = int((snap["State"] == int(State.INFECTED)).sum())
    R = int((snap["State"] == int(State.REMOVED)).sum())
    return {"S": S, "I": I, "R": R}


def compute_metrics(model: InfectionModel) -> Dict[str, float]:
    X = states_timeseries(model)
    total = int(X[["Susceptible", "Infected", "Removed"]].iloc[0].sum()) if not X.empty else model.N
    peak_I = int(X["Infected"].max()) if not X.empty else 0
    final_R = int(X["Removed"].iloc[-1]) if not X.empty else 0
    attack = 100.0 * final_R / max(1, total)
    rts = []
    for t in range(1, len(model.new_infections_history)):
        new = model.new_infections_history[t]
        prev_I = model.infected_history[t - 1]
        rts.append(new / max(1, prev_I))
    rt_mean = float(np.mean(rts)) if rts else 0.0
    return {"Pico_I": peak_I, "Ataque_%": round(attack, 2), "Rt_promedio": round(rt_mean, 3), "Muertes": int(model.dead_agents)}


# -----------------------------
# Plots
# -----------------------------

def build_sir_area_plot(X: pd.DataFrame, title: str, steps_init: int, N: int):
    src = ColumnDataSource(X)
    p = figure(width=820, height=360, title=title, x_axis_label="Step", y_axis_label="Agents",
               tools="", sizing_mode="fixed")
    p.x_range = Range1d(0, max(1, steps_init))
    p.y_range = Range1d(0, max(1, N))
    colors = list(Category10[3][:3])
    p.varea_stack(stackers=["Susceptible", "Infected", "Removed"], x="Step", color=colors,
                  legend_label=["Susceptible", "Infected", "Removed"], source=src, alpha=0.85)
    p.legend.location = "top_right"
    p.background_fill_color = "#f5f6fa"
    p.background_fill_alpha = 1.0
    p.toolbar.logo = None
    return p, src


def build_sir_line_plot(X: pd.DataFrame, title: str, steps_init: int, N: int):
    src = ColumnDataSource(X)
    p = figure(width=820, height=360, title=title, x_axis_label="Step", y_axis_label="Agents",
               tools="", sizing_mode="fixed")
    p.x_range = Range1d(0, max(1, steps_init))
    p.y_range = Range1d(0, max(1, N))
    colors = list(Category10[3][:3])
    p.line("Step", "Susceptible", source=src, line_width=3, alpha=0.95, color=colors[0], legend_label="Susceptible")
    p.line("Step", "Infected",    source=src, line_width=3, alpha=0.95, color=colors[1], legend_label="Infected")
    p.line("Step", "Removed",     source=src, line_width=3, alpha=0.95, color=colors[2], legend_label="Removed")
    p.legend.location = "top_right"
    p.background_fill_color = "#ffffff"
    p.toolbar.logo = None
    return p, src


def build_comp_bar(model: InfectionModel):
    comp = composition_now(model)
    df = pd.DataFrame({"cat": ["Susceptible", "Infected", "Removed"], "val": [comp['S'], comp['I'], comp['R']]})
    src = ColumnDataSource(df)
    p = figure(width=360, height=240, y_range=list(df["cat"][::-1]), tools="", title="Composición actual",
               sizing_mode="fixed")
    p.hbar(y="cat", right="val", height=0.6, source=src)
    p.background_fill_color = "#f5f6fa"
    p.toolbar.logo = None
    return p, src


def build_particles_plot(model: InfectionModel):
    """Crea el plot de partículas con 3 fuentes (S, I, R) para poder actualizarlas."""
    colors = {0: Category10[3][0], 1: Category10[3][1], 2: Category10[3][2]}
    p = figure(width=600, height=600, title="Mapa de partículas (agentes)",
               tools="pan,wheel_zoom,reset,save", sizing_mode="fixed")
    p.x_range = Range1d(-1, model.grid.width + 1)
    p.y_range = Range1d(-1, model.grid.height + 1)
    p.background_fill_color = "#000000"
    p.grid.grid_line_color = None
    p.toolbar.logo = None

    # Fuentes vacías iniciales (se rellenan en refresh_plots)
    src_s = ColumnDataSource({"x": [], "y": []})
    src_i = ColumnDataSource({"x": [], "y": []})
    src_r = ColumnDataSource({"x": [], "y": []})

    p.scatter(x="x", y="y", size=6, alpha=0.85, color=colors[0], legend_label="Susceptible", source=src_s)
    p.scatter(x="x", y="y", size=6, alpha=0.85, color=colors[1], legend_label="Infected",    source=src_i)
    p.scatter(x="x", y="y", size=6, alpha=0.85, color=colors[2], legend_label="Removed",     source=src_r)
    p.legend.location = "top_left"

    # Devolvemos un dict con las tres fuentes
    return p, {"S": src_s, "I": src_i, "R": src_r}


# -----------------------------
# Utilidad: next tick seguro
# -----------------------------

def _next_tick(func):
    """Ejecuta func en el próximo tick si hay curdoc; si no, lo ejecuta ya."""
    doc = pn.state.curdoc
    if doc is None:
        func()
    else:
        doc.add_next_tick_callback(func)


# -----------------------------
# App
# -----------------------------

def make_app(
    pop=400, width=20, height=20, ptrans=0.25, death_rate=0.01,
    recovery_days=21, recovery_sd=7,
    vaccination_rate=0.2, mask_compliance=0.7, mask_effect=0.5,
    distancing_compliance=0.5, distancing_effect=0.5,
    mobility_factor=0.7,
    isolation_compliance=0.6, detection_delay=3,
    steps_init=150, update_ms=150,
):
    model = InfectionModel(
        N=pop, width=width, height=height,
        ptrans=ptrans, death_rate=death_rate,
        recovery_days=recovery_days, recovery_sd=recovery_sd,
        vaccination_rate=vaccination_rate,
        mask_compliance=mask_compliance, mask_effect=mask_effect,
        distancing_compliance=distancing_compliance, distancing_effect=distancing_effect,
        mobility_factor=mobility_factor,
        isolation_compliance=isolation_compliance, detection_delay=detection_delay,
    )

    pn.extension()

    # Datos iniciales
    X0 = states_timeseries(model)
    p_area, src_area = build_sir_area_plot(X0, title=f"S/I/R — Step={model.steps}", steps_init=steps_init, N=pop)
    p_line, src_line = build_sir_line_plot(X0, title=f"S/I/R (líneas) — Step={model.steps}", steps_init=steps_init, N=pop)
    p_comp, src_comp = build_comp_bar(model)
    p_pts, src_pts_map = build_particles_plot(model)

    # Métricas
    m_div = Div(text="", width=360)

    def update_metrics():
        m = compute_metrics(model)
        m_div.text = (
            f"<b>Métricas</b><br>"
            f"• Pico de Infectados: <b>{m['Pico_I']}</b><br>"
            f"• Ataque final: <b>{m['Ataque_%']}%</b><br>"
            f"• Rt promedio (proxy): <b>{m['Rt_promedio']}</b><br>"
            f"• Muertes: <b>{m['Muertes']}</b>"
        )

    update_metrics()

    # Layout
    resumen = pn.Column(
        pn.Row(p_area, pn.Column(p_comp, m_div, sizing_mode="fixed"), sizing_mode="fixed"),
        p_line,
        sizing_mode="fixed"
    )
    espacio = pn.Row(p_pts, sizing_mode="fixed")
    tabs = pn.Tabs(("Resumen", resumen), ("Espacio", espacio), sizing_mode="fixed")

    def maybe_extend_xrange(fig, step, margin=10):
        if step + margin > fig.x_range.end:
            fig.x_range.end = step + margin

    def refresh_plots():
        X = states_timeseries(model)
        # Área
        src_area.data = {col: X[col].tolist() for col in X.columns}
        p_area.title.text = f"S/I/R — Step={model.steps}"
        maybe_extend_xrange(p_area, model.steps)
        p_area.y_range.end = max(p_area.y_range.end, model.N)
        # Líneas
        src_line.data = {col: X[col].tolist() for col in X.columns}
        p_line.title.text = f"S/I/R (líneas) — Step={model.steps}"
        maybe_extend_xrange(p_line, model.steps)
        p_line.y_range.end = max(p_line.y_range.end, model.N)
        # Composición
        comp = composition_now(model)
        src_comp.data = {"cat": ["Susceptible", "Infected", "Removed"],
                         "val": [comp["S"], comp["I"], comp["R"]]}

        # Partículas: construir coordenadas separadas por estado
        xs_s, ys_s, xs_i, ys_i, xs_r, ys_r = [], [], [], [], [], []
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell = model.grid.get_cell_list_contents([(x, y)])
                for a in cell:
                    jx = x + np.random.uniform(-0.4, 0.4)
                    jy = y + np.random.uniform(-0.4, 0.4)
                    if a.state == State.SUSCEPTIBLE:
                        xs_s.append(jx); ys_s.append(jy)
                    elif a.state == State.INFECTED:
                        xs_i.append(jx); ys_i.append(jy)
                    else:
                        xs_r.append(jx); ys_r.append(jy)

        src_pts_map["S"].data = {"x": xs_s, "y": ys_s}
        src_pts_map["I"].data = {"x": xs_i, "y": ys_i}
        src_pts_map["R"].data = {"x": xs_r, "y": ys_r}

        update_metrics()

    # Bucle
    def update():
        if not model.running:
            cb.stop()
            return
        try:
            model.step()
            _next_tick(refresh_plots)
        except Exception as e:
            print(f"Error en update: {e}")
            cb.stop()

    cb = pn.state.add_periodic_callback(update, period=update_ms, start=False)

    # Controles
    start_btn = pn.widgets.Button(name="Start", button_type="primary")
    stop_btn  = pn.widgets.Button(name="Stop",  button_type="danger", disabled=True)
    reset_btn = pn.widgets.Button(name="Reset", button_type="success")

    def start_evt(_):
        if not cb.running:
            cb.start()
            start_btn.disabled = True
            stop_btn.disabled = False

    def stop_evt(_):
        if cb.running:
            cb.stop()
            start_btn.disabled = False
            stop_btn.disabled = True

    def reset_evt(_):
        if cb.running:
            cb.stop()
        start_btn.disabled = False
        stop_btn.disabled = True
        nonlocal model
        model = InfectionModel(
            N=s_pop.value, width=s_w.value, height=s_h.value,
            ptrans=s_p.value, death_rate=s_dr.value,
            recovery_days=s_rec.value, recovery_sd=s_sd.value,
            vaccination_rate=s_vacc.value,
            mask_compliance=s_mcomp.value, mask_effect=s_meff.value,
            distancing_compliance=s_dcomp.value, distancing_effect=s_deff.value,
            mobility_factor=s_mob.value,
            isolation_compliance=s_isoc.value, detection_delay=s_det.value,
        )
        # Reset de rangos
        p_area.y_range.end = max(1, s_pop.value)
        p_line.y_range.end = max(1, s_pop.value)
        p_area.x_range.end = max(1, steps_init)
        p_line.x_range.end = max(1, steps_init)
        _next_tick(refresh_plots)

    start_btn.on_click(start_evt)
    stop_btn.on_click(stop_evt)
    reset_btn.on_click(reset_evt)

    # Sliders (se aplican con Reset)
    s_pop  = pn.widgets.IntSlider(name="Población", value=pop, start=50, end=5000, step=50)
    s_w    = pn.widgets.IntSlider(name="Ancho grid", value=width, start=10, end=60)
    s_h    = pn.widgets.IntSlider(name="Alto grid", value=height, start=10, end=60)
    s_p    = pn.widgets.FloatSlider(name="p_trans (base)", value=ptrans, start=0.0, end=1.0, step=0.01)
    s_dr   = pn.widgets.FloatSlider(name="Mortalidad por paso", value=death_rate, start=0.0, end=0.1, step=0.001)
    s_rec  = pn.widgets.IntSlider(name="Recuperación media", value=recovery_days, start=3, end=60)
    s_sd   = pn.widgets.IntSlider(name="Recuperación sd", value=recovery_sd, start=0, end=30)
    s_vacc = pn.widgets.FloatSlider(name="Vacunación inicial", value=vaccination_rate, start=0, end=1, step=0.05)
    s_mcomp= pn.widgets.FloatSlider(name="Cumplimiento mascarilla", value=mask_compliance, start=0, end=1, step=0.05)
    s_meff = pn.widgets.FloatSlider(name="Eficacia mascarilla", value=mask_effect, start=0, end=0.9, step=0.05)
    s_dcomp= pn.widgets.FloatSlider(name="Cumplimiento distanciamiento", value=distancing_compliance, start=0, end=1, step=0.05)
    s_deff = pn.widgets.FloatSlider(name="Efecto distanciamiento", value=distancing_effect, start=0, end=1, step=0.05)
    s_mob  = pn.widgets.FloatSlider(name="Movilidad (0=lockdown, 1=normal)", value=mobility_factor, start=0, end=1, step=0.05)
    s_isoc = pn.widgets.FloatSlider(name="Cumplimiento aislamiento", value=isolation_compliance, start=0, end=1, step=0.05)
    s_det  = pn.widgets.IntSlider(name="Retardo detección (steps)", value=detection_delay, start=0, end=14)

    apply_btn = pn.widgets.Button(name="Aplicar parámetros (Reset)", button_type="warning")
    apply_btn.on_click(lambda _: reset_evt(None))

    controls = pn.Column(
        pn.Row(start_btn, stop_btn, reset_btn, sizing_mode="fixed"),
        pn.Row(s_pop, s_w, s_h, sizing_mode="fixed"),
        pn.Row(s_p, s_dr, sizing_mode="fixed"),
        pn.Row(s_rec, s_sd, sizing_mode="fixed"),
        pn.Row(s_vacc, sizing_mode="fixed"),
        pn.Row(s_mcomp, s_meff, sizing_mode="fixed"),
        pn.Row(s_dcomp, s_deff, sizing_mode="fixed"),
        pn.Row(s_mob, sizing_mode="fixed"),
        pn.Row(s_isoc, s_det, sizing_mode="fixed"),
        apply_btn,
        sizing_mode="fixed"
    )

    # Primera actualización visual (segura)
    _next_tick(refresh_plots)

    return pn.Column(controls, tabs, sizing_mode="fixed")


if __name__ == "__main__":
    try:
        import mesa
        print(f"Mesa version: {mesa.__version__}")
    except Exception:
        print("Mesa version: desconocida")

    APP = make_app()
    pn.serve(
        APP,
        title="ABM infección — Visualización",
        show=True,
        port=5006,
        session_token_expiration=3600,
        session_timeout=3600
    )

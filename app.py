"""F1 Race Predictor web app. Run: streamlit run app.py"""
import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

import data
import features
import model

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏁", layout="wide")
# categorical palette in fixed order, stepped for the active theme
PALETTE = (["#3987e5", "#d95926", "#199e70", "#c98500"] if st.context.theme.type == "dark"
           else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
GROUP_COLORS = alt.Scale(domain=list(features.GROUPS), range=PALETTE)
METHOD_COLORS = alt.Scale(domain=model.METHODS, range=PALETTE[:3])


@st.cache_data(ttl=3600, show_spinner="Downloading results and weather (the first run takes a few minutes)…")
def dataset(today):
    df, history = features.build(data.load_history(today))
    return df, history


@st.cache_resource(show_spinner="Training models…")
def trained(today):
    df, _ = dataset(today)
    return {with_grid: model.train(df, with_grid) for with_grid in (False, True)}


@st.cache_data(show_spinner="Replaying the held-out seasons race by race (about a minute)…")
def evaluation(today):
    df, _ = dataset(today)
    return model.backtest(df, range(model.TEST_FROM, int(df["season"].max()) + 1))


@st.cache_data(ttl=600, show_spinner="Checking the next race…")
def next_race(today):
    return data.next_race(today)


today = dt.date.today()
with st.sidebar:
    st.title("🏁 F1 Race Predictor")
    rain = st.radio("Race conditions", ["Forecast", "Dry", "Wet"],
                    help="Forecast uses Open-Meteo once the race is within 16 days, and assumes dry before that.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

df, history = dataset(today)
models = trained(today)
race, entries, forecast, (fp_sessions, fp_gaps) = next_race(today)
st.sidebar.caption(f"Trained on {df[['season', 'round']].drop_duplicates().shape[0]} races "
                   f"({df['season'].min()}–{df['season'].max()}). Data: Jolpica F1, Open-Meteo.")

tab_next, tab_accuracy, tab_explore = st.tabs(["Next race", "Model accuracy", "Race explorer"])

with tab_next:
    if race is None:
        st.info("No upcoming race this season.")
    else:
        if rain == "Forecast":
            wet = bool(forecast)
            conditions = ("Wet" if wet else "Dry") + (" (forecast)" if forecast is not None else " (no forecast yet, assumed)")
        else:
            wet, conditions = rain == "Wet", f"{rain} (set manually)"
        if not entries:  # before qualifying, assume the last race's line-up
            last = df[(df["season"] == df["season"].max())]
            entries = last[last["round"] == last["round"].max()][["driver", "name", "team", "number"]].to_dict("records")
        for e in entries:
            e["practice"] = fp_gaps.get(e["number"], features.NAN)
        with_grid = all("grid" in e for e in entries)
        stage = ", ".join([f"{fp_sessions} practice session{'s' * (fp_sessions != 1)} used" if fp_sessions else "no practice run yet",
                           "qualifying done, using grid positions" if with_grid else "qualifying not run yet"])
        X = features.upcoming(history, entries, int(race["season"]), race["Circuit"]["circuitId"], wet)
        ranked = model.rank(models[with_grid], X)

        st.header(f"{race['season']} {race['raceName']}")
        st.caption(f"{race['Circuit']['circuitName']} · {race['date']} · {conditions} · {stage}. "
                   "Predicted from earlier races and this weekend's sessions only; refresh data after each session.")
        for col, (label, name) in zip(st.columns(3), zip(["Predicted winner", "Predicted P2", "Predicted P3"], ranked["name"])):
            col.metric(label, name)

        left, right = st.columns([5, 6])
        table = ranked[["predicted", "name", "team", "grid", "practice", "car_level", "car_trend", "teammate_delta", "track_delta"]]
        left.dataframe(table if with_grid else table.drop(columns="grid"), hide_index=True, height=35 * len(table) + 38,
                       column_config={
                           "predicted": "P", "name": "Driver", "team": "Team", "grid": "Grid",
                           "practice": st.column_config.NumberColumn("Practice gap", format="+%.2f%%",
                                                                     help="Best lap vs the fastest lap of the same practice session, best across sessions"),
                           "car_level": st.column_config.NumberColumn("Car pace", format="%.1f", help=features.LABELS["car_level"]),
                           "car_trend": st.column_config.NumberColumn("Car trend", format="%+.2f",
                                                                      help="Places gained (−) or lost (+) per race over the last 6 races"),
                           "teammate_delta": st.column_config.NumberColumn("vs teammate", format="%+.1f",
                                                                           help="Recent average places ahead (−) or behind (+) the teammate"),
                           "track_delta": st.column_config.NumberColumn("At this track", format="%+.1f",
                                                                        help="Past results here vs that season's average (− = better here)"),
                       })

        long = ranked.melt(id_vars=["name", "predicted"], value_vars=list(features.GROUPS), var_name="Factor", value_name="Effect")
        bars = alt.Chart(long).mark_bar(size=12).encode(
            x=alt.X("Effect:Q", title="← pushes down the order · pushes up the order →"),
            y=alt.Y("name:N", sort=alt.EncodingSortField("predicted", op="min"), title=None),
            color=alt.Color("Factor:N", scale=GROUP_COLORS, legend=alt.Legend(orient="top", title=None)),
            tooltip=["name:N", "Factor:N", alt.Tooltip("Effect:Q", format="+.2f")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeWidth=1, opacity=0.4).encode(x="x:Q")
        right.markdown("**Why each driver is placed there** (SHAP contribution of each factor group)")
        right.altair_chart((bars + zero).properties(height=24 * len(ranked)), width="stretch")

with tab_accuracy:
    scores, preds = evaluation(today)
    st.header(f"How well it predicted {model.TEST_FROM}–{df['season'].max()}")
    st.caption(f"Held-out seasons: model settings were tuned on {model.TUNE_SEASONS.start}–{model.TUNE_SEASONS.stop - 1} only "
               f"(python tune.py). Each race was predicted by a model trained only on the races before it, "
               f"then compared with the drivers who finished. {len(scores) // len(model.METHODS)} races.")
    st.dataframe(model.summary(scores).reset_index(), hide_index=True, column_config={
        "method": "Prediction",
        "spearman": st.column_config.NumberColumn("Rank correlation", format="%.2f", help="1 = predicted order exactly right"),
        "winner": st.column_config.NumberColumn("Winner correct", format="percent"),
        "podium": st.column_config.NumberColumn("Podium drivers correct (of 3)", format="%.2f"),
    })

    st.markdown("**By regulation phase** (rank correlation): rule resets are where past form says least")
    st.dataframe(model.summary(scores, ["phase"])["spearman"].unstack()[model.METHODS].round(2))

    left, right = st.columns(2)
    by_season = model.summary(scores, ["season"]).reset_index()
    base = alt.Chart(by_season).encode(
        x=alt.X("season:O", title=None),
        y=alt.Y("spearman:Q", title="Rank correlation", scale=alt.Scale(zero=False)),
        color=alt.Color("method:N", scale=METHOD_COLORS, legend=alt.Legend(orient="top", title=None)))
    lines = base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True).encode(
        tooltip=["season:O", "method:N", alt.Tooltip("spearman:Q", format=".2f"),
                 alt.Tooltip("winner:Q", title="winner correct", format=".0%")])
    left.markdown("**Accuracy per season**")
    left.altair_chart(lines.properties(height=320), width="stretch")

    imp = model.importance(models[True]).assign(label=lambda d: d["feature"].map(features.LABELS))
    right.markdown("**What the model relies on** (share of total split gain)")
    right.altair_chart(alt.Chart(imp).mark_bar(size=14, cornerRadiusEnd=4).encode(
        x=alt.X("share:Q", title=None, axis=alt.Axis(format="%")),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color("group:N", scale=GROUP_COLORS, legend=alt.Legend(orient="top", title=None)),
        tooltip=["label:N", "group:N", alt.Tooltip("share:Q", format=".1%")]).properties(height=320), width="stretch")

with tab_explore:
    scores, preds = evaluation(today)
    races = preds[["season", "round", "race_name"]].drop_duplicates().sort_values(["season", "round"], ascending=False)
    season, rnd, _ = st.selectbox("Race", races.itertuples(index=False), format_func=lambda r: f"{r.season} R{r.round} · {r.race_name}")
    sel = preds[(preds["season"] == season) & (preds["round"] == rnd)].sort_values("pos")
    for col, method in zip(st.columns(3), model.METHODS):
        s = scores[(scores["season"] == season) & (scores["round"] == rnd) & (scores["method"] == method)].iloc[0]
        col.metric(method, f"{s['spearman']:.2f}", help="Rank correlation with the actual finishers")
    st.dataframe(sel.assign(pos=sel["pos"].where(sel["finished"])), hide_index=True,
                 column_order=["pos", "name", "team", "grid", "pred_pre", "pred_grid"], column_config={
                     "pos": st.column_config.NumberColumn("Finished", help="Blank = did not finish"),
                     "name": "Driver", "team": "Team", "grid": "Grid",
                     "pred_pre": "Predicted (before qualifying)", "pred_grid": "Predicted (with grid)"})

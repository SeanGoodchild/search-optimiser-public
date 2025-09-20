
import streamlit as st
from services import store, data
from ui.charts import chart_with_events


st.set_page_config(page_title="PMG Budget Optimizer", layout="wide")
st.title("PMG Budget Optimizer")
st.markdown("Select custom points on the curves, or allow the optimizer to choose for you.")
store.init()


def main():
    data.import_and_init('data/sample.csv')
    build_curves_section()
    build_optimization_section()


def build_curves_section():
    events = {}
    strategies = store.get_all_strategies()
    for strategy_id, strategy in strategies.items():
        st.subheader(strategy['name'])
        events[strategy_id] = chart_with_events(strategy)
    handle_events(events)


def handle_events(events: dict) -> None:
    for strategy_id, event in events.items():
        if not event:
            continue
        clicked_x = float(event[0]['x'])
        clicked_y = float(event[0]['y'])

        (current_x, current_y) = store.get_strategy(strategy_id)['selected_point']
        if clicked_x != current_x or clicked_y != current_y:
            store.set_strategy(strategy_id, {'selected_point': (clicked_x, clicked_y)})
            st.toast(f"Selected cost £{clicked_x:.2f} with {clicked_y:.2f}")
            st.rerun()


def build_optimization_section():
    st.header("Optimize")


if __name__ == "__main__":
    main()

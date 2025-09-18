
# Budget Optimizer (Streamlit)

A scaffolded Streamlit app for interactive bid-curve exploration and optimization.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

- `domain/`: pure models and math (frontier building, pruning).
- `services/`: IO and optimization (OR-Tools hook; greedy fallback here).
- `ui/`: charts and widgets.
- `state/`: session state wrapper.
- `pages/`: multipage Streamlit UI.
- `data/`: sample data.

## Notes

- Click points on a chart (Plotly) to select for an entity.
- Use the increment selector to allow continuous cost choices densified to a sensible step (1,2,5,10,20,50,100,... scaled to range).
- Optimization discretizes continuous choices using your selected step.

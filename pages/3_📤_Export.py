
import streamlit as st
from state import store
from services.export import result_to_csv

store.init()
st.header("Export")

res = store.get_result()
if not res:
    st.info("No results yet. Run optimization first.")
else:
    data = result_to_csv(res)
    st.download_button("Download CSV", data=data, file_name="optimization_result.csv", mime="text/csv")

import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling as pp
import streamlit as st
import typeconv as tp

df = pd.read_csv("D:\DK\Dev\diabetes\diabetes_dataset.csv")

df1 = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])

profile = pp.ProfileReport(df, title = "Pandas profiling report")
print(df.head(5))

st_profile_report(profile)

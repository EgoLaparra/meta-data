import sys
import pandas as pd

months = pd.read_csv(sys.argv[1])
years = pd.read_csv(sys.argv[2])

dates = pd.concat([months, years], axis=1)
dates = dates.apply(lambda x: pd.Series([x["fileid"][0], x["filename"][0], " ".join(x["DATE"].astype(str))]), axis=1)
dates = dates.rename(columns={0: "fileid", 1: "filename", 2: "DATE"})
dates.to_csv(sys.argv[3], index=False)

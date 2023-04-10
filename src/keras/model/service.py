import bentoml
from bentoml.io import NumpyNdarray, PandasDataFrame
import numpy as np
import pandas as pd
import tensorflow as tf


runner = bentoml.tensorflow.get("modeltest:latest").to_runner()
svc = bentoml.Service("modeltest", runners=[runner])

@svc.api(input=PandasDataFrame(orient="records"),
        output=NumpyNdarray(dtype="float32"))

async def predict(df):
        # Optional pre-processing, post-processing code goes here
        print(df)
        return await runner.async_run(df)


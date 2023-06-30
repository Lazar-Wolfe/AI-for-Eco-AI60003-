# ---------------The following code is WIP--------------
# |This is not supposed to work with the existing files|
# ------------------------------------------------------

import time
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, GRU, Dropout, Dense, Activation
import datetime as dt
import pandas as pd
from binance.client import Client

client = Client("***")

intervals = [
    Client.KLINE_INTERVAL_1DAY,
    Client.KLINE_INTERVAL_5MINUTE,
    Client.KLINE_INTERVAL_15MINUTE,
]
sleeptime = [60, 300, 60 * 15]
leverage = [1, 5, 15]
mode = int(sys.argv[1])
columns = [
    "dateTime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeTime",
    "quoteAssetVolume",
    "numberOfTrades",
    "takerBuyBaseVol",
    "takerBuyQuoteVol",
    "ignore",
]

SEQ_LEN = 100
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

filename = sys.argv[2]


with open(filename, "w") as f:
    f.write("")


def to_float(x):
    try:
        return float(x)
    except ValueError:
        return float("NaN")


data = client.get_historical_klines("BTCUSDT", intervals[mode])
df = pd.DataFrame(
    data,
    columns=columns,
)
df.dateTime = pd.to_datetime(df.dateTime, unit="ms").dt.strftime("%Y-%m-%d %H-%M-%S")
df.set_index("dateTime", inplace=True)

df = df.drop(
    [
        "closeTime",
        "quoteAssetVolume",
        "numberOfTrades",
        "takerBuyBaseVol",
        "takerBuyQuoteVol",
        "ignore",
    ],
    axis=1,
)
close_price = df.close.values.reshape(-1, 1)

interpreter = tf.lite.Interpreter("model_v2.tflite")
interpreter.allocate_tensors()
the_predictor = interpreter.get_signature_runner()

owned_btc, owned_usd, delta = 5000 / float(close_price[-1]), 5000.0, 0.0
buy_usd, sell_btc = 0.0, 0.0
scaler = MinMaxScaler()
while True:
    # Square OFF
    data = client.get_historical_klines("BTCUSDT", intervals[mode], limit=99)
    df = pd.DataFrame(
        data,
        columns=columns,
    )
    df.set_index("dateTime", inplace=True)
    df = df.drop(
        [
            "closeTime",
            "quoteAssetVolume",
            "numberOfTrades",
            "takerBuyBaseVol",
            "takerBuyQuoteVol",
            "ignore",
        ],
        axis=1,
    )

    df = df.applymap(to_float)

    scaled_prices = (df - df.min()) / (df.max() - df.min())
    scaled_prices_arr = np.float32(scaled_prices.values)[np.newaxis, ...]

    res = the_predictor(bidirectional_6_input=scaled_prices_arr)["activation_2"].argmax()
    current_price = float(df.close.values[-1])

    owned_usd = owned_usd - (current_price * sell_btc) + buy_usd
    owned_btc = owned_btc + sell_btc - (buy_usd / current_price)
    sell_btc = 0
    buy_usd = 0
    delta = 0.1 if (res == 1) else -0.1

    if delta < 0:
        sell_btc = (-delta) * owned_btc
        owned_btc -= sell_btc
        owned_usd += sell_btc * current_price
    else:
        buy_usd = delta * owned_usd
        owned_usd -= buy_usd
        owned_btc += buy_usd / current_price
    with open(filename, "a") as f:
        s = "{},{},{},{},{},{}\n".format(
            time.time(),
            sell_btc,
            buy_usd,
            owned_btc,
            owned_usd,
            owned_btc * current_price + owned_usd,
        )
        f.write(s)
        print(s)
    time.sleep(sleeptime[mode])

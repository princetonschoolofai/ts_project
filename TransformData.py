import datetime as dt
import ta
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
#Custom Transformer that extracts columns passed as argument to its constructor

class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X,y = None):
        self.columns_ = X.columns
        self.index_ = X.index
        return self

    def transform(self, X, y = None):
        etf_ta = normalize(X,axis = 0)
        etf_ta = pd.DataFrame(etf_ta, index = self.index_, columns = self.columns_)
        return etf_ta

class PCA_(BaseEstimator, TransformerMixin):
    def __init__(self, desired_percentage):
        self.percentage = desired_percentage
        return None

    def fit(self, X, y = None):
        self.index_ = X.index
        pca_ = PCA().fit(X)
        # Let's count the number of principal components that will account
        # for 95% of the variance of the Data
        self.count = 0
        variance = 0
        for value in pca_.explained_variance_ratio_ :
            if variance <self.percentage:
                variance += value
                self.count += 1
            else:
                break

        return self

    def transform(self, X, y = None):
        # Now let's do a Principal Component analysis
        etf_ta = PCA(self.count).fit_transform(X)
        etf_ta = pd.DataFrame(etf_ta, index = self.index_)

        return etf_ta

class GetTA( BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self ):
        return None

    #Return self nothing else to do here
    def fit( self, X, y = None ):

        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        etf_ta = ta.add_all_ta_features(X, "open", "high", "low", "close", "volume", fillna=True)
        # Technical Analysis (TA) Indicators
        # https://technical-analysis-library-in-python.readthedocs.io/en/latest/
        #
        # * The library has implemented 31 indicators:
        #
        # Volume
        #
        # Accumulation/Distribution Index (ADI)
        # On-Balance Volume (OBV)
        # Chaikin Money Flow (CMF)
        # Force Index (FI)
        # Ease of Movement (EoM, EMV)
        # Volume-price Trend (VPT)
        # Negative Volume Index (NVI)
        # Volatility
        #
        # Average True Range (ATR)
        # Bollinger Bands (BB)
        # Keltner Channel (KC)
        # Donchian Channel (DC)
        # Trend
        #
        # Moving Average Convergence Divergence (MACD)
        # Average Directional Movement Index (ADX)
        # Vortex Indicator (VI)
        # Trix (TRIX)
        # Mass Index (MI)
        # Commodity Channel Index (CCI)
        # Detrended Price Oscillator (DPO)
        # KST Oscillator (KST)
        # Ichimoku Kinkō Hyō (Ichimoku)
        # Momentum
        #
        # Money Flow Index (MFI)
        # Relative Strength Index (RSI)
        # True strength index (TSI)
        # Ultimate Oscillator (UO)
        # Stochastic Oscillator (SR)
        # Williams %R (WR)
        # Awesome Oscillator (AO)
        # Others
        #
        # Daily Return (DR)
        # Daily Log Return (DLR)
        # Cumulative Return (CR)


        return etf_ta

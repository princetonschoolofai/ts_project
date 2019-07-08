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
        OriginalData = X[['open', 'high', 'low', 'close', 'adjusted_close',
        'volume','dividend_amount', 'split_coefficient']]

        Volume = X[['volume_adi', 'volume_obv','volume_cmf', 'volume_fi',
        'volume_em', 'volume_vpt', 'volume_nvi']]

        Volatility = X[['volatility_atr', 'volatility_bbh', 'volatility_bbl',
        'volatility_bbm',
        'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
        'volatility_kch', 'volatility_kcl', 'volatility_kchi',
        'volatility_kcli', 'volatility_dch', 'volatility_dcl',
        'volatility_dchi', 'volatility_dcli']]

        Trend = X[['trend_macd', 'trend_macd_signal',
        'trend_macd_diff', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
        'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
        'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_trix',
        'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
        'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a',
        'trend_ichimoku_b', 'trend_visual_ichimoku_a',
        'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down',
        'trend_aroon_ind']]

        Momentum = X[['momentum_rsi', 'momentum_mfi', 'momentum_tsi',
        'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
        'momentum_ao']]

        Others = X[['others_dr', 'others_dlr', 'others_cr']]

        self.index_ = X.index
        self.number_of_col = []
        self.dataName = ['OriginalData', 'Volume', 'Volatility', 'Trend', 'Momentum', 'Others']

        for database in [OriginalData, Volume, Volatility, Trend, Momentum, Others]:

            pca_ = PCA().fit(database)
            self.count = 0
            variance = 0
            for value in pca_.explained_variance_ratio_ :

                if variance <self.percentage:
                    variance += value
                    self.count += 1
                else:
                    self.number_of_col.append(self.count)
                    break
        return self

    def transform(self, X, y = None):
        OriginalData = X[['open', 'high', 'low', 'close', 'adjusted_close', 'volume',
        'dividend_amount', 'split_coefficient']]

        Volume = X[['volume_adi', 'volume_obv',
        'volume_cmf', 'volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi']]

        Volatility = X[['volatility_atr', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm',
        'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
        'volatility_kch', 'volatility_kcl', 'volatility_kchi',
        'volatility_kcli', 'volatility_dch', 'volatility_dcl',
        'volatility_dchi', 'volatility_dcli']]

        Trend = X[['trend_macd', 'trend_macd_signal',
        'trend_macd_diff', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
        'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
        'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_trix',
        'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
        'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a',
        'trend_ichimoku_b', 'trend_visual_ichimoku_a',
        'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down',
        'trend_aroon_ind']]

        Momentum = X[['momentum_rsi', 'momentum_mfi', 'momentum_tsi',
        'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
        'momentum_ao']]

        Others = X[['others_dr', 'others_dlr', 'others_cr']]

        count_ = 0

        final_data = pd.DataFrame(index = self.index_)
        for database in [OriginalData, Volume, Volatility, Trend, Momentum, Others]:
            indexCol = self.number_of_col[count_]
            colnames = []
            for i in range(indexCol):
                colnames.append(self.dataName[count_]+'_'+str(i))

            temp = PCA(indexCol).fit_transform(database)
            final_data = pd.concat([final_data,pd.DataFrame(temp, index = self.index_, columns = colnames)], axis = 1)
            count_ += 1
        return final_data

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

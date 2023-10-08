import logging

import numpy as np
import pandas as pd
from hamilton import base, driver, log_setup
from hamilton.experimental import h_ray
from hamilton.experimental import h_cache
from hydra import initialize, compose
import ray
from ray import workflow

from ats.app.env_mgr import EnvMgr
from ats.features.preprocess.test_utils import run_features
from ats.market_data import market_data_mgr
from ats.util import logging_utils


def test_close_low_5_ff():
    result = run_features("close_low_5_ff", 50)
    print(f"result:{result['close_low_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_low_5_ff"],
        [940.25, 940.25, 941.75, 942.75, 943.5, 944.5, 946.75, 946.75, 946.0, 945.75,
         945.75, 945.75, 945.75, 945.25, 945.25, 944.75, 944.75, 944.75, 944.75,
         944.75, 944.5, 944.5, 944.5, 944.5, 944.5, 945.25, 945.5, 945.75, 945.75,
         945.75, 945.75, 947.0, 947.25, 945.75, 945.75, 945.75, 945.75, 945.75,
         949.25, 953.75, 953.75, 953.75, 953.75, 953.75, 954.5, 955.0, 955.5, 957.5, 957.5, 957.5],
        decimal=3
    )

def test_close_high_5_ff():
    result = run_features("close_high_5_ff", 50)
    print(f"result:{result['close_high_5_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_high_5_ff"],
        [943.5, 944.5, 947.0, 947.75, 947.75, 947.75, 947.75, 947.75, 947.0, 946.75,
         947.25, 947.25, 947.25, 947.25, 947.25, 946.75, 946.0, 945.75, 945.75, 945.5,
         945.5, 945.5, 945.5, 945.75, 946.0, 947.5, 947.5, 947.5, 948.25, 948.25, 949.0,
         949.0, 949.0, 949.0, 949.25, 959.25, 959.25, 961.5, 961.5, 961.5, 961.5, 961.5,
         955.5, 958.25, 958.25, 960.0, 960.0, 960.0, 960.0, 961.5],
        decimal=3
    )

def test_close_low_21_ff():
    result = run_features("close_low_21_ff", 50)
    print(f"result:{result['close_low_21_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_low_21_ff"],
        [936.5, 937.75, 937.75, 937.75, 937.75, 939.0, 939.0, 939.0, 939.5, 939.5,
         940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 940.25, 941.75,
         942.75, 943.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5,
         944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5, 944.5,
         944.5, 944.5, 945.25, 945.5, 945.75, 945.75, 945.75, 945.75, 945.75, 945.75, 945.75],
        decimal=3
    )

def test_close_high_21_ff():
    result = run_features("close_high_21_ff", 50)
    print(f"result:{result['close_high_21_ff'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close_high_21_ff"],
        [943.5, 944.5, 947.0, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75,
         947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75, 947.75,
         947.75, 947.75, 947.75, 947.75, 947.75, 947.25, 947.5, 947.5, 947.5,
         948.25, 948.25, 949.0, 949.0, 949.0, 949.0, 949.25, 959.25, 959.25, 961.5,
         961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5, 961.5],
        decimal=3
    )

def test_close_feature():
    result = run_features("close", 50)[10:15]
    print(f"result:{result['close'].to_list()}")
    np.testing.assert_array_almost_equal(
        result["close"],
        [947.25, 946.75, 946.  , 945.25, 945.75],
        decimal=3
    )

def test_joined_last_daily_close_0():
    result = run_features("joined_last_daily_close_0", 200)[10:15]
    logging.error(f"result:{result}")
    close_list = result['joined_last_daily_close_0']
    close_time_list = result.index.get_level_values(level=0)
    np.testing.assert_array_almost_equal(
        close_list,
        [902.5, 902.5, 902.5, 902.5, 902.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283202000, 1283205600, 1283207400, 1283209200, 1283211000],
        decimal=3
    )
    
def test_last_daily_close_0():
    result = run_features("last_daily_close_0", 50)[:10]
    close_list = result['last_daily_close_0']
    close_time_list = result.index.get_level_values(level=0)
    ticker_list = result.index.get_level_values(level=1)
    np.testing.assert_array_almost_equal(
        close_list,
        [932.5 , 928.25, 892.75, 882.25, 879.75, 872.  , 881.75, 917.  ,
         924.75, 929.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200, 1278705600],
        decimal=3
    )

def test_last_daily_close_1():
    result = run_features("last_daily_close_1", 50)[:10]
    logging.error(f"result:{result}")
    open_list = result['last_daily_close_1']
    close_time_list = result.index.get_level_values(level=0)
    np.testing.assert_array_almost_equal(
        open_list,
        [928.  , 932.5 , 928.25, 892.75, 882.25, 879.75, 872.  , 881.75,
         917.  , 924.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200, 1278705600],
        decimal=3
    )

def test_last_daily_close_2():
    result = run_features("last_daily_close_2", 50)[:10]
    open_list = result['last_daily_close_2']
    close_time_list = result.index.get_level_values(level=0)
    np.testing.assert_array_almost_equal(
        open_list,
        [944.75, 928.  , 932.5 , 928.25, 892.75, 882.25, 879.75, 872.  ,
         881.75, 917],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200, 1278705600],
        decimal=3
    )

def test_daily_close_df():
    result = run_features("daily_close_df", 50)[:10]
    logging.error(f"result:{result}")
    daily_close_0_list = result['daily_close_0']
    daily_close_1_list = result['daily_close_1']
    daily_close_2_list = result['daily_close_2']
    close_time_list = result.index.get_level_values(level=0)
    ticker_list = result.index.get_level_values(level=1)
    np.testing.assert_array_almost_equal(
        daily_close_0_list,
        [932.5 , 928.25, 892.75, 882.25, 879.75, 872.  , 881.75, 917.  ,
         924.75, 929.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        daily_close_1_list,
        [928.  , 932.5 , 928.25, 892.75, 882.25, 879.75, 872.  , 881.75,
         917.  , 924.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        daily_close_2_list,
        [944.75, 928.  , 932.5 , 928.25, 892.75, 882.25, 879.75, 872.  ,
         881.75, 917.],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200, 1278705600],
        decimal=3
    )

def test_daily_open():
    result = run_features("daily_open", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_open']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [928.  , 932.75, 928.25, 893.  , 882.25, 880.  , 872.75, 881.75, 916.5], 
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )


def test_daily_close_series():
    result = run_features("daily_close", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_close']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [932.5, 928.25, 892.75, 882.25, 879.75, 872.0, 881.75, 917.0, 924.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_high_series():
    result = run_features("daily_high", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_high']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [937.25, 937.5 , 932.5 , 902.25, 887.5 , 890.25, 896.25, 917.5 ,
         926.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_daily_low_series():
    result = run_features("daily_low", 50)
    close_list = result.query('(close_time>=1277409600) and (close_time<=1278619200)')['daily_low']
    close_time_list = close_list.index.get_level_values(level=0)[:10].to_list()
    np.testing.assert_array_almost_equal(
        close_list,
        [920.5 , 924.25, 888.  , 880.75, 863.75, 868.5 , 860.5 , 874.  ,
         911.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1277496000, 1277755200, 1277841600, 1277928000, 1278014400,
         1278100800, 1278446400, 1278532800, 1278619200],
        decimal=3
    )

def test_weekly_open():
    result = run_features("weekly_open", 50)
    open_list = result['weekly_open'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [903.5 , 888.  , 866.  , 912.  , 924.5 , 920.75, 875.5 , 910.  ,
         936.25, 934.],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_close():
    result = run_features("weekly_close", 50)
    close_list = result['weekly_close'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    # TODO: there seems to be a bug with first Friday.
    # 2009-10-02 15:30:00  865.75  868.00  863.25  865.25  161658  1.399255e+08   
    # 2009-10-02 16:00:00  865.25  866.75  864.25  866.25   90128  7.802734e+07
    # 15:30:00 is treated as last instead of 16:00:00.
    np.testing.assert_array_almost_equal(
        close_list,
        [885.5 , 865.25, 912.5 , 926.5 , 921.  , 876.5 , 910.5 , 935.75,
         934.5 , 933.5],
        decimal=3
    )
    # 2009-09-25 16:00:00
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_high():
    result = run_features("weekly_high", 50)
    high_list = result['weekly_high'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        high_list,
        [920.  , 910.  , 912.75, 939.75, 943.25, 932.75, 913.75, 947.5 ,
         956.5 , 955.75],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_weekly_low():
    result = run_features("weekly_low", 50)
    low_list = result['weekly_low'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        low_list,
        [880.5 , 856.25, 863.75, 907.25, 914.5 , 873.75, 870.25, 909.25,
         927.75, 911.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1253908800, 1254513600, 1255118400, 1255723200, 1256328000,
         1256932800, 1257541200, 1258146000, 1258750800, 1259344800],
        decimal=3
    )

def test_monthly_open():
    result = run_features("monthly_open", 50)
    open_list = result['monthly_open'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        open_list,
        [1173.25, 1156.75, 1218.5 , 1236.25, 1116.25, 1099.75, 1115.25,
         999.5 ,  797.75,  729.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1206993600, 1209585600, 1212177600, 1214856000, 1217534400,
         1220040000, 1222804800, 1225483200, 1227895200, 1230757200],
        decimal=3
    )

def test_monthly_close():
    result = run_features("monthly_close", 50)
    close_list = result['monthly_close'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        close_list,
        [1157.25, 1220.  , 1236.25, 1116.25, 1099.75, 1116.75, 1000.  ,
         798.5 ,  726.75,  724.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1206993600, 1209585600, 1212177600, 1214856000, 1217534400,
         1220040000, 1222804800, 1225483200, 1227895200, 1230757200],
        decimal=3
    )

def test_monthly_high():
    result = run_features("monthly_high", 50)
    high_list = result['monthly_high'][:10]
    close_time_list = result.index.get_level_values(level=0)[:10]
    ticker_list = result.index.get_level_values(level=1)[:10]
    np.testing.assert_array_almost_equal(
        high_list,
        [1176.  , 1242.5 , 1276.5 , 1246.5 , 1128.  , 1146.75, 1136.75,
         1006.  ,  840.5 ,  751.25],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1206993600, 1209585600, 1212177600, 1214856000, 1217534400,
         1220040000, 1222804800, 1225483200, 1227895200, 1230757200],
        decimal=3
    )

def test_monthly_low():
    result = run_features("monthly_low", 50)[:10]
    low_list = result['monthly_low']
    close_time_list = result.index.get_level_values(level=0)
    np.testing.assert_array_almost_equal(
        low_list,
        [1145.  , 1151.25, 1206.  , 1106.  , 1034.  , 1079.5 ,  944.  ,
         657.  ,  571.  ,  645.],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1206993600, 1209585600, 1212177600, 1214856000, 1217534400,
         1220040000, 1222804800, 1225483200, 1227895200, 1230757200],
        decimal=3
    )

def test_close_low_1d_ff_shift_1d():
    result = run_features("close_low_1d_ff_shift_1d", 50)[:4]
    close = result["close_low_1d_ff_shift_1d"]
    close_time_list = result.index.get_level_values(level=0)
    np.testing.assert_array_almost_equal(
        close,
        [944.5, 944.5, 944.5, 944.5],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1283452200, 1283454000, 1283455800, 1283457600],
        decimal=3
    )
    
def test_close_low_1d_ff():
    result = run_features("close_low_1d_ff", 50)[:4]
    close = result["close_low_1d_ff"]
    np.testing.assert_array_almost_equal(
        close,
        [934.5, 934.5, 934.5, 934.5],
        decimal=3
    )

def test_kc_10d_05_high():
    result = run_features("kc_10d_05_high", 50)
    logging.error(f"result:{result}")
    close = result["kc_10d_05_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [981.131, 982.062, 977.389, 971.721, 966.409],
        decimal=3
    )

def test_kc_10d_10_high():
    result = run_features("kc_10d_10_high", 50)
    close = result["kc_10d_10_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [990.44 , 991.315, 987.479, 981.652, 976.172],
        decimal=3
    )

def test_kc_10d_20_high():
    result = run_features("kc_10d_20_high", 50)
    close = result["kc_10d_20_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1009.058, 1009.821, 1007.659, 1001.514,  995.698],
        decimal=3
    )

def test_kc_10d_15_high():
    result = run_features("kc_10d_15_high", 50)
    close = result["kc_10d_15_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [999.749, 1000.568,  997.569,  991.583,  985.935],
        decimal=3
    )

def test_kc_20d_10_high():
    result = run_features("kc_20d_10_high", 50)
    close = result["kc_20d_10_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [978.913, 980.416, 980.242, 977.728, 975.071],
        decimal=3
    )

def test_kc_10w_05_high():
    result = run_features("kc_10w_05_high", 500)
    logging.error(f"result:{result}")
    close = result["kc_10w_05_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [966.239, 948.058, 925.703, 895.169, 857.538],
        decimal=3
    )

def test_kc_10w_10_high():
    result = run_features("kc_10w_10_high", 500)
    close = result["kc_10w_10_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1019.612, 1004.044,  981.665,  952.822,  916.564],
        decimal=3
    )

def test_kc_10w_15_high():
    result = run_features("kc_10w_15_high", 500)
    close = result["kc_10w_15_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1072.985, 1060.029, 1037.627, 1010.476,  975.589],
        decimal=3
    )

def test_kc_10w_20_high():
    result = run_features("kc_10w_20_high", 500)
    close = result["kc_10w_20_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1126.358, 1116.015, 1093.589, 1068.129, 1034.615],
        decimal=3
    )

def test_kc_20w_10_high():
    result = run_features("kc_20w_10_high", 500)
    close = result["kc_20w_10_high"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1110.522, 1096.197, 1075.676, 1053.225, 1025.977],
        decimal=3
    )

def test_kc_10m_05_high():
    result = run_features("kc_10m_05_high", 500)
    close = result["kc_10m_05_high"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [969.107, 908.692, 871.54 , 851.795, 843.759],
        decimal=3
    )

def test_kc_10m_10_high():
    result = run_features("kc_10m_10_high", 500)
    close = result["kc_10m_10_high"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [1039.91 ,  979.589,  943.747,  922.182,  910.457],
        decimal=3
    )

def test_kc_10m_15_high():
    result = run_features("kc_10m_15_high", 500)
    close = result["kc_10m_15_high"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [1110.712, 1050.487, 1015.955,  992.568,  977.155],
        decimal=3
    )

def test_kc_10m_20_high():
    result = run_features("kc_10m_20_high", 500)
    close = result["kc_10m_20_high"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [1181.515, 1121.384, 1088.162, 1062.955, 1043.854],
        decimal=3
    )

def test_kc_10d_05_low():
    result = run_features("kc_10d_05_low", 50)
    close = result["kc_10d_05_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [962.513, 963.556, 957.208, 951.858, 946.883],
        decimal=3
    )

def test_kc_10d_10_low():
    result = run_features("kc_10d_10_low", 50)
    close = result["kc_10d_10_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [953.204, 954.303, 947.118, 941.927, 937.12],
        decimal=3
    )

def test_kc_10d_15_low():
    result = run_features("kc_10d_15_low", 50)
    close = result["kc_10d_15_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [943.895, 945.05 , 937.027, 931.996, 927.357],
        decimal=3
    )

def test_kc_10d_20_low():
    result = run_features("kc_10d_20_low", 50)
    close = result["kc_10d_20_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [934.586, 935.797, 926.937, 922.065, 917.593],
        decimal=3
    )


def test_kc_20d_10_low():
    result = run_features("kc_20d_10_low", 50)
    close = result["kc_20d_10_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [941.678, 943.404, 939.881, 938.003, 936.019],
        decimal=3
    )

def test_kc_10w_05_low():
    result = run_features("kc_10w_05_low", 500)
    close = result["kc_10w_05_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [859.494, 836.087, 813.779, 779.862, 739.487],
        decimal=3
    )

def test_kc_10w_10_low():
    result = run_features("kc_10w_10_low", 500)
    close = result["kc_10w_10_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [806.121, 780.101, 757.817, 722.209, 680.462],
        decimal=3
    )

def test_kc_10w_15_low():
    result = run_features("kc_10w_15_low", 500)
    close = result["kc_10w_15_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [752.748, 724.116, 701.855, 664.555, 621.436],
        decimal=3
    )

def test_kc_10w_20_low():
    result = run_features("kc_10w_20_low", 500)
    close = result["kc_10w_20_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [699.375, 668.13 , 645.893, 606.902, 562.411],
        decimal=3
    )

def test_kc_20w_10_low():
    result = run_features("kc_20w_10_low", 500)
    close = result["kc_20w_10_low"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [897.03 , 872.254, 851.828, 822.612, 789.875],
        decimal=3
    )

def test_kc_10m_05_low():
    result = run_features("kc_10m_05_low", 500)
    logging.error(f"result:{result}")
    close = result["kc_10m_05_low"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [827.502, 766.898, 727.125, 711.021, 710.363],
        decimal=3
    )

def test_kc_10m_10_low():
    result = run_features("kc_10m_10_low", 500)
    close = result["kc_10m_10_low"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [756.7  , 696.   , 654.917, 640.635, 643.665],
        decimal=3
    )

def test_kc_10m_15_low():
    result = run_features("kc_10m_15_low", 500)
    close = result["kc_10m_15_low"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [685.897, 625.103, 582.709, 570.248, 576.967],
        decimal=3
    )

def test_kc_10m_20_low():
    result = run_features("kc_10m_20_low", 500)
    close = result["kc_10m_20_low"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [615.095, 554.206, 510.502, 499.861, 510.269],
        decimal=3
    )

def test_kc_10d_05_mid():
    result = run_features("kc_10d_05_mid", 50)
    close = result["kc_10d_05_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [971.822, 972.809, 967.298, 961.789, 956.646],
        decimal=3
    )

def test_kc_10d_10_mid():
    result = run_features("kc_10d_10_mid", 50)
    close = result["kc_10d_10_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [971.822, 972.809, 967.298, 961.789, 956.646],
        decimal=3
    )


def test_kc_20d_05_mid():
    result = run_features("kc_20d_05_mid", 50)
    close = result["kc_20d_05_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [960.296, 961.91 , 960.062, 957.865, 955.545],
        decimal=3
    )

def test_kc_20d_10_mid():
    result = run_features("kc_20d_10_mid", 50)
    close = result["kc_20d_10_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [960.296, 961.91 , 960.062, 957.865, 955.545],
        decimal=3
    )

def test_kc_10w_05_mid():
    result = run_features("kc_10w_05_mid", 500)
    close = result["kc_10w_05_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [912.866, 892.073, 869.741, 837.516, 798.513],
        decimal=3
    )

def test_kc_10w_10_mid():
    result = run_features("kc_10w_10_mid", 500)
    close = result["kc_10w_10_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [912.866, 892.073, 869.741, 837.516, 798.513],
        decimal=3
    )

def test_kc_20w_10_mid():
    result = run_features("kc_20w_10_mid", 500)
    close = result["kc_20w_10_mid"][30:35]
    np.testing.assert_array_almost_equal(
        close,
        [1003.776,  984.226,  963.752,  937.918,  907.926],
        decimal=3
    )

def test_kc_10m_05_mid():
    result = run_features("kc_10m_05_mid", 500)
    close = result["kc_10m_05_mid"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [898.305, 837.795, 799.332, 781.408, 777.061],
        decimal=3
    )

def test_kc_10m_10_mid():
    result = run_features("kc_10m_10_mid", 500)
    close = result["kc_10m_10_mid"][10:15]
    np.testing.assert_array_almost_equal(
        close,
        [898.305, 837.795, 799.332, 781.408, 777.061],
        decimal=3
    )

def test_vwap_around_london_close_20230411():
    timestamp = 1681192800
    result = run_features("vwap_around_london_close", 100000, timestamp)
    close_list = result.query(f"(timestamp>={timestamp-7200}) and (timestamp<={timestamp+7200})")['vwap_around_london_close']
    print(f"close_list:{close_list}")
    close_time_list = close_list.index.get_level_values(level=0).to_list()
    np.testing.assert_array_almost_equal(
        close_list.to_list(),
        [4126.026, 4126.026, 4126.026, 4126.026, 4126.026, 4126.026,
         4126.026, 4126.026, 4126.026],
        decimal=3
    )
    np.testing.assert_array_almost_equal(
        close_time_list,
        [1681185600, 1681187400, 1681189200, 1681191000, 1681192800,
         1681194600, 1681196400, 1681198200, 1681200000],
        decimal=3
    )

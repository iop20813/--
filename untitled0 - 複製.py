
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

db_name = 'traffic_accident.db'

def load_data_to_db(year):
    """讀取 CSV 並存入 SQLite"""
    df = pd.read_csv(f'{year}整理.csv')
    
    # 清理欄位名稱
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\s+", "_", regex=True)

    with sqlite3.connect(db_name) as con:
        df.to_sql(f'traffic_accident_{year}', con, if_exists='replace', index=False)
    
    print(df.columns)  # 確認欄位名稱

def get_total_deaths(year):
    """查詢特定年份的總死亡人數"""
    with sqlite3.connect(db_name) as con:
        query = f"SELECT SUM(死亡人數) FROM traffic_accident_{year}"
        return con.execute(query).fetchone()[0] or 0

def plot_death_trend(years):
    """繪製各年度死亡人數趨勢圖"""
    deaths = [get_total_deaths(year) for year in years]
    plt.figure(figsize=(10, 5))
    plt.bar(years, deaths, color='skyblue')
    plt.xlabel('年度')
    plt.ylabel('死亡人數')
    plt.title('各年度交通事故死亡人數')
    plt.show()

def analyze_top_causes(year, threshold=100):
    """找出造成死亡人數最多的車種"""
    with sqlite3.connect(db_name) as con:
        query = f"""
            SELECT "車種", SUM(死亡人數) AS death_count
            FROM traffic_accident_{year}
            GROUP BY "車種"
            HAVING death_count > {threshold}
            ORDER BY death_count DESC
        """
        return pd.read_sql(query, con)

# 主要執行邏輯
years = [107, 108, 109, 110]
for year in years:
    load_data_to_db(year)

plot_death_trend(years)
for year in years:
    print(f'年度 {year} 主要肇因分析:')
    print(analyze_top_causes(year))




# def load_data(year):
#     """從 SQLite 載入資料"""
#     with sqlite3.connect(db_name) as con:
#         query = f"SELECT * FROM traffic_accident_{year}"
#         df = pd.read_sql(query, con)
#     return df

# # 讀取多年度資料
# df_list = [load_data(year) for year in years]
# df = pd.concat(df_list, ignore_index=True)

# # 清理欄位名稱
# df.columns = df.columns.str.strip()
# df.columns = df.columns.str.replace("\s+", "_", regex=True)

# # 處理發生時間（轉換為小時、月份等特徵）
# df['發生時間'] = pd.to_datetime(df['發生時間'], errors='coerce')
# df['小時'] = df['發生時間'].dt.hour
# df['月份'] = df['發生時間'].dt.month

# # 處理類別型特徵（車種）
# label_encoder = LabelEncoder()
# df['車種'] = label_encoder.fit_transform(df['車種'])

# # 移除 NaN
# df = df.dropna(subset=['死亡人數'])

# # 設定特徵與標籤
# X = df[['車種', '小時', '月份']]
# y = df['死亡人數']

# # 分割訓練集與測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 建立 XGBoost 模型
# model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# # 設定超參數範圍
# param_grid = {
#     'n_estimators': [100, 200, 300], 
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.7, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2]
# }

# # 超參數調優
# grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)

# # 最佳參數
# best_params = grid_search.best_params_
# print("最佳參數:", best_params)

# # 重新訓練最佳模型
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# # 評估模型
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f"RMSE: {rmse:.2f}")

# # 視覺化預測結果
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label="預測 vs 實際")
# plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="理想值")
# plt.xlabel("實際死亡人數")
# plt.ylabel("預測死亡人數")
# plt.title("XGBoost 預測結果")
# plt.legend()
# plt.show()


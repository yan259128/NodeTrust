import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子
random.seed(42)
np.random.seed(42)


def generate_china_premium_traceability(num_records=2000):
    # 1. 产地库：15省，每省3-5个真实县名及经纬度
    china_origin_counties = [
        # 东北
        {"prov": "黑龙江省", "county": "五常市", "lon": 127.15, "lat": 44.93, "crops": ["水稻", "大豆"]},
        {"prov": "黑龙江省", "county": "富锦市", "lon": 132.03, "lat": 47.25, "crops": ["玉米", "大豆"]},
        {"prov": "黑龙江省", "county": "嫩江市", "lon": 125.22, "lat": 49.17, "crops": ["大豆", "小麦"]},
        {"prov": "吉林省", "county": "榆树市", "lon": 126.55, "lat": 44.82, "crops": ["玉米", "水稻"]},
        {"prov": "吉林省", "county": "公主岭市", "lon": 124.82, "lat": 43.52, "crops": ["玉米", "大豆"]},
        {"prov": "吉林省", "county": "前郭县", "lon": 124.81, "lat": 45.11, "crops": ["水稻", "大麦"]},
        {"prov": "辽宁省", "county": "昌图县", "lon": 124.11, "lat": 42.78, "crops": ["玉米", "大豆"]},
        {"prov": "辽宁省", "county": "盘山县", "lon": 122.04, "lat": 41.24, "crops": ["水稻"]},
        {"prov": "辽宁省", "county": "黑山县", "lon": 122.12, "lat": 41.69, "crops": ["花生", "玉米"]},
        # 西北
        {"prov": "新疆维吾尔自治区", "county": "沙雅县", "lon": 82.78, "lat": 41.22, "crops": ["棉花"]},
        {"prov": "新疆维吾尔自治区", "county": "玛纳斯县", "lon": 86.21, "lat": 44.30, "crops": ["棉花", "小麦"]},
        {"prov": "新疆维吾尔自治区", "county": "奇台县", "lon": 89.59, "lat": 44.02, "crops": ["小麦", "大麦"]},
        {"prov": "内蒙古自治区", "county": "阿荣旗", "lon": 123.51, "lat": 48.12, "crops": ["大豆", "玉米"]},
        {"prov": "内蒙古自治区", "county": "五原县", "lon": 108.26, "lat": 41.10, "crops": ["小麦", "玉米"]},
        {"prov": "内蒙古自治区", "county": "突泉县", "lon": 121.57, "lat": 46.33, "crops": ["玉米", "大豆"]},
        {"prov": "甘肃省", "county": "永昌县", "lon": 101.97, "lat": 38.24, "crops": ["大麦", "小麦"]},
        {"prov": "甘肃省", "county": "民乐县", "lon": 100.81, "lat": 38.43, "crops": ["油菜籽", "大麦"]},
        {"prov": "甘肃省", "county": "瓜州县", "lon": 95.77, "lat": 40.51, "crops": ["棉花"]},
        # 中原及华北
        {"prov": "河南省", "county": "滑县", "lon": 114.52, "lat": 35.57, "crops": ["小麦", "玉米"]},
        {"prov": "河南省", "county": "正阳县", "lon": 114.38, "lat": 32.60, "crops": ["花生", "小麦"]},
        {"prov": "河南省", "county": "唐河县", "lon": 112.83, "lat": 32.68, "crops": ["小麦", "花生"]},
        {"prov": "山东省", "county": "平度市", "lon": 119.95, "lat": 36.78, "crops": ["小麦", "花生"]},
        {"prov": "山东省", "county": "莒南县", "lon": 118.83, "lat": 35.17, "crops": ["花生", "大豆"]},
        {"prov": "山东省", "county": "汶上县", "lon": 116.48, "lat": 35.73, "crops": ["玉米", "小麦"]},
        {"prov": "河北省", "county": "藁城区", "lon": 114.84, "lat": 38.03, "crops": ["小麦", "玉米"]},
        {"prov": "河北省", "county": "宁晋县", "lon": 114.92, "lat": 37.62, "crops": ["小麦", "玉米"]},
        {"prov": "河北省", "county": "大名县", "lon": 115.15, "lat": 36.28, "crops": ["花生", "小麦"]},
        # 南方及华东
        {"prov": "江苏省", "county": "兴化市", "lon": 119.85, "lat": 32.93, "crops": ["水稻", "小麦"]},
        {"prov": "江苏省", "county": "东海县", "lon": 118.77, "lat": 34.54, "crops": ["水稻", "花生"]},
        {"prov": "江苏省", "county": "射阳县", "lon": 120.25, "lat": 33.75, "crops": ["棉花", "水稻"]},
        {"prov": "安徽省", "county": "寿县", "lon": 116.78, "lat": 32.57, "crops": ["水稻", "小麦"]},
        {"prov": "安徽省", "county": "蒙城县", "lon": 116.56, "lat": 33.26, "crops": ["小麦", "玉米"]},
        {"prov": "安徽省", "county": "怀远县", "lon": 117.20, "lat": 32.97, "crops": ["水稻", "大豆"]},
        {"prov": "湖北省", "county": "监利市", "lon": 112.90, "lat": 29.83, "crops": ["水稻", "油菜籽"]},
        {"prov": "湖北省", "county": "枣阳市", "lon": 112.75, "lat": 32.13, "crops": ["小麦", "水稻"]},
        {"prov": "湖北省", "county": "钟祥市", "lon": 112.58, "lat": 31.17, "crops": ["油菜籽", "水稻"]},
        {"prov": "湖南省", "county": "安乡县", "lon": 112.17, "lat": 29.41, "crops": ["水稻", "油菜籽"]},
        {"prov": "湖南省", "county": "南县", "lon": 112.40, "lat": 29.36, "crops": ["水稻"]},
        {"prov": "湖南省", "county": "桃源县", "lon": 111.48, "lat": 28.90, "crops": ["水稻", "油菜籽"]},
        {"prov": "四川省", "county": "中江县", "lon": 104.67, "lat": 31.03, "crops": ["油菜籽", "玉米"]},
        {"prov": "四川省", "county": "安岳县", "lon": 105.33, "lat": 30.10, "crops": ["小麦", "油菜籽"]},
        {"prov": "四川省", "county": "三台县", "lon": 105.09, "lat": 31.09, "crops": ["水稻", "玉米"]},
        {"prov": "广西壮族自治区", "county": "宾阳县", "lon": 108.81, "lat": 23.21, "crops": ["水稻", "玉米"]},
        {"prov": "广西壮族自治区", "county": "桂平市", "lon": 110.08, "lat": 23.39, "crops": ["水稻"]},
        {"prov": "广西壮族自治区", "county": "武宣县", "lon": 109.66, "lat": 23.59, "crops": ["水稻", "玉米"]}
    ]

    # 2. 销售端城市库：每个城市3个区/县及真实经纬度
    sales_destinations = {
        "北京市": [
            {"dist": "朝阳区", "lon": 116.48, "lat": 39.92},
            {"dist": "海淀区", "lon": 116.30, "lat": 39.96},
            {"dist": "丰台区", "lon": 116.28, "lat": 39.85}
        ],
        "上海市": [
            {"dist": "浦东新区", "lon": 121.54, "lat": 31.22},
            {"dist": "闵行区", "lon": 121.38, "lat": 31.11},
            {"dist": "宝山区", "lon": 121.48, "lat": 31.40}
        ],
        "广州市": [
            {"dist": "天河区", "lon": 113.33, "lat": 23.13},
            {"dist": "白云区", "lon": 113.27, "lat": 23.15},
            {"dist": "番禺区", "lon": 113.38, "lat": 22.93}
        ],
        "深圳市": [
            {"dist": "福田区", "lon": 114.05, "lat": 22.54},
            {"dist": "南山区", "lon": 113.93, "lat": 22.54},
            {"dist": "宝安区", "lon": 113.88, "lat": 22.55}
        ],
        "成都市": [
            {"dist": "武侯区", "lon": 104.04, "lat": 30.64},
            {"dist": "龙泉驿区", "lon": 104.27, "lat": 30.55},
            {"dist": "新都区", "lon": 104.15, "lat": 30.82}
        ],
        "武汉市": [
            {"dist": "洪山区", "lon": 114.34, "lat": 30.50},
            {"dist": "江岸区", "lon": 114.30, "lat": 30.59},
            {"dist": "东西湖区", "lon": 114.13, "lat": 30.62}
        ],
        "南京市": [
            {"dist": "玄武区", "lon": 118.79, "lat": 32.06},
            {"dist": "江宁区", "lon": 118.83, "lat": 31.95},
            {"dist": "南京浦口区", "lon": 118.62, "lat": 32.05}
        ],
        "西安市": [
            {"dist": "雁塔区", "lon": 108.94, "lat": 34.21},
            {"dist": "未央区", "lon": 108.94, "lat": 34.29},
            {"dist": "长安区", "lon": 108.87, "lat": 34.15}
        ]
    }

    # 3. 作物参数配置 (种植天数, 采摘月份, 农事阶段)
    crop_agronomy = {
        "水稻": {"grow_days": 140, "harvest_months": [9, 10],
                 "ops": ["播种育秧", "田间底肥", "插秧作业", "分蘖期灌溉", "除草控虫", "晒田补肥"]},
        "小麦": {"grow_days": 230, "harvest_months": [6],
                 "ops": ["深耕整地", "秋播种子处理", "越冬水灌溉", "拔节期追肥", "条锈病防治", "干热风防护"]},
        "玉米": {"grow_days": 110, "harvest_months": [9, 10],
                 "ops": ["免耕直播", "苗期中耕", "喇叭口期追肥", "大斑病防治", "人工授粉辅助"]},
        "棉花": {"grow_days": 180, "harvest_months": [10, 11],
                 "ops": ["覆膜播种", "人工打顶", "叶面喷施补硼", "枯萎病监测", "喷施脱叶剂"]},
        "大豆": {"grow_days": 125, "harvest_months": [9],
                 "ops": ["拌种包衣", "等距点播", "结荚期根外追肥", "食心虫绿色防控", "除草中耕"]},
        "油菜籽": {"grow_days": 210, "harvest_months": [5],
                   "ops": ["育苗移栽", "越冬防冻肥", "薹期水肥调节", "菌核病无人机防治"]},
        "花生": {"grow_days": 120, "harvest_months": [8, 9],
                 "ops": ["清棵蹲苗", "培土迎葑", "钙镁肥补充", "下针期控旺", "地下虫害防治"]},
        "大麦": {"grow_days": 100, "harvest_months": [6, 7],
                 "ops": ["翻耕播种", "分蘖期施氮", "化学除草", "灌浆期监控"]}
    }

    # 4. 行业标准逻辑配置
    biz_logic = {
        "水稻": {"proc": "清理去石->砻谷脱壳->糙米碾白->抛光->色选->包装", "store": "平房仓(准低温)",
                 "log_req": "防潮, 湿度<14%", "veh": "散粮车"},
        "小麦": {"proc": "磁选除杂->着水润麦->多级研磨->高方筛理->包装", "store": "钢板筒仓",
                 "log_req": "干燥, 禁止异味混载", "veh": "全封闭货车"},
        "玉米": {"proc": "物理清理->脱粒烘干->破碎加工->筛理分级", "store": "现代化圆筒仓",
                 "log_req": "通风良好, 严控高温", "veh": "重型散粮卡车"},
        "棉花": {"proc": "皮棉清理->锯齿轧花->纤维分级->压力打包", "store": "专业纤维库",
                 "log_req": "严禁烟火, 防静电, 干燥", "veh": "高栏平板车"},
        "大豆": {"proc": "磁选->破碎压胚->高温压榨->物理过滤->精炼", "store": "不锈钢储油罐区",
                 "log_req": "避光, 控温25℃下", "veh": "罐式车"},
        "油菜籽": {"proc": "多级筛选->炒籽->螺旋压榨->精制过滤", "store": "恒温罐区", "log_req": "避光, 防氧化",
                   "veh": "罐式车"},
        "花生": {"proc": "剥壳分选->烘炒->压榨提取->物理精炼", "store": "避光库房", "log_req": "避光防潮",
                 "veh": "厢式货车"},
        "大麦": {"proc": "杂质清理->脱壳处理->分级磨皮->包装", "store": "平房仓", "log_req": "防霉通风",
                 "veh": "散装车"}
    }

    data_list = []

    for i in range(num_records):
        # --- A. 产地环节 ---
        origin = random.choice(china_origin_counties)
        crop_name = random.choice(origin["crops"])
        agro = crop_agronomy[crop_name]

        # 确定采摘日期
        target_month = random.choice(agro["harvest_months"])
        harvest_date = datetime(2024, target_month, random.randint(1, 28), random.randint(8, 17), random.randint(0, 59))
        # 逆推种植日期
        planting_date = harvest_date - timedelta(days=agro["grow_days"] + random.randint(-5, 5))

        # 模拟农事操作 (均匀分布在生长期)
        farm_ops_log = []
        for idx, op_name in enumerate(agro["ops"]):
            op_day = planting_date + timedelta(days=idx * (agro["grow_days"] // len(agro["ops"])))
            farm_ops_log.append(f"{op_day.strftime('%Y-%m-%d')} [{op_name}]")

        farm_lat = origin["lat"] + np.random.uniform(-0.06, 0.06)
        farm_lon = origin["lon"] + np.random.uniform(-0.06, 0.06)

        # --- B. 加工与仓储 (分钟级) ---
        conf = biz_logic[crop_name]
        proc_start = harvest_date + timedelta(days=random.randint(1, 4), hours=random.randint(2, 6))
        proc_duration = random.randint(150, 600)
        proc_end = proc_start + timedelta(minutes=proc_duration)

        wh_entry = proc_end + timedelta(hours=random.randint(4, 12))
        storage_days = random.randint(10, 50)
        wh_exit = wh_entry + timedelta(days=storage_days)

        # 仓库/加工厂坐标 (县域内漂移)
        factory_lat = origin["lat"] + np.random.uniform(-0.01, 0.01)
        factory_lon = origin["lon"] + np.random.uniform(-0.01, 0.01)

        # --- C. 物流与销售 (多网点) ---
        city_name = random.choice(list(sales_destinations.keys()))
        dist_info = random.choice(sales_destinations[city_name])

        log_start = wh_exit + timedelta(hours=random.randint(6, 15))
        transit_hours = random.randint(24, 130)  # 跨省距离模拟
        arrival_time = log_start + timedelta(hours=transit_hours, minutes=random.randint(0, 59))

        sale_time = arrival_time + timedelta(days=random.randint(1, 3))

        data_list.append({
            "溯源批次码": f"CN-AG-{harvest_date.strftime('%Y%m')}-{i + 10000:05d}",
            "作物名称": crop_name,
            "生产省份": origin["prov"],
            "生产县区": origin["county"],
            "基地经度": round(farm_lon, 6),
            "基地纬度": round(farm_lat, 6),
            "种植日期": planting_date.strftime('%Y-%m-%d'),
            "全流程农事操作记录": " -> ".join(farm_ops_log),
            "采摘完成时刻": harvest_date.strftime('%Y-%m-%d %H:%M'),
            "采摘重量(kg)": round(random.uniform(5000, 40000), 2),

            "加工中心": f"{origin['county']}第{random.randint(1, 5)}深加工基地",
            "加工厂经纬度": f"{round(factory_lon, 6)},{round(factory_lat, 6)}",
            "加工具体工序": conf["proc"],
            "加工开始时刻": proc_start.strftime('%Y-%m-%d %H:%M'),
            "加工结束时刻": proc_end.strftime('%Y-%m-%d %H:%M'),

            "存储仓库ID": f"WH-CN-{random.randint(100, 999)}",
            "仓库类型": conf["store"],
            "入库时刻": wh_entry.strftime('%Y-%m-%d %H:%M'),
            "出库时刻": wh_exit.strftime('%Y-%m-%d %H:%M'),
            "仓储温湿度环境": "准低温低氧控制" if crop_name in ["水稻", "小麦", "大豆"] else "避光控温控制",

            "物流运单号": f"LOG-CN-{random.randint(10 ** 10, 10 ** 11 - 1)}",
            "运输车辆类型": conf["veh"],
            "物流作业要求": conf["log_req"],
            "发货起运时刻": log_start.strftime('%Y-%m-%d %H:%M'),
            "终端签收时刻": arrival_time.strftime('%Y-%m-%d %H:%M'),

            "销售城市": city_name,
            "销售区县网点": dist_info["dist"],
            "网点经度": dist_info["lon"],
            "网点纬度": dist_info["lat"],
            "终端上架时刻": sale_time.strftime('%Y-%m-%d %H:%M')
        })

    return pd.DataFrame(data_list)


# 生成数据集
df_trace = generate_china_premium_traceability(2000)

# 查看预览
print("--- 中国农业大宗产品全生命周期溯源数据预览 ---")
print(df_trace[['溯源批次码', '作物名称', '种植日期', '销售区县网点', '终端上架时刻']].head())

# 导出 CSV (UTF-8 with BOM 以防Excel乱码)
df_trace.to_csv("china_agri_traceability_v10_final.csv", index=False, encoding='utf-8-sig')
from FeatureImportance import *

feature_list = [{"feature": "consume_age_olduser", "weight": 556},
                {"feature": "home_poi_sales_olduser", "weight": 543},
                {"feature": "home_poi_cnt_olduser", "weight": 524},
                {"feature": "canyinwaimai_prefer_score_olduser", "weight": 503},
                {"feature": "userid_rtb_30d_imp_olduser", "weight": 499},
                {"feature": "work_poi_sales_olduser", "weight": 493},
                {"feature": "silence_age_olduser", "weight": 487},
                {"feature": "userid_all_source_view_day30_cnt_olduser", "weight": 483},
                {"feature": "last_visit_day_view_sku_cnt_olduser", "weight": 471},
                {"feature": "first_visit_gap_olduser", "weight": 470},
                {"feature": "arouse_poi_num_day_1_olduser", "weight": 453},
                {"feature": "work_poi_cnt_olduser", "weight": 436},
                {"feature": "userid_all_source_pageview_day30_cnt_olduser", "weight": 430},
                {"feature": "userid_all_source_click_day30_cnt_olduser", "weight": 408},
                {"feature": "last_visit_gap_olduser", "weight": 400},
                {"feature": "last_visit_source_olduser", "weight": 394},
                {"feature": "total_visit_times_olduser", "weight": 385},
                {"feature": "arouse_poi_num_day_30_olduser", "weight": 384},
                {"feature": "daocan_prefer_score_olduser", "weight": 383},
                {"feature": "recent_day30_promotion_reduce_per_order_olduser", "weight": 372},
                {"feature": "poi_distance_olduser", "weight": 370},
                {"feature": "total_visit_day_num_olduser", "weight": 369},
                {"feature": "recent_day30_sale_amt_per_order_olduser", "weight": 367},
                {"feature": "userid_all_source_view_day14_cnt_olduser", "weight": 335},
                {"feature": "userid_rtb_7d_imp_olduser", "weight": 333}, {"feature": "age_level", "weight": 328},
                {"feature": "city_day3_fugou_rate_olduser", "weight": 322},
                {"feature": "city_day14_fugou_rate_olduser", "weight": 311},
                {"feature": "city_day1_fugou_rate_olduser", "weight": 310},
                {"feature": "total_visit_ord_rate_olduser", "weight": 304},
                {"feature": "reduce_amt_per_order_olduser", "weight": 304},
                {"feature": "current_hour", "weight": 296},
                {"feature": "arouse_visit_num_30_dt_cnt_olduser", "weight": 286},
                {"feature": "userid_tx_day30_visit_cnt_olduser", "weight": 285},
                {"feature": "userid_all_source_view_day7_cnt_olduser", "weight": 280},
                {"feature": "view_sku_cnt_rec_day7_olduser", "weight": 278},
                {"feature": "userid_rtb_14d_imp_olduser", "weight": 274},
                {"feature": "userid_all_source_addcar_day30_cnt_olduser", "weight": 274},
                {"feature": "userid_all_source_lj_buy_day30_cnt_olduser", "weight": 273},
                {"feature": "arouse_visit_trend_day30today14_olduser", "weight": 271},
                {"feature": "city_day7_fugou_rate_olduser", "weight": 267},
                {"feature": "often_visit_source_olduser", "weight": 265},
                {"feature": "userid_rtb_tx_day3_req_cnt_olduser", "weight": 264},
                {"feature": "arouse_poi_num_day_14_olduser", "weight": 264},
                {"feature": "arouse_poi_num_day_7_olduser", "weight": 260},
                {"feature": "userid_rtb_30d_click_olduser", "weight": 255},
                {"feature": "last_ord_reduce_rate_olduser", "weight": 250},
                {"feature": "userid_rtb_3d_imp_olduser", "weight": 250},
                {"feature": "userid_rtb_tx_day30_req_cnt_olduser", "weight": 246},
                {"feature": "first_ord_original_price_olduser", "weight": 244},
                {"feature": "userid_rtb_24h_imp", "weight": 244},
                {"feature": "userid_rtb_tx_day14_req_cnt_olduser", "weight": 244},
                {"feature": "xiuxianyule_prefer_score_olduser", "weight": 239},
                {"feature": "arouse_num_30day_olduser", "weight": 238},
                {"feature": "total_ord_dt_cnt_olduser", "weight": 237},
                {"feature": "userid_all_source_click_day14_cnt_olduser", "weight": 232},
                {"feature": "userid_all_source_pageview_day14_cnt_olduser", "weight": 228},
                {"feature": "first_ord_reduce_rate_olduser", "weight": 224},
                {"feature": "total_promotion_reduce_olduser", "weight": 221},
                {"feature": "cid_rtb_30d_cate1_ctr", "weight": 221},
                {"feature": "arouse_num_14day_olduser", "weight": 221},
                {"feature": "recent_day14_sale_amt_per_order_olduser", "weight": 221},
                {"feature": "cid_rtb_1h_ctr", "weight": 217}, {"feature": "cid_rtb_30d_cate2_ctr", "weight": 216},
                {"feature": "total_real_pay_olduser", "weight": 212},
                {"feature": "userid_rtb_tx_day7_req_cnt_olduser", "weight": 211},
                {"feature": "userid_all_source_view_day1_cnt_olduser", "weight": 207},
                {"feature": "total_promotion_reduce_rate_olduser", "weight": 206},
                {"feature": "userid_rtb_tx_day1_req_cnt_olduser", "weight": 206},
                {"feature": "userid_all_source_addcar_day7_cnt_olduser", "weight": 206},
                {"feature": "userid_all_source_addcar_day14_cnt_olduser", "weight": 205},
                {"feature": "last_ord_original_price_olduser", "weight": 200},
                {"feature": "userid_rtb_24h_ctr", "weight": 197},
                {"feature": "userid_all_source_view_day3_cnt_olduser", "weight": 195},
                {"feature": "first_ord_reduce_olduser", "weight": 195},
                {"feature": "recent_day7_sale_amt_per_order_olduser", "weight": 192},
                {"feature": "view_sku_cnt_rec_day3_olduser", "weight": 192},
                {"feature": "cid_rtb_30d_cate1_ctcvr", "weight": 190},
                {"feature": "arouse_visit_trend_day14today7_olduser", "weight": 186},
                {"feature": "crt_ord_num_workday_olduser", "weight": 185},
                {"feature": "cid_rtb_12h_ctr", "weight": 183},
                {"feature": "recent_day30_sale_amt_olduser", "weight": 181},
                {"feature": "cid_rtb_3h_ctr", "weight": 178},
                {"feature": "arouse_visit_trend_day7today3_olduser", "weight": 176},
                {"feature": "arouse_num_7day_olduser", "weight": 175},
                {"feature": "userid_all_source_click_day3_cnt_olduser", "weight": 173},
                {"feature": "maoyan_prefer_score_olduser", "weight": 173},
                {"feature": "often_visit_city_level_olduser", "weight": 169},
                {"feature": "recent_day90_sale_amt_olduser", "weight": 168},
                {"feature": "cid_rtb_30d_cate2_ctcvr", "weight": 167},
                {"feature": "cid_rtb_30d_ctcvr", "weight": 165},
                {"feature": "total_page_stay_time_olduser", "weight": 162},
                {"feature": "userid_tx_day14_visit_cnt_olduser", "weight": 162},
                {"feature": "total_cat3_name_total_olduser", "weight": 161},
                {"feature": "cid_rtb_24h_ctr", "weight": 161},
                {"feature": "userid_all_source_pageview_day7_cnt_olduser", "weight": 159},
                {"feature": "userid_all_source_addcar_day3_cnt_olduser", "weight": 158},
                {"feature": "userid_all_source_click_day7_cnt_olduser", "weight": 157},
                {"feature": "last_ord_real_pay_olduser", "weight": 156},
                {"feature": "waimai_r_score_olduser", "weight": 155},
                {"feature": "waimai_m_score_olduser", "weight": 154},
                {"feature": "first_ord_real_pay_olduser", "weight": 152},
                {"feature": "liren_prefer_score_olduser", "weight": 151},
                {"feature": "crt_ord_num_holiday_olduser", "weight": 149},
                {"feature": "userid_all_source_pageview_day3_cnt_olduser", "weight": 146},
                {"feature": "last_ord_reduce_olduser", "weight": 145},
                {"feature": "salary_level_olduser", "weight": 145}, {"feature": "cid_rtb_7d_ctcvr", "weight": 144},
                {"feature": "last_ord_sale_num_olduser", "weight": 143},
                {"feature": "arouse_visit_num_14_dt_cnt_olduser", "weight": 143},
                {"feature": "often_ord_cellular_type_olduser", "weight": 138},
                {"feature": "cid_rtb_7d_ctr", "weight": 137}, {"feature": "arouse_num_1day_olduser", "weight": 136},
                {"feature": "userid_all_source_lj_buy_day14_cnt_olduser", "weight": 136},
                {"feature": "daocan_r_score_olduser", "weight": 135},
                {"feature": "first_ord_region_olduser", "weight": 132},
                {"feature": "waimai_re_score_olduser", "weight": 131},
                {"feature": "is_day30_visit_wxapp_olduser", "weight": 131},
                {"feature": "cid_rtb_14d_ctcvr", "weight": 130},
                {"feature": "total_refund_amt_olduser", "weight": 129},
                {"feature": "cid_rtb_30d_cate3_ctcvr", "weight": 128},
                {"feature": "recent_day30_ord_num_olduser", "weight": 127},
                {"feature": "cid_rtb_30d_cate5_ctcvr", "weight": 127},
                {"feature": "real_pay_per_order_olduser", "weight": 125},
                {"feature": "userid_all_source_pageview_day1_cnt_olduser", "weight": 125},
                {"feature": "cid_rtb_30d_cate3_ctr", "weight": 124},
                {"feature": "recent_day90_promotion_reduce_per_order_olduser", "weight": 124},
                {"feature": "first_ord_sale_num_olduser", "weight": 123},
                {"feature": "gender_olduser", "weight": 123}, {"feature": "cid_rtb_30d_ctr", "weight": 123},
                {"feature": "recent_day90_sale_amt_per_order_olduser", "weight": 122},
                {"feature": "cid_rtb_3d_ctr", "weight": 121},
                {"feature": "recent_day90_ord_num_olduser", "weight": 121},
                {"feature": "cid_rtb_3d_ctcvr", "weight": 121},
                {"feature": "cid_rtb_30d_cate4_ctcvr", "weight": 121},
                {"feature": "daocan_m_score_olduser", "weight": 121},
                {"feature": "cid_rtb_30d_cate4_ctr", "weight": 120}, {"feature": "cid_rtb_3d_cvr", "weight": 119},
                {"feature": "cid_rtb_30d_cate5_ctr", "weight": 118},
                {"feature": "userid_all_source_addcar_day1_cnt_olduser", "weight": 116},
                {"feature": "last_ord_region_olduser", "weight": 116},
                {"feature": "cid_rtb_14d_ctr", "weight": 116},
                {"feature": "userid_all_source_click_day1_cnt_olduser", "weight": 115},
                {"feature": "cid_rtb_14d_cvr", "weight": 115},
                {"feature": "userid_rtb_14d_click_olduser", "weight": 115},
                {"feature": "userid_cate3id_youxuan_1y_click", "weight": 115},
                {"feature": "cid_rtb_7d_cvr", "weight": 111}, {"feature": "waimai_f_score_olduser", "weight": 111},
                {"feature": "cid_rtb_today_imp", "weight": 110}, {"feature": "is_have_baby_olduser", "weight": 108},
                {"feature": "page_log_times_day_olduser", "weight": 107},
                {"feature": "recent_day14_ord_num_olduser", "weight": 107},
                {"feature": "userid_cate4id_youxuan_1y_click", "weight": 105},
                {"feature": "is_student_olduser", "weight": 104}, {"feature": "cid_rtb_30d_cvr", "weight": 102},
                {"feature": "userid_all_source_lj_buy_day7_cnt_olduser", "weight": 100},
                {"feature": "cid_rtb_24h_click", "weight": 97}, {"feature": "cid_rtb_3d_gmv", "weight": 96},
                {"feature": "total_cat2_name_total_olduser", "weight": 96},
                {"feature": "first_visit_source_olduser", "weight": 96},
                {"feature": "cid_rtb_1h_imp", "weight": 95}, {"feature": "cid_rtb_3h_imp", "weight": 95},
                {"feature": "cid_rtb_12h_click", "weight": 93}, {"feature": "cid_rtb_3d_click", "weight": 93},
                {"feature": "cid_rtb_14d_gmv", "weight": 91}, {"feature": "cid_rtb_12h_cost", "weight": 90},
                {"feature": "cid_rtb_1h_cost", "weight": 89}, {"feature": "cid_rtb_24h_imp", "weight": 89},
                {"feature": "arouse_poi_day_cnt_30_olduser", "weight": 89},
                {"feature": "cid_rtb_today_click", "weight": 89}, {"feature": "cid_rtb_24h_cost", "weight": 87},
                {"feature": "total_ord_week_cnt_olduser", "weight": 82},
                {"feature": "cid_rtb_7d_gmv", "weight": 82}, {"feature": "cid_rtb_30d_gmv", "weight": 80},
                {"feature": "cid_rtb_3d_imp", "weight": 80}, {"feature": "cid_rtb_1h_click", "weight": 79},
                {"feature": "total_refund_ord_num_olduser", "weight": 79},
                {"feature": "recent_day14_sale_amt_olduser", "weight": 78},
                {"feature": "userid_rtb_12h_imp", "weight": 77}, {"feature": "cid_rtb_12h_imp", "weight": 75},
                {"feature": "userid_rtb_7d_click_olduser", "weight": 75},
                {"feature": "cid_rtb_3d_conv", "weight": 74}, {"feature": "userid_rtb_12h_ctr", "weight": 74},
                {"feature": "cid_rtb_7d_imp", "weight": 73}, {"feature": "cid_rtb_30d_click", "weight": 73},
                {"feature": "is_day14_visit_wxapp_olduser", "weight": 72},
                {"feature": "userid_rtb_24h_click", "weight": 72},
                {"feature": "is_visited_wxapp_olduser", "weight": 72},
                {"feature": "userid_rtb_12h_click", "weight": 71},
                {"feature": "total_sale_amt_olduser", "weight": 71},
                {"feature": "consume_style_olduser", "weight": 70}, {"feature": "cid_rtb_14d_click", "weight": 70},
                {"feature": "first_ord_province_level_olduser", "weight": 69},
                {"feature": "cid_rtb_3h_click", "weight": 69}, {"feature": "cid_rtb_30d_conv", "weight": 67},
                {"feature": "cid_rtb_14d_imp", "weight": 66}, {"feature": "is_white_collar_olduser", "weight": 66},
                {"feature": "cid_rtb_30d_imp", "weight": 66},
                {"feature": "userid_cate4id_youxuan_1y_jiesuan", "weight": 66},
                {"feature": "cid_rtb_3h_cost", "weight": 66}, {"feature": "cid_rtb_7d_click", "weight": 65},
                {"feature": "arouse_visit_num_7_dt_cnt_olduser", "weight": 64},
                {"feature": "recent_day7_ord_num_olduser", "weight": 63},
                {"feature": "is_married_olduser", "weight": 63},
                {"feature": "daocan_f_score_olduser", "weight": 62},
                {"feature": "first_ord_city_level_olduser", "weight": 62},
                {"feature": "userid_tx_day7_visit_cnt_olduser", "weight": 61},
                {"feature": "userid_cate3id_youxuan_1y_jiesuan", "weight": 60},
                {"feature": "userid_rtb_3h_imp", "weight": 60},
                {"feature": "daocan_re_score_olduser", "weight": 58},
                {"feature": "userid_all_source_lj_buy_day3_cnt_olduser", "weight": 56},
                {"feature": "last_ord_province_level_olduser", "weight": 55},
                {"feature": "total_cat1_name_total_olduser", "weight": 54},
                {"feature": "arouse_poi_day_cnt_14_olduser", "weight": 54},
                {"feature": "arouse_poi_num_day_avg_1_olduser", "weight": 54},
                {"feature": "refund_action_level", "weight": 52},
                {"feature": "recent_day7_sale_amt_olduser", "weight": 51},
                {"feature": "cid_rtb_14d_conv", "weight": 51}, {"feature": "cid_rtb_7d_conv", "weight": 50},
                {"feature": "edu_level_olduser", "weight": 49}, {"feature": "maoyan_m_score_olduser", "weight": 48},
                {"feature": "sensitivity_level_olduser", "weight": 47},
                {"feature": "userid_rtb_3d_click_olduser", "weight": 46},
                {"feature": "is_cook_olduser", "weight": 46},
                {"feature": "userid_all_source_lj_buy_day1_cnt_olduser", "weight": 45},
                {"feature": "maoyan_r_score_olduser", "weight": 44},
                {"feature": "userid_cate3id_youxuan_1y_lj_buy", "weight": 44},
                {"feature": "income_level_olduser", "weight": 41},
                {"feature": "userid_tx_day3_visit_cnt_olduser", "weight": 40},
                {"feature": "peisong_r_score_olduser", "weight": 40},
                {"feature": "first_ord_cooperation_type_olduser", "weight": 38},
                {"feature": "last_ord_city_level_olduser", "weight": 36},
                {"feature": "is_visited_group_wxapp_olduser", "weight": 36},
                {"feature": "page_stay_time_per_visit_olduser", "weight": 36},
                {"feature": "is_cate_100112", "weight": 35}, {"feature": "car_owner_label_olduser", "weight": 35},
                {"feature": "total_case_num_olduser", "weight": 34},
                {"feature": "userid_cate1id_rtb_30d_imp", "weight": 32},
                {"feature": "userid_rtb_1h_imp", "weight": 32}, {"feature": "is_cate_100108", "weight": 32},
                {"feature": "is_day7_visit_wxapp_olduser", "weight": 32},
                {"feature": "userid_rtb_1h_click", "weight": 31},
                {"feature": "is_day1_visit_wxapp_olduser", "weight": 31},
                {"feature": "userid_cate1id_rtb_30d_ctr", "weight": 31},
                {"feature": "cid_rtb_today_conv", "weight": 30},
                {"feature": "userid_cate4id_youxuan_1y_submit", "weight": 29},
                {"feature": "maoyan_re_score_olduser", "weight": 29},
                {"feature": "userid_cate5id_youxuan_1y_click", "weight": 29},
                {"feature": "arouse_poi_day_cnt_7_olduser", "weight": 29},
                {"feature": "userid_cate4id_youxuan_1y_lj_buy", "weight": 28},
                {"feature": "peisong_re_score_olduser", "weight": 27},
                {"feature": "peisong_m_score_olduser", "weight": 27}, {"feature": "is_cate_230031", "weight": 27},
                {"feature": "userid_cate4id_youxuan_1y_order", "weight": 26},
                {"feature": "maoyan_f_score_olduser", "weight": 26},
                {"feature": "userid_rtb_3h_click", "weight": 25},
                {"feature": "userid_cate3id_youxuan_1y_submit", "weight": 22},
                {"feature": "userid_cate3id_youxuan_1y_order", "weight": 21},
                {"feature": "userid_cate2id_rtb_30d_ctr", "weight": 20},
                {"feature": "userid_cate5id_youxuan_1y_jiesuan", "weight": 17},
                {"feature": "userid_cate2id_rtb_30d_imp", "weight": 16},
                {"feature": "is_cate_100111", "weight": 16},
                {"feature": "arouse_poi_day_cnt_1_olduser", "weight": 16},
                {"feature": "is_cate_100005", "weight": 15},
                {"feature": "userid_rtb_30d_pctr_olduser", "weight": 14},
                {"feature": "userid_cate1id_rtb_14d_imp", "weight": 14},
                {"feature": "sale_amt_per_order_olduser", "weight": 14},
                {"feature": "userid_rtb_12h_conv", "weight": 14}, {"feature": "is_cate_100114", "weight": 13},
                {"feature": "userid_rtb_12h_cvr", "weight": 13},
                {"feature": "arouse_poi_num_day_avg_30_olduser", "weight": 13},
                {"feature": "userid_rtb_1h_cvr", "weight": 13},
                {"feature": "total_ord_month_cnt_olduser", "weight": 13},
                {"feature": "userid_tx_day1_visit_cnt_olduser", "weight": 13},
                {"feature": "userid_cate2id_rtb_14d_imp", "weight": 13},
                {"feature": "is_cate_240180", "weight": 13}, {"feature": "total_ord_num_olduser", "weight": 12},
                {"feature": "is_cate_240020", "weight": 12},
                {"feature": "last_ord_cooperation_type_olduser", "weight": 12},
                {"feature": "is_cate_240021", "weight": 11}, {"feature": "userid_rtb_3h_ctr", "weight": 11},
                {"feature": "arouse_poi_num_day_avg_7_olduser", "weight": 11},
                {"feature": "userid_rtb_30d_conv_olduser", "weight": 11},
                {"feature": "is_visited_group_yx_wxapp_olduser", "weight": 11},
                {"feature": "userid_cate3id_rtb_30d_ctr", "weight": 10},
                {"feature": "userid_cate1id_rtb_30d_click", "weight": 10},
                {"feature": "cid_rtb_is_contain_xg", "weight": 10},
                {"feature": "peisong_f_score_olduser", "weight": 10},
                {"feature": "userid_rtb_3d_pctr_olduser", "weight": 10},
                {"feature": "is_cate_100004", "weight": 10}, {"feature": "userid_rtb_7d_conv_olduser", "weight": 9},
                {"feature": "userid_rtb_14d_pctr_olduser", "weight": 9},
                {"feature": "arouse_poi_num_day_avg_14_olduser", "weight": 9},
                {"feature": "userid_rtb_7d_pctr_olduser", "weight": 8},
                {"feature": "userid_cate3id_rtb_30d_ctcvr", "weight": 7},
                {"feature": "userid_cate1id_rtb_14d_ctr", "weight": 7},
                {"feature": "is_day30_visit_olduser", "weight": 7},
                {"feature": "userid_cate5id_youxuan_1y_order", "weight": 7},
                {"feature": "is_cate_230038", "weight": 7},
                {"feature": "userid_cate5id_youxuan_1y_lj_buy", "weight": 6},
                {"feature": "userid_cate3id_rtb_14d_ctr", "weight": 6},
                {"feature": "userid_cate3id_rtb_14d_ctcvr", "weight": 6},
                {"feature": "userid_cate1id_rtb_30d_ctcvr", "weight": 5},
                {"feature": "userid_cate1id_rtb_30d_cvr", "weight": 5}, {"feature": "is_cate_230029", "weight": 5},
                {"feature": "is_cate_100100", "weight": 5},
                {"feature": "userid_cate5id_youxuan_1y_submit", "weight": 5},
                {"feature": "userid_rtb_24h_cvr", "weight": 5}, {"feature": "is_cate_100106", "weight": 4},
                {"feature": "userid_rtb_1h_ctr", "weight": 4},
                {"feature": "userid_cate1id_rtb_14d_cvr", "weight": 4}, {"feature": "cid_rtb_1h_gmv", "weight": 4},
                {"feature": "userid_rtb_14d_conv_olduser", "weight": 4},
                {"feature": "userid_cate2id_rtb_30d_ctcvr", "weight": 4},
                {"feature": "userid_rtb_14d_pcvr_olduser", "weight": 3},
                {"feature": "cid_rtb_3h_conv", "weight": 3}, {"feature": "userid_rtb_1h_conv", "weight": 3},
                {"feature": "is_day14_visit_olduser", "weight": 3},
                {"feature": "userid_rtb_3d_pcvr_olduser", "weight": 3}, {"feature": "cid_rtb_12h_gmv", "weight": 3},
                {"feature": "arouse_visit_num_1_dt_cnt_olduser", "weight": 3},
                {"feature": "userid_cate2id_rtb_14d_ctr", "weight": 3},
                {"feature": "userid_cate1id_rtb_14d_click", "weight": 3},
                {"feature": "is_cate_100001", "weight": 3}, {"feature": "cid_rtb_1h_conv", "weight": 2},
                {"feature": "userid_rtb_30d_pcvr_olduser", "weight": 2}, {"feature": "is_cate_240472", "weight": 2},
                {"feature": "userid_cate3id_rtb_30d_cvr", "weight": 2}, {"feature": "is_cate_240459", "weight": 2},
                {"feature": "is_cate_230033", "weight": 2},
                {"feature": "userid_cate1id_rtb_14d_ctcvr", "weight": 2},
                {"feature": "userid_cate2id_rtb_30d_click", "weight": 1},
                {"feature": "is_cate_240131", "weight": 1}, {"feature": "userid_cate2id_rtb_30d_cvr", "weight": 1},
                {"feature": "cid_rtb_12h_cvr", "weight": 1}, {"feature": "cid_rtb_12h_conv", "weight": 1},
                {"feature": "is_day1_visit_olduser", "weight": 1},
                {"feature": "userid_cate2id_rtb_14d_click", "weight": 1},
                {"feature": "is_cate_240143", "weight": 1},
                {"feature": "userid_cate2id_rtb_14d_ctcvr", "weight": 1},
                {"feature": "is_cate_240500", "weight": 1}]

CalOneFeature(feature_list, "is_treat")
CalOneFeature(feature_list, "current_hour")
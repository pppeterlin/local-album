"""English → 繁體中文 place-name translation for reverse-geocoded city names.

The ``reverse-geocoder`` package returns city names from the GeoNames
dataset in English (e.g. "Nantou", "Banqiao", "Kyoto"). For a
Mandarin-speaking user this reads as foreign. ``zh_place()`` looks the
English name up in :data:`PLACE_ZH` and returns the local Chinese name
when known.

User overrides:
    Drop a JSON file at ``$METADATA_DIR/index/location_names_zh.json``
    with shape ``{"English": "中文", ...}`` — its entries win over the
    bundled defaults.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional


# Bundled defaults — Taiwan + East Asia + popular travel destinations.
# Pull requests welcome for any missing common city.
PLACE_ZH: Dict[str, str] = {
    # ── Taiwan: counties / cities ─────────────────────────────────
    "Taipei": "台北",
    "New Taipei": "新北",
    "Taoyuan": "桃園",
    "Taoyuan District": "桃園",
    "Taichung": "台中",
    "Tainan": "台南",
    "Kaohsiung": "高雄",
    "Hsinchu": "新竹",
    "Keelung": "基隆",
    "Chiayi": "嘉義",
    "Pingtung": "屏東",
    "Yilan": "宜蘭",
    "Hualien": "花蓮",
    "Taitung": "台東",
    "Nantou": "南投",
    "Miaoli": "苗栗",
    "Changhua": "彰化",
    "Yunlin": "雲林",
    "Penghu": "澎湖",
    "Kinmen": "金門",
    "Lienchiang": "馬祖",
    # ── Taiwan: New Taipei / Taipei districts (common ones) ──────
    "Banqiao": "板橋",
    "Sanchong": "三重",
    "Yonghe": "永和",
    "Zhonghe": "中和",
    "Xinzhuang": "新莊",
    "Tucheng": "土城",
    "Sanxia": "三峽",
    "Xindian": "新店",
    "Shulin": "樹林",
    "Luzhou": "蘆洲",
    "Tamsui": "淡水",
    "Danshui": "淡水",
    "Wugu": "五股",
    "Linkou": "林口",
    "Ruifang": "瑞芳",
    "Jinshan": "金山",
    "Wanli": "萬里",
    "Shilin": "士林",
    "Beitou": "北投",
    "Neihu": "內湖",
    "Nangang": "南港",
    "Wenshan": "文山",
    "Songshan": "松山",
    "Xinyi": "信義",
    "Daan": "大安",
    "Da'an": "大安",
    "Zhongshan": "中山",
    "Zhongzheng": "中正",
    "Datong": "大同",
    "Wanhua": "萬華",
    "Banciao": "板橋",
    # ── Taiwan: other municipal districts ────────────────────────
    "Toufen": "頭份",
    "Zhubei": "竹北",
    "Yuanlin": "員林",
    "Douliu": "斗六",
    "Magong": "馬公",
    "Su'ao": "蘇澳",
    "Suao": "蘇澳",
    "Luodong": "羅東",
    "Jiaoxi": "礁溪",
    "Yuanshan": "員山",
    "Ji'an": "吉安",
    "Yuli": "玉里",
    "Chenggong": "成功",
    "Beigang": "北港",
    "Puli": "埔里",
    "Lugang": "鹿港",

    # ── Japan ─────────────────────────────────────────────────────
    "Tokyo": "東京",
    "Kyoto": "京都",
    "Osaka": "大阪",
    "Yokohama": "橫濱",
    "Nagoya": "名古屋",
    "Sapporo": "札幌",
    "Fukuoka": "福岡",
    "Kobe": "神戶",
    "Hiroshima": "廣島",
    "Nara": "奈良",
    "Naha": "那霸",
    "Okinawa": "沖繩",
    "Sendai": "仙台",
    "Nagano": "長野",
    "Kanazawa": "金澤",
    "Hakodate": "函館",
    "Otaru": "小樽",
    "Asahikawa": "旭川",
    "Niigata": "新潟",
    "Matsumoto": "松本",
    "Takayama": "高山",
    "Nikko": "日光",
    "Kamakura": "鎌倉",
    "Yokosuka": "橫須賀",
    "Chiba": "千葉",
    "Saitama": "埼玉",
    "Kawasaki": "川崎",
    "Shibuya": "澀谷",
    "Shinjuku": "新宿",
    "Akihabara": "秋葉原",
    "Asakusa": "淺草",
    "Ginza": "銀座",
    "Ueno": "上野",

    # ── Korea ─────────────────────────────────────────────────────
    "Seoul": "首爾",
    "Busan": "釜山",
    "Incheon": "仁川",
    "Daegu": "大邱",
    "Daejeon": "大田",
    "Gwangju": "光州",
    "Jeju": "濟州",
    "Jeju City": "濟州",
    "Suwon": "水原",
    "Ulsan": "蔚山",
    "Gangneung": "江陵",

    # ── Greater China / HK / Macau ────────────────────────────────
    "Beijing": "北京",
    "Shanghai": "上海",
    "Guangzhou": "廣州",
    "Shenzhen": "深圳",
    "Chengdu": "成都",
    "Hangzhou": "杭州",
    "Suzhou": "蘇州",
    "Wuhan": "武漢",
    "Xi'an": "西安",
    "Xian": "西安",
    "Nanjing": "南京",
    "Tianjin": "天津",
    "Chongqing": "重慶",
    "Qingdao": "青島",
    "Dalian": "大連",
    "Xiamen": "廈門",
    "Hong Kong": "香港",
    "Kowloon": "九龍",
    "Macau": "澳門",
    "Macao": "澳門",

    # ── Southeast Asia ────────────────────────────────────────────
    "Bangkok": "曼谷",
    "Chiang Mai": "清邁",
    "Phuket": "普吉",
    "Pattaya": "芭達雅",
    "Singapore": "新加坡",
    "Kuala Lumpur": "吉隆坡",
    "Penang": "檳城",
    "George Town": "喬治市",
    "Manila": "馬尼拉",
    "Cebu City": "宿霧",
    "Ho Chi Minh City": "胡志明市",
    "Hanoi": "河內",
    "Da Nang": "峴港",
    "Hoi An": "會安",
    "Phnom Penh": "金邊",
    "Siem Reap": "暹粒",
    "Jakarta": "雅加達",
    "Bali": "峇里",
    "Denpasar": "登巴薩",
    "Yangon": "仰光",
    "Vientiane": "永珍",
    "Luang Prabang": "龍坡邦",

    # ── Other popular travel ──────────────────────────────────────
    "New York City": "紐約",
    "New York": "紐約",
    "Los Angeles": "洛杉磯",
    "San Francisco": "舊金山",
    "Seattle": "西雅圖",
    "Boston": "波士頓",
    "Chicago": "芝加哥",
    "Las Vegas": "拉斯維加斯",
    "Honolulu": "檀香山",
    "Vancouver": "溫哥華",
    "Toronto": "多倫多",
    "Montreal": "蒙特婁",
    "London": "倫敦",
    "Paris": "巴黎",
    "Rome": "羅馬",
    "Milan": "米蘭",
    "Florence": "佛羅倫斯",
    "Venice": "威尼斯",
    "Madrid": "馬德里",
    "Barcelona": "巴塞隆納",
    "Lisbon": "里斯本",
    "Amsterdam": "阿姆斯特丹",
    "Berlin": "柏林",
    "Munich": "慕尼黑",
    "Vienna": "維也納",
    "Prague": "布拉格",
    "Zurich": "蘇黎世",
    "Geneva": "日內瓦",
    "Stockholm": "斯德哥爾摩",
    "Copenhagen": "哥本哈根",
    "Helsinki": "赫爾辛基",
    "Oslo": "奧斯陸",
    "Reykjavik": "雷克雅維克",
    "Sydney": "雪梨",
    "Melbourne": "墨爾本",
    "Brisbane": "布里斯本",
    "Auckland": "奧克蘭",
    "Wellington": "威靈頓",
}


_USER_OVERRIDE_LOADED_AT: Optional[float] = None
_MERGED: Optional[Dict[str, str]] = None


def _resolve_map(override_path: Optional[Path]) -> Dict[str, str]:
    """Return merged map. Bundled defaults < user JSON override.
    Cached by override file mtime so edits hot-reload without restart."""
    global _USER_OVERRIDE_LOADED_AT, _MERGED
    if not override_path or not override_path.exists():
        if _MERGED is None:
            _MERGED = dict(PLACE_ZH)
        return _MERGED
    mtime = override_path.stat().st_mtime
    if _MERGED is None or _USER_OVERRIDE_LOADED_AT != mtime:
        merged = dict(PLACE_ZH)
        try:
            merged.update(json.loads(override_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
        _MERGED = merged
        _USER_OVERRIDE_LOADED_AT = mtime
    return _MERGED


def zh_place(name: Optional[str], override_path: Optional[Path] = None) -> Optional[str]:
    """Translate a reverse-geocoded place name to 繁中 if mapped.

    Accepts either "Nantou" or "Nantou, TW"; CC suffix is dropped when a
    translation is found. Unmapped names are returned unchanged so the
    user still gets useful text for places we don't have in the table.
    """
    if not name:
        return name
    m = _resolve_map(override_path)
    if name in m:
        return m[name]
    city = name.split(",")[0].strip()
    if city in m:
        return m[city]
    return name

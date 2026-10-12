from __future__ import annotations

from stock_multi_agent.models.chat import QueryIntent


INTENT_EXAMPLES: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.STOCK_ANALYSIS: (
        "phan tich co phieu nay", "danh gia ma nay", "con nay co on khong",
        "cho xin goc nhin ve ma", "co phieu nay dang nhu the nao", "ma nay ngon khong",
        "xem giup toi co phieu", "co dang de y khong", "sap toi co on khong", "nhan xet ma nay",
    ),
    QueryIntent.STOCK_CHART: (
        "cho toi bieu do", "ve chart gia", "cho xem duong gia", "show lich su gia",
        "gia may thang qua ra sao", "ve nen nhat", "nhin bien dong nua nam",
        "price chart please", "gia trong mot nam", "chart dau", "ve ca hai",
    ),
    QueryIntent.TECHNICAL_ANALYSIS: (
        "phan tich ky thuat", "xu huong gia hien tai", "dang uptrend hay downtrend",
        "momentum nhu the nao", "do thi dang dep khong", "price action ra sao",
        "xet ky thuat co khoe khong", "trend con nay", "tin hieu ky thuat",
    ),
    QueryIntent.FUNDAMENTAL_ANALYSIS: (
        "phan tich co ban", "nen tang doanh nghiep", "cong ty lam an the nao",
        "co phieu co dat khong", "dinh gia doanh nghiep", "loi nhuan ra sao",
        "tai chinh co khoe khong", "pe cao khong", "fundamental analysis",
    ),
    QueryIntent.RISK_ANALYSIS: (
        "co rui ro khong", "co nguy hiem khong", "diem xau la gi", "case xau nhat",
        "neu mua thi so gi", "can can than dieu gi", "downside ra sao",
        "rui ro mat von", "con nay cang khong", "worst case",
    ),
    QueryIntent.MARKET_OVERVIEW: (
        "thi truong hom nay sao", "chung khoan viet nam the nao", "vnindex khoe khong",
        "market dang dep khong", "hom nay do hay xanh", "thi truong dang trend gi",
        "tong quan thi truong", "chi so chung ra sao",
    ),
    QueryIntent.STOCK_DISCOVERY: (
        "co ma nao ngon khong", "goi y vai ma", "co con nao dang chu y",
        "tim co phieu nganh", "scan giup may ma", "tim vai ma tiem nang",
        "co phieu nao chi bao thap", "ma nao dang tot", "ma khac thi sao",
    ),
    QueryIntent.STOCK_RECOMMENDATION: (
        "cho toi ma co phieu nen mua hom nay", "goi y co phieu tot de dau tu",
        "ma nao dang mua nhat hien tai", "toi nen mua co phieu nao bay gio",
        "tim top co phieu de mua", "chon co phieu lai cao va co phieu an toan",
    ),
    QueryIntent.STOCK_COMPARISON: (
        "hai ma con nao on hon", "so sanh hai co phieu", "ma nay voi ma kia",
        "giua hai ma chon cai nao", "versus con nao hon", "compare with",
        "so voi ma vua roi", "con nao khoe hon", "so sanh hai ma nay",
    ),
    QueryIntent.INDICATOR_QUERY: (
        "rsi the nao", "macd ra sao", "cho xem thanh khoan", "ma20 dang o dau",
        "pe bao nhieu", "volume ma nay", "chi bao hien tai", "rsi cua co phieu",
    ),
    QueryIntent.RESEARCH: (
        "bao cao tai chinh moi nhat", "annual report cua doanh nghiep", "cong bo thong tin gan day",
        "nghi quyet hoi dong quan tri", "ban lanh dao noi gi", "tai lieu nha dau tu", "filing moi nhat",
    ),
    QueryIntent.THEORY: (
        "pe la gi", "roe la gi", "rsi duoc tinh the nao", "giai thich bao cao tai chinh",
        "dcf la gi", "tai sao chia co tuc lam gia giam", "sharpe ratio co y nghia gi",
        "phan biet roe va roa", "look ahead bias la gi", "cong thuc current ratio",
    ),
    QueryIntent.CURRENT_RULE: (
        "bien do hose hien tai bao nhieu", "ato va atc khac nhau the nao",
        "quy dinh giao dich hien hanh", "chu ky thanh toan hien tai", "gia tran gia san",
        "san hnx dang ap dung bien do bao nhieu", "lenh lo hien nay",
    ),
    QueryIntent.MIXED: (
        "roe cua fpt co tot khong", "ap dung pe de danh gia vnm", "rsi la gi va rsi fpt hien tai",
        "giai thich pb roi phan tich hpg", "dung dcf de danh gia co phieu fpt",
    ),
    QueryIntent.FOLLOW_UP: (
        "the con tiep theo", "con neu dai han", "noi them di", "thi sao nua",
        "con vua roi", "nua nam thoi", "what about it",
    ),
    QueryIntent.EXPLANATION: (
        "chi bao nay nghia la gi", "giai thich ket qua", "tai sao danh gia nhu vay",
        "dung de lam gi", "vi sao rui ro cao", "giai thich cho toi",
    ),
    QueryIntent.HELP: (
        "ban lam duoc gi", "huong dan su dung", "help me", "toi co the hoi gi",
        "co nhung loai phan tich nao", "tro giup",
    ),
    QueryIntent.GENERAL_CHAT: (
        "xin chao", "cam on nhe", "hello bot", "chao buoi sang", "ban khoe khong",
    ),
    QueryIntent.UNKNOWN: (
        "xem giup toi cai do", "lam cai kia", "toi khong biet", "abc xyz",
    ),
}

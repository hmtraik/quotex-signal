# zitro_bot.py

# -----------------------------------------------------------------------------
# القسم 1: الاستيرادات الأساسية والإعدادات الأولية
# -----------------------------------------------------------------------------
import asyncio
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests # For TwelveData example

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

# Technical Analysis Libraries
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands # AverageTrueRange لم تعد مستخدمة مباشرة هنا

# --- Configuration ---
try:
    import config # يفترض أن config.py في نفس المجلد
except ImportError:
    print("CRITICAL: config.py not found. Please create it.")
    exit()

# --- Quotex Broker Functions ---
# افترض أن broker_executor.py في نفس المجلد
try:
    import broker_executor # هذا هو الملف الذي يحتوي على دوال Quotex
except ImportError:
    print("WARNING: broker_executor.py not found. Quotex trading functionality will be disabled.")
    broker_executor = None # لتعطيل وظائف Quotex بأمان

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global state for Quotex driver (إذا كنت تستخدمه) ---
quotex_driver_instance = None
is_quotex_logged_in = False

# -----------------------------------------------------------------------------
# القسم 2: دوال جلب البيانات (كانت في data_fetcher.py)
# -----------------------------------------------------------------------------
def _fetch_twelvedata_candles_internal(api_key: str, symbol: str, interval: str, outputsize: int) -> pd.DataFrame | None:
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}&timezone=Etc/UTC"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok" and "values" in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df.sort_index(ascending=True, inplace=True)
            return df
        else:
            logger.error(f"TwelveData API error for {symbol}: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        logger.error(f"TwelveData request failed for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing TwelveData response for {symbol}: {e}", exc_info=True)
        return None

def fetch_data_from_source(source: str, symbol: str, interval_or_timeframe: str, count: int, asset_config: dict) -> pd.DataFrame | None:
    logger.debug(f"Data Fetcher: Requesting {count} candles for {symbol} on {interval_or_timeframe} from {source}")
    df = None
    if source.lower() == "twelvedata":
        td_symbol = asset_config.get("TWELVEDATA_SYMBOL", symbol)
        # استخدام config.TWELVEDATA_API_KEY مباشرة
        df = _fetch_twelvedata_candles_internal(config.TWELVEDATA_API_KEY, td_symbol, interval_or_timeframe, count)
    # elif source.lower() == "iqoption":
    #     # ... (منطق IQ Option هنا إذا لزم الأمر)
    #     logger.info("IQ Option data fetching not fully implemented in this example.")
    else:
        logger.warning(f"Data source '{source}' not supported.")

    if df is not None and not df.empty:
        logger.debug(f"Data Fetcher: Successfully fetched {len(df)} candles for {symbol}.")
    elif df is not None and df.empty:
        logger.warning(f"Data Fetcher: Fetched 0 candles for {symbol}.")
    else:
        logger.warning(f"Data Fetcher: Failed to fetch data for {symbol}.")
    return df

# -----------------------------------------------------------------------------
# القسم 3: دوال أنماط الشموع (كانت في strategy/candle_patterns.py)
# -----------------------------------------------------------------------------
def detect_candlestick_patterns(
    df_input: pd.DataFrame,
    doji_threshold: float = 0.1,
    hammer_body_max_ratio: float = 0.4,
    hammer_lower_shadow_min_ratio: float = 0.5,
    hammer_upper_shadow_max_ratio: float = 0.2,
) -> pd.DataFrame:
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df_input.columns for col in required_columns):
        logger.error("DataFrame is missing OHLC columns for pattern detection.")
        empty_patterns = pd.DataFrame(columns=[
            'bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star', 'doji',
            'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'
        ], index=df_input.index)
        return empty_patterns.fillna(False)

    df = df_input.copy()
    candle_body_size = np.abs(df['close'] - df['open'])
    candle_total_range = df['high'] - df['low']
    candle_total_range_safe = candle_total_range.replace(0, np.nan)
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']

    df['doji'] = (candle_body_size / candle_total_range_safe) < doji_threshold
    is_hammer_shape = (
        (candle_body_size / candle_total_range_safe < hammer_body_max_ratio) &
        (lower_shadow / candle_total_range_safe > hammer_lower_shadow_min_ratio) &
        (upper_shadow / candle_total_range_safe < hammer_upper_shadow_max_ratio)
    )
    df['hammer'] = is_hammer_shape
    is_shooting_star_shape = (
        (candle_body_size / candle_total_range_safe < hammer_body_max_ratio) &
        (upper_shadow / candle_total_range_safe > hammer_lower_shadow_min_ratio) &
        (lower_shadow / candle_total_range_safe < hammer_upper_shadow_max_ratio)
    )
    df['shooting_star'] = is_shooting_star_shape
    df['bullish_engulfing'] = (
        (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) &
        (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
    )
    df['bearish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) &
        (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    )
    prev_is_bearish = df['close'].shift(2) < df['open'].shift(2)
    middle_is_small_body_star = (df['doji'].shift(1) | df['hammer'].shift(1))
    current_is_bullish = df['close'] > df['open']
    current_closes_in_first_body = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
    df['morning_star'] = prev_is_bearish & middle_is_small_body_star & current_is_bullish & current_closes_in_first_body
    prev_is_bullish = df['close'].shift(2) > df['open'].shift(2)
    middle_is_small_body_star_evening = (df['doji'].shift(1) | df['shooting_star'].shift(1))
    current_is_bearish = df['close'] < df['open']
    current_closes_in_first_body_evening = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
    df['evening_star'] = prev_is_bullish & middle_is_small_body_star_evening & current_is_bearish & current_closes_in_first_body_evening
    is_bullish_candle = df['close'] > df['open']
    prev_is_bullish_candle = df['close'].shift(1) > df['open'].shift(1)
    prev_prev_is_bullish_candle = df['close'].shift(2) > df['open'].shift(2)
    closes_are_higher = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
    df['three_white_soldiers'] = is_bullish_candle & prev_is_bullish_candle & prev_prev_is_bullish_candle & closes_are_higher
    is_bearish_candle = df['close'] < df['open']
    prev_is_bearish_candle = df['close'].shift(1) < df['open'].shift(1)
    prev_prev_is_bearish_candle = df['close'].shift(2) < df['open'].shift(2)
    closes_are_lower = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    df['three_black_crows'] = is_bearish_candle & prev_is_bearish_candle & prev_prev_is_bearish_candle & closes_are_lower
    pattern_columns = ['bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star', 'doji', 'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows']
    for col in pattern_columns:
        if col in df.columns: df[col] = df[col].fillna(False)
        else: df[col] = False
    return df[pattern_columns]


# -----------------------------------------------------------------------------
# القسم 4: دوال المؤشرات الفردية وتجميع الإشارة (كانت في indicators.py)
# -----------------------------------------------------------------------------
def ind_ema_cross(df: pd.DataFrame, short_window=20, long_window=50) -> str:
    if len(df) < long_window: return "NEUTRAL"
    ema_short = EMAIndicator(df['close'], window=short_window, fillna=True).ema_indicator()
    ema_long = EMAIndicator(df['close'], window=long_window, fillna=True).ema_indicator()
    if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]: return "BUY"
    elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]: return "SELL"
    elif ema_short.iloc[-1] > ema_long.iloc[-1]: return "BUY"
    elif ema_short.iloc[-1] < ema_long.iloc[-1]: return "SELL"
    return "NEUTRAL"

def ind_ichimoku_tenkan(df: pd.DataFrame, window=9) -> str:
    if len(df) < window: return "NEUTRAL"
    tenkan = (df['high'].rolling(window).max() + df['low'].rolling(window).min()) / 2
    if df['close'].iloc[-1] > tenkan.iloc[-1] and df['open'].iloc[-1] <= tenkan.iloc[-1]: return "BUY"
    elif df['close'].iloc[-1] < tenkan.iloc[-1] and df['open'].iloc[-1] >= tenkan.iloc[-1]: return "SELL"
    elif df['close'].iloc[-1] > tenkan.iloc[-1]: return "BUY"
    elif df['close'].iloc[-1] < tenkan.iloc[-1]: return "SELL"
    return "NEUTRAL"

def ind_rsi(df: pd.DataFrame, window=9, oversold=30, overbought=70) -> str:
    if len(df) < window: return "NEUTRAL"
    rsi_val = RSIIndicator(df['close'], window=window, fillna=True).rsi()
    if rsi_val.iloc[-1] < oversold and rsi_val.iloc[-2] >= oversold : return "BUY"
    elif rsi_val.iloc[-1] > overbought and rsi_val.iloc[-2] <= overbought: return "SELL"
    return "NEUTRAL"

def ind_macd(df: pd.DataFrame, window_slow=26, window_fast=12, window_sign=9) -> str:
    if len(df) < window_slow: return "NEUTRAL"
    macd_obj = MACD(df['close'], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=True)
    macd_line, signal_line = macd_obj.macd(), macd_obj.macd_signal()
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]: return "BUY"
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]: return "SELL"
    elif macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-1] > 0 : return "BUY"
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-1] < 0 : return "SELL"
    return "NEUTRAL"

def ind_stochastic(df: pd.DataFrame, window=5, smooth_window=3, oversold=20, overbought=80) -> str:
    if len(df) < window + smooth_window: return "NEUTRAL"
    stoch_k = StochasticOscillator(df['high'], df['low'], df['close'], window=window, smooth_window=smooth_window, fillna=True).stoch()
    if stoch_k.iloc[-1] > oversold and stoch_k.iloc[-2] <= oversold: return "BUY"
    elif stoch_k.iloc[-1] < overbought and stoch_k.iloc[-2] >= overbought: return "SELL"
    return "NEUTRAL"

def ind_obv_trend(df: pd.DataFrame, short_window=20, long_window=50) -> str:
    if 'volume' not in df.columns or df['volume'].isnull().all() or len(df) < long_window: return "NEUTRAL"
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'], fillna=True).on_balance_volume()
    obv_sma_short, obv_sma_long = obv.rolling(window=short_window).mean(), obv.rolling(window=long_window).mean()
    if obv_sma_short.iloc[-1] > obv_sma_long.iloc[-1] and obv_sma_short.iloc[-2] <= obv_sma_long.iloc[-2]: return "BUY"
    elif obv_sma_short.iloc[-1] < obv_sma_long.iloc[-1] and obv_sma_short.iloc[-2] >= obv_sma_long.iloc[-2]: return "SELL"
    elif obv_sma_short.iloc[-1] > obv_sma_long.iloc[-1]: return "BUY"
    elif obv_sma_short.iloc[-1] < obv_sma_long.iloc[-1]: return "SELL"
    return "NEUTRAL"

def ind_bollinger_touch(df: pd.DataFrame, window=20, window_dev=2) -> str:
    if len(df) < window: return "NEUTRAL"
    boll = BollingerBands(df['close'], window=window, window_dev=window_dev, fillna=True)
    lower_band, upper_band = boll.bollinger_lband(), boll.bollinger_hband()
    if df['low'].iloc[-1] <= lower_band.iloc[-1] and df['close'].iloc[-1] > df['open'].iloc[-1]: return "BUY"
    elif df['high'].iloc[-1] >= upper_band.iloc[-1] and df['close'].iloc[-1] < df['open'].iloc[-1]: return "SELL"
    return "NEUTRAL"

def ind_candlestick_bullish(df: pd.DataFrame) -> str:
    patterns = detect_candlestick_patterns(df.copy())
    last_patterns = patterns.iloc[-1]
    if last_patterns['bullish_engulfing'] or last_patterns['hammer'] or last_patterns['morning_star'] or last_patterns['three_white_soldiers']:
        return "BUY"
    return "NEUTRAL"

def ind_candlestick_bearish(df: pd.DataFrame) -> str:
    patterns = detect_candlestick_patterns(df.copy())
    last_patterns = patterns.iloc[-1]
    if last_patterns['bearish_engulfing'] or last_patterns['shooting_star'] or last_patterns['evening_star'] or last_patterns['three_black_crows']:
        return "SELL"
    return "NEUTRAL"

def ind_adx_trend_simple(df: pd.DataFrame, window=14, strength_threshold=25) -> str:
    if len(df) < window * 2 : return "NEUTRAL"
    adx_obj = ADXIndicator(df['high'], df['low'], df['close'], window=window, fillna=True)
    adx_val, pdi_val, ndi_val = adx_obj.adx(), adx_obj.adx_pos(), adx_obj.adx_neg()
    if adx_val.iloc[-1] > strength_threshold:
        if pdi_val.iloc[-1] > ndi_val.iloc[-1]: return "BUY"
        elif ndi_val.iloc[-1] > pdi_val.iloc[-1]: return "SELL"
    return "NEUTRAL"

ALL_INDICATOR_FUNCTIONS = [
    ind_ema_cross, ind_ichimoku_tenkan, ind_rsi, ind_macd, ind_stochastic,
    ind_obv_trend, ind_bollinger_touch, ind_candlestick_bullish,
    ind_candlestick_bearish, ind_adx_trend_simple,
]

def get_final_signal_decision(df: pd.DataFrame, signal_threshold: int) -> tuple[str, dict]:
    buy_strength, sell_strength, neutral_count = 0, 0, 0
    individual_signals = {}
    if df is None or df.empty or len(df) < 50:
        return "NEUTRAL", {"error": "Not enough data for analysis."}

    for func in ALL_INDICATOR_FUNCTIONS:
        name = func.__name__.replace("ind_", "").replace("_", " ").title()
        try:
            signal = func(df.copy())
            individual_signals[name] = signal
            if signal == "BUY": buy_strength += 1
            elif signal == "SELL": sell_strength += 1
            else: neutral_count += 1
        except Exception as e:
            logger.error(f"Error in indicator {name}: {e}", exc_info=True)
            individual_signals[name] = "ERROR"
            neutral_count += 1
    
    final_signal = "NEUTRAL"
    if buy_strength >= signal_threshold and buy_strength >= sell_strength: final_signal = "BUY"
    elif sell_strength >= signal_threshold: final_signal = "SELL"
    return final_signal, individual_signals

# -----------------------------------------------------------------------------
# القسم 5: منطق البوت الأساسي (كان في main_bot.py)
# -----------------------------------------------------------------------------
def get_asset_config_by_common_name(common_name: str) -> dict | None:
    for asset_conf in config.ASSETS_TO_MONITOR:
        if asset_conf["COMMON_NAME"].upper() == common_name.upper():
            return asset_conf
    return None

def analyze_single_asset_timeframe(asset_common_name: str, timeframe: str) -> tuple[str, dict]:
    logger.info(f"Analyzing {asset_common_name} on timeframe {timeframe}...")
    asset_conf = get_asset_config_by_common_name(asset_common_name)
    if not asset_conf:
        return "NEUTRAL", {"error": f"Asset config not found for {asset_common_name}"}

    df_data = fetch_data_from_source(
        source=config.ACTIVE_DATA_SOURCE,
        symbol=asset_common_name,
        interval_or_timeframe=timeframe,
        count=config.CANDLE_COUNT_TO_FETCH,
        asset_config=asset_conf
    )
    if df_data is None or df_data.empty:
        return "NEUTRAL", {"error": f"No data for {asset_common_name}@{timeframe}"}
    
    min_required_data = max(50, config.CANDLE_COUNT_TO_FETCH // 2)
    if len(df_data) < min_required_data:
        return "NEUTRAL", {"error": f"Insufficient data points ({len(df_data)})"}

    final_signal, individual_signals = get_final_signal_decision(df_data, config.SIGNAL_THRESHOLD)
    logger.info(f"Signal for {asset_common_name}@{timeframe}: {final_signal}. Details: {individual_signals}")
    return final_signal, individual_signals

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        rf"أهلاً {user.mention_html()}! أنا بوت تحليل الإشارات المدمج."
        f"\nاستخدم /check SYMBOL (مثال: <code>/check EUR/USD</code>) لتحليل فوري."
        f"\nأو /status لعرض حالة البوت.",
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global is_quotex_logged_in # تأكد من أن هذا المتغير العام محدث
    assets_str = ", ".join([a['COMMON_NAME'] for a in config.ASSETS_TO_MONITOR])
    frames_str = ", ".join(config.ANALYSIS_FRAMES)
    
    quotex_status = "وحدة Quotex غير مفعلة"
    if broker_executor: # تحقق مما إذا تم استيراد الوحدة بنجاح
        quotex_status = "مسجل الدخول" if is_quotex_logged_in else "غير مسجل الدخول / خطأ"
    
    num_total_indicators = len(ALL_INDICATOR_FUNCTIONS)
    status_message = (
        f"📊 **حالة البوت** 📊\n"
        f"▫️ مصدر البيانات: <code>{config.ACTIVE_DATA_SOURCE}</code>\n"
        f"▫️ الأصول المراقبة: <code>{assets_str or 'لا يوجد'}</code>\n"
        f"▫️ الأطر الزمنية: <code>{frames_str or 'لا يوجد'}</code>\n"
        f"▫️ عتبة الإشارة: <b>{config.SIGNAL_THRESHOLD}</b> من <b>{num_total_indicators}</b> مؤشرات\n"
        f"▫️ حالة Quotex: <code>{quotex_status}</code>\n"
        f"▫️ التداول التلقائي: {'مفعل (إذا تم تسجيل الدخول)' if broker_executor and config.QUOTEX_EMAIL != 'your_actual_quotex_email@example.com' else 'غير مفعل'}"
    )
    await update.message.reply_text(status_message, parse_mode=ParseMode.HTML)

async def check_asset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("الاستخدام: /check <SYMBOL>\nمثال: <code>/check EUR/USD</code>", parse_mode=ParseMode.HTML)
        return

    asset_to_check_common = context.args[0].upper()
    asset_conf = get_asset_config_by_common_name(asset_to_check_common)
    if not asset_conf:
        await update.message.reply_text(f"لم يتم العثور على الأصل <code>{asset_to_check_common}</code>.", parse_mode=ParseMode.HTML)
        return

    await update.message.reply_text(f"🔍 جاري تحليل <code>{asset_to_check_common}</code> على الأطر: {', '.join(config.ANALYSIS_FRAMES)}...", parse_mode=ParseMode.HTML)
    results_summary_parts = [f"<b>تحليل لـ {asset_to_check_common}:</b>"]
    overall_signal_consensus = []

    for frame in config.ANALYSIS_FRAMES:
        signal, individual_details = analyze_single_asset_timeframe(asset_to_check_common, frame)
        overall_signal_consensus.append(signal)
        buy_count = sum(1 for s in individual_details.values() if s == "BUY")
        sell_count = sum(1 for s in individual_details.values() if s == "SELL")
        other_count = len(individual_details) - buy_count - sell_count
        frame_summary = [
            f"\n--- <u>الإطار: {frame}</u> ---",
            f"<b>الإشارة: {signal}</b> (شراء: {buy_count}, بيع: {sell_count}, أخرى: {other_count})"
        ]
        results_summary_parts.extend(frame_summary)

    final_consolidated_signal = "NEUTRAL"
    if overall_signal_consensus and all(s == overall_signal_consensus[0] for s in overall_signal_consensus) and overall_signal_consensus[0] != "NEUTRAL":
        final_consolidated_signal = overall_signal_consensus[0]
        results_summary_parts.append(f"\n🏁 <b>الإشارة المجمعة (توافق الأطر): {final_consolidated_signal}</b>")
    else:
        results_summary_parts.append(f"\n⚠️ <b>لا يوجد توافق قوي عبر الأطر.</b>")
    await update.message.reply_html("\n".join(results_summary_parts))

async def background_analysis_loop(application: Application) -> None:
    global quotex_driver_instance, is_quotex_logged_in # تأكد من استخدام global
    logger.info("Background analysis loop started.")
    iteration_count = 0
    while True:
        iteration_count += 1
        logger.info(f"Background Iteration #{iteration_count}...")
        start_time_loop = time.time()

        for asset_config_item in config.ASSETS_TO_MONITOR:
            asset_common_name = asset_config_item["COMMON_NAME"]
            asset_quotex_name = asset_config_item.get("QUOTEX_SYMBOL", asset_common_name)
            logger.info(f"Analyzing {asset_common_name} for background signals...")
            
            signals_from_frames = {}
            individual_reports = {}

            for frame in config.ANALYSIS_FRAMES:
                signal, individual_details = analyze_single_asset_timeframe(asset_common_name, frame)
                signals_from_frames[frame] = signal
                individual_reports[frame] = individual_details
            
            final_trade_signal = "NEUTRAL"
            unique_signals = set(s for s in signals_from_frames.values() if s != "NEUTRAL")
            if len(unique_signals) == 1:
                final_trade_signal = unique_signals.pop()
            
            if final_trade_signal != "NEUTRAL":
                message_parts = [
                    f"🔔 <b>إشارة تداول لـ {asset_common_name}! الاتجاه: {final_trade_signal.upper()}</b>",
                    f"<i>(عتبة: {config.SIGNAL_THRESHOLD}/{len(ALL_INDICATOR_FUNCTIONS)} مؤشرات لكل إطار)</i>"
                ]
                for fr, details in individual_reports.items():
                    if not isinstance(details, dict) : continue
                    buy_s = sum(1 for s_val in details.values() if s_val == "BUY")
                    sell_s = sum(1 for s_val in details.values() if s_val == "SELL")
                    message_parts.append(f"\n<u>{fr}:</u> ش: {buy_s}, ب: {sell_s} (إشارة: {signals_from_frames.get(fr, 'N/A')})")
                
                tg_message = "\n".join(message_parts)
                if application.bot and config.TELEGRAM_CHAT_ID:
                    try:
                        await application.bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=tg_message, parse_mode=ParseMode.HTML)
                    except Exception as e_tg: logger.error(f"Telegram send error: {e_tg}", exc_info=True)
                
                if broker_executor and config.QUOTEX_EMAIL != 'your_actual_quotex_email@example.com':
                    if not is_quotex_logged_in and quotex_driver_instance:
                        is_quotex_logged_in = broker_executor.login_quotex(quotex_driver_instance, config.QUOTEX_EMAIL, config.QUOTEX_PASSWORD)
                    
                    if is_quotex_logged_in and quotex_driver_instance:
                        quotex_direction = "call" if final_trade_signal == "BUY" else "put"
                        trade_success = broker_executor.place_trade(
                            driver=quotex_driver_instance, asset=asset_quotex_name, 
                            direction=quotex_direction, amount=config.TRADE_AMOUNT, duration=config.TRADE_DURATION
                        )
                        if trade_success:
                            logger.info(f"Quotex: Trade PLACED for {asset_quotex_name} - {quotex_direction}.")
                            await application.bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=f"✅ صفقة {quotex_direction.upper()}/{asset_quotex_name} في Quotex.")
                        else:
                            logger.warning(f"Quotex: Trade FAILED for {asset_quotex_name}.")
                            is_quotex_logged_in = False
                            await application.bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=f"❌ فشل صفقة {quotex_direction.upper()}/{asset_quotex_name} في Quotex.")
            await asyncio.sleep(2)

        loop_duration = time.time() - start_time_loop
        desired_cycle_time = 60 * 2 # دورة كل دقيقتين كمثال
        sleep_time = max(10, desired_cycle_time - loop_duration)
        logger.info(f"Background iteration completed in {loop_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
        await asyncio.sleep(sleep_time)

async def post_bot_init_setup(application: Application) -> None:
    global quotex_driver_instance, is_quotex_logged_in
    logger.info("Bot post_init: Performing initial setup...")
    if broker_executor and config.QUOTEX_EMAIL != 'your_actual_quotex_email@example.com':
        logger.info("Quotex: Setting up browser and attempting initial login...")
        try:
            quotex_driver_instance = broker_executor.setup_browser(headless=True) # True للتشغيل بدون واجهة
            if quotex_driver_instance:
                is_quotex_logged_in = broker_executor.login_quotex(quotex_driver_instance, config.QUOTEX_EMAIL, config.QUOTEX_PASSWORD)
                if is_quotex_logged_in: logger.info("Quotex: Initial login successful.")
                else: logger.warning("Quotex: Initial login failed.")
            else: logger.error("Quotex: Browser setup failed.")
        except Exception as e_qx_setup:
            logger.error(f"Quotex: Error during initial setup/login: {e_qx_setup}", exc_info=True)
            if quotex_driver_instance: broker_executor.close_browser(quotex_driver_instance)
            quotex_driver_instance = None
            is_quotex_logged_in = False
    
    asyncio.create_task(background_analysis_loop(application))
    logger.info("Background analysis loop scheduled.")

def main() -> None:
    logger.info("Starting Zitro Bot (Consolidated Version)...")
    if not config.TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in config.TELEGRAM_BOT_TOKEN :
        logger.critical("CRITICAL: TELEGRAM_BOT_TOKEN missing or placeholder. Bot cannot start.")
        return

    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(post_bot_init_setup).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("check", check_asset_command))

    logger.info("Telegram Bot application configured. Starting polling...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Bot polling CRASHED: {e}", exc_info=True)
    finally:
        global quotex_driver_instance
        if quotex_driver_instance and broker_executor:
            logger.info("Shutting down: Closing Quotex browser...")
            broker_executor.close_browser(quotex_driver_instance)
        logger.info("Bot shut down.")

if __name__ == "__main__":
    print("=============================================")
    print("   Zitro Signal Bot - Consolidated Version   ")
    print("=============================================")
    # ... (يمكنك إضافة طباعة تفاصيل الإعدادات هنا كما في السابق)
    main()
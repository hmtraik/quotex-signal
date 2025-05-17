# config.py

# --- TwelveData API Key ---
TWELVEDATA_API_KEY = "b7ef1a4efe984f93a02cf9a5653e3621" # !!! استخدم مفتاحك الفعلي إذا كان لديك واحد !!!
# إذا لم يكن لديك مفتاح، يمكنك الحصول على واحد مجاني من موقع TwelveData للاختبار

# --- Quotex Credentials ---
# !!! استبدل ببياناتك الحقيقية أو اتركها كـ placeholders إذا كنت لا تريد التداول التلقائي الآن !!!
# إذا تركتها placeholders، لن يحاول البوت تسجيل الدخول أو التداول على Quotex.
QUOTEX_EMAIL = "hmtraik@gmail.com"
QUOTEX_PASSWORD = "0668526953"

# --- Telegram Settings ---
TELEGRAM_BOT_TOKEN = "7868153790:AAFLWOUzAaR5kcmXDG1cy6Or2T3xtjq4PEU" # توكن البوت الخاص بك
TELEGRAM_CHAT_ID = "@zitro_signal" # أو Chat ID رقمي لمجموعة خاصة أو مستخدم
# للحصول على Chat ID لمجموعة، يمكنك إضافة البوت للمجموعة ثم إرسال رسالة من البوت
# والتحقق من الـ API response (مثلاً باستخدام @get_id_bot في تيليجرام)
# أو إذا كانت قناة عامة، فاسم المستخدم @YourChannelName كافٍ.

# --- Trading & Analysis Settings ---
SIGNAL_THRESHOLD = 5  # عدد المؤشرات المطلوبة للتوافق (مثلاً 5 من 10)
TRADE_AMOUNT = 1      # مثال: مبلغ التداول بالدولار (تأكد أنه مناسب لحسابك)
TRADE_DURATION = "1m" # مثال: مدة الصفقة (يجب أن يدعمها broker_executor.py، حاليًا "1m" مدعومة)

# --- Data Fetching Settings ---
ACTIVE_DATA_SOURCE = "twelvedata" # حاليًا الكود يدعم "twelvedata" بشكل رئيسي
ASSETS_TO_MONITOR = [
    # يجب أن يكون COMMON_NAME فريدًا ويستخدم في الأوامر مثل /check EUR/USD
    # TWELVEDATA_SYMBOL هو الرمز المستخدم لواجهة برمجة تطبيقات TwelveData
    # QUOTEX_SYMBOL هو الاسم كما يظهر في منصة Quotex عند البحث (مهم للتداول)
    {"COMMON_NAME": "EUR/USD", "TWELVEDATA_SYMBOL": "EUR/USD", "QUOTEX_SYMBOL": "EUR/USD"},
    {"COMMON_NAME": "GBP/USD", "TWELVEDATA_SYMBOL": "GBP/USD", "QUOTEX_SYMBOL": "GBP/USD"},
    {"COMMON_NAME": "AUD/USD", "TWELVEDATA_SYMBOL": "AUD/USD", "QUOTEX_SYMBOL": "AUD/USD"},
    # يمكنك إضافة المزيد من الأصول هنا، مثل:
    # {"COMMON_NAME": "USD/JPY", "TWELVEDATA_SYMBOL": "USD/JPY", "QUOTEX_SYMBOL": "USD/JPY"},
    # {"COMMON_NAME": "BTC/USD", "TWELVEDATA_SYMBOL": "BTC/USD", "QUOTEX_SYMBOL": "BTCUSD"}, # انتبه لـ QUOTEX_SYMBOL للعملات الرقمية
]
ANALYSIS_FRAMES = ["5min", "15min"] # الأطر الزمنية للتحليل لكل أصل
CANDLE_COUNT_TO_FETCH = 200       # عدد الشموع لجلبها (للمؤشرات التي تحتاج تاريخ أطول)

# --- Quotex Selectors (المعرفات الخاصة بواجهة Quotex) ---
# هذه المعرفات حساسة جدًا للتغييرات في واجهة Quotex. يجب التحقق منها بانتظام.
QUOTEX_LOGIN_URL = "https://qxbroker.com/en/sign-in" # أو الرابط الصحيح لمنطقتك/لغتك
QUOTEX_EMAIL_INPUT_NAME = "email" # اسم حقل الإيميل
QUOTEX_PASSWORD_INPUT_NAME = "password" # اسم حقل كلمة المرور
# عنصر يظهر بعد تسجيل الدخول بنجاح (مثال: صورة الأفاتار أو رصيد الحساب)
QUOTEX_LOGIN_SUCCESS_INDICATOR_XPATH = "//div[contains(@class,'header-avatar__photo') or contains(@class,'user-balance__value') or @data-testid='header-profile-avatar']" # مثال مركب، اختر الأنسب
QUOTEX_ASSET_SELECTOR_BUTTON_CSS = "button.pair-button" # زر فتح قائمة اختيار الأصول
QUOTEX_ASSET_SEARCH_MODAL_CSS = "div.search-modal" # النافذة المنبثقة للبحث عن الأصول
QUOTEX_ASSET_SEARCH_INPUT_CSS = "div.search-modal input[type='text']" # حقل البحث داخل النافذة
# XPATH للعنصر الذي يتم النقر عليه لاختيار الأصل بعد البحث. {} ستُستبدل باسم الأصل.
QUOTEX_ASSET_SEARCH_RESULT_XPATH_TEMPLATE = "//div[contains(@class,'asset-item')]//div[contains(normalize-space(),'{}')]"
# زر فتح قائمة اختيار مدة الصفقة (قد لا يكون ضروريًا إذا كانت أزرار المدد ظاهرة دائمًا)
# QUOTEX_TIME_SELECTOR_BUTTON_CSS = "button.time-button"
# XPATH لزر مدة دقيقة واحدة (النص قد يختلف: "1:00" أو "1m" أو رمز الساعة)
QUOTEX_DURATION_1M_BUTTON_XPATH = "//button[contains(@class,'time-item') and (normalize-space(.)='1:00' or normalize-space(.)='1m')]"
QUOTEX_AMOUNT_INPUT_CSS = "input.input-sum" # حقل إدخال مبلغ الصفقة
QUOTEX_CALL_BUTTON_CSS = "button.deal-button--up" # زر الشراء (الأخضر)
QUOTEX_PUT_BUTTON_CSS = "button.deal-button--down" # زر البيع (الأحمر)

# --- Selenium Settings (اختياري) ---
# إذا كنت تريد التحكم في مسار chromedriver يدويًا بدلاً من webdriver-manager
# CHROMEDRIVER_PATH = "/path/to/your/chromedriver" # مثال: "C:/webdrivers/chromedriver.exe"
# اتركه معلقًا (commented out) لاستخدام webdriver-manager الافتراضي
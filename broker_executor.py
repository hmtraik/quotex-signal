# broker_executor.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, NoSuchElementException
import time
import logging # إضافة تسجيل الأخطاء هنا أيضًا

logger = logging.getLogger(__name__) # استخدام logger خاص بهذه الوحدة

# --- يمكنك استيراد هذه من config.py إذا نقلتها هناك ---
# أو استخدامها مباشرة من config إذا تم استيراد config في هذا الملف
try:
    import config # لاستخدام المتغيرات من config.py
    LOGIN_URL = config.QUOTEX_LOGIN_URL
    EMAIL_INPUT_NAME = config.QUOTEX_EMAIL_INPUT_NAME
    PASSWORD_INPUT_NAME = config.QUOTEX_PASSWORD_INPUT_NAME
    LOGIN_SUCCESS_INDICATOR_XPATH = config.QUOTEX_LOGIN_SUCCESS_INDICATOR_XPATH
    ASSET_SELECTOR_BUTTON_CSS = config.QUOTEX_ASSET_SELECTOR_BUTTON_CSS
    ASSET_SEARCH_MODAL_CSS = config.QUOTEX_ASSET_SEARCH_MODAL_CSS
    ASSET_SEARCH_INPUT_CSS = config.QUOTEX_ASSET_SEARCH_INPUT_CSS
    ASSET_SEARCH_RESULT_XPATH_TEMPLATE = config.QUOTEX_ASSET_SEARCH_RESULT_XPATH_TEMPLATE
    DURATION_1M_BUTTON_XPATH = config.QUOTEX_DURATION_1M_BUTTON_XPATH
    AMOUNT_INPUT_CSS = config.QUOTEX_AMOUNT_INPUT_CSS
    CALL_BUTTON_CSS = config.QUOTEX_CALL_BUTTON_CSS
    PUT_BUTTON_CSS = config.QUOTEX_PUT_BUTTON_CSS
except ImportError:
    logger.critical("broker_executor: Could not import config.py. Using hardcoded fallbacks (NOT RECOMMENDED).")
    # ضع هنا قيمًا افتراضية إذا فشل الاستيراد (غير موصى به للإنتاج)
    LOGIN_URL = "https://qxbroker.com/en/sign-in" # مثال
    # ... وهكذا للبقية

# --- بقية دوال Quotex كما كانت ---
# setup_browser, login_quotex, select_asset, set_trade_duration,
# set_trade_amount, place_trade, close_browser
# تأكد من أن هذه الدوال تستخدم logger.info(), logger.warning(), logger.error() بدلاً من print()

def setup_browser(headless: bool = True, browser_type: str = "chrome") -> webdriver.Chrome | None:
    try:
        logger.info(f"Quotex: Setting up {browser_type} browser (headless={headless})...")
        if browser_type.lower() == "chrome":
            options = Options()
            if headless:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu") # مهم للـ headless
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--no-sandbox") # مهم لبيئات مثل Docker أو Linux servers
            options.add_argument("--disable-dev-shm-usage") # لتجنب مشاكل الذاكرة المشتركة
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
            # إذا كنت تستخدم chromedriver مثبت محليًا:
            # driver = webdriver.Chrome(options=options)
            # إذا كنت تستخدم webdriver-manager:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            try:
                driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
                logger.info("Quotex: Chrome browser initialized using webdriver-manager.")
            except Exception as e_webdriver_manager:
                logger.error(f"Quotex: Failed to initialize Chrome with webdriver-manager: {e_webdriver_manager}. Trying default path.")
                # محاولة استخدام المسار الافتراضي إذا فشل webdriver-manager
                driver = webdriver.Chrome(options=options)
                logger.info("Quotex: Chrome browser initialized using default path.")

            return driver
        else:
            logger.error(f"Quotex: Browser type '{browser_type}' not supported.")
            return None
    except Exception as e:
        logger.error(f"Quotex: Error setting up browser: {e}", exc_info=True)
        return None

def login_quotex(driver: webdriver.Chrome, email: str, password: str, timeout: int = 45) -> bool:
    if not driver: return False
    try:
        logger.info(f"Quotex: Navigating to login page: {LOGIN_URL}")
        driver.get(LOGIN_URL)
        
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.NAME, EMAIL_INPUT_NAME))
        ).send_keys(email)
        logger.debug("Quotex: Email entered.")

        driver.find_element(By.NAME, PASSWORD_INPUT_NAME).send_keys(password)
        logger.debug("Quotex: Password entered.")
        
        # محاولة النقر على زر تسجيل الدخول الأكثر شيوعًا
        try:
            # محاولة تحديد زر submit بشكل عام أكثر
            login_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "form button[type='submit'], button.button--primary")) # محاولة CSS أعم
            )
            login_button.click()
            logger.debug("Quotex: Clicked login button (CSS).")
        except TimeoutException:
            logger.warning("Quotex: Login button by CSS not found or not clickable, trying submit on password field.")
            try:
                driver.find_element(By.NAME, PASSWORD_INPUT_NAME).submit()
                logger.debug("Quotex: Submitted password field.")
            except Exception as e_submit:
                 logger.error(f"Quotex: Failed to submit password field: {e_submit}")
                 # قد يكون هناك زر تسجيل دخول آخر، أو أن الصفحة لم تحمل بشكل صحيح
                 # driver.save_screenshot("quotex_login_submit_fail.png")
                 return False


        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, LOGIN_SUCCESS_INDICATOR_XPATH))
        )
        logger.info("Quotex: Login successful.")
        return True
    except TimeoutException:
        logger.error("Quotex: Timeout during login process. Page might not have loaded correctly or selectors are wrong.")
        # driver.save_screenshot("quotex_login_timeout.png") # جيد للتصحيح
    except Exception as e:
        logger.error(f"Quotex: Error during login: {e}", exc_info=True)
        # driver.save_screenshot("quotex_login_error.png")
    return False

def select_asset(driver: webdriver.Chrome, asset_name_quotex: str, timeout: int = 20) -> bool:
    if not driver: return False
    try:
        logger.info(f"Quotex: Selecting asset '{asset_name_quotex}'...")
        # 1. انقر على زر اختيار الأصل لفتح القائمة/النافذة
        asset_selector_btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ASSET_SELECTOR_BUTTON_CSS))
        )
        asset_selector_btn.click()
        logger.debug("Quotex: Clicked asset selector button.")

        # 2. انتظر ظهور نافذة البحث عن الأصول
        WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ASSET_SEARCH_MODAL_CSS))
        )
        logger.debug("Quotex: Asset search modal is visible.")
        
        # 3. أدخل اسم الأصل في حقل البحث
        search_input = driver.find_element(By.CSS_SELECTOR, ASSET_SEARCH_INPUT_CSS)
        search_input.clear()
        # Quotex عادة لا تستخدم "/" في البحث، لكن اسم الأصل المعروض قد يحتوي عليها.
        # قد تحتاج إلى تطبيع asset_name_quotex قبل البحث (مثال: إزالة "/")
        search_input.send_keys(asset_name_quotex.replace("/", "")) # مثال: إزالة "/"
        logger.debug(f"Quotex: Typed '{asset_name_quotex.replace('/', '')}' into asset search.")
        time.sleep(1) # انتظر قليلاً لفلترة النتائج

        # 4. انقر على الأصل المطابق في النتائج
        # asset_name_quotex يجب أن يكون النص *المعروض* في القائمة
        # قد تحتاج إلى تعديل هذا XPATH ليكون أكثر مرونة (مثل تجاهل الفراغات الزائدة)
        asset_xpath = ASSET_SEARCH_RESULT_XPATH_TEMPLATE.format(asset_name_quotex)
        
        # قد تحتاج إلى انتظار حتى يصبح العنصر قابلاً للنقر وليس فقط مرئيًا
        # وأحيانًا، قد يكون هناك عدة عناصر مطابقة إذا كان البحث عامًا
        asset_to_click = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, asset_xpath))
        )
        asset_to_click.click()
        logger.info(f"Quotex: Asset '{asset_name_quotex}' selected.")
        time.sleep(0.5) # انتظر قليلاً لتحديث الواجهة بعد الاختيار
        return True
    except TimeoutException:
        logger.error(f"Quotex: Timeout selecting asset '{asset_name_quotex}'. Modal or asset not found/clickable.", exc_info=True)
        # driver.save_screenshot(f"quotex_select_asset_timeout_{asset_name_quotex.replace('/','')}.png")
    except ElementClickInterceptedException:
        logger.error(f"Quotex: Element click intercepted for asset '{asset_name_quotex}'. Another element is obscuring it.", exc_info=True)
        # driver.save_screenshot(f"quotex_select_asset_intercepted_{asset_name_quotex.replace('/','')}.png")
    except Exception as e:
        logger.error(f"Quotex: Error selecting asset '{asset_name_quotex}': {type(e).__name__} - {e}", exc_info=True)
        # driver.save_screenshot(f"quotex_select_asset_error_{asset_name_quotex.replace('/','')}.png")
    return False

def set_trade_duration(driver: webdriver.Chrome, duration_str: str, timeout: int = 10) -> bool:
    if not driver: return False
    # هذا يعتمد على أن لديك زر واضح للمدة "1m" (أو ما يعادلها "1:00")
    try:
        logger.info(f"Quotex: Setting trade duration to '{duration_str}'...")
        if duration_str == "1m": # أو أي قيمة أخرى تدعمها مباشرة بالزر
            duration_button_xpath = DURATION_1M_BUTTON_XPATH
            # قد تحتاج أولاً إلى النقر لفتح قائمة اختيار الوقت إذا لم تكن الأزرار ظاهرة مباشرة
            # WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.time-button"))).click() # مثال
            # time.sleep(0.5)

            WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, duration_button_xpath))
            ).click()
            logger.info(f"Quotex: Duration '{duration_str}' set.")
            time.sleep(0.3)
            return True
        else:
            logger.warning(f"Quotex: Duration '{duration_str}' not explicitly supported by this function's example. Please adapt.")
            return False # أو حاول إدخالها بطريقة أخرى إذا كانت الواجهة تسمح
            
    except TimeoutException:
        logger.error(f"Quotex: Timeout setting duration '{duration_str}'. Button not found or clickable.", exc_info=True)
    except Exception as e:
        logger.error(f"Quotex: Error setting duration '{duration_str}': {e}", exc_info=True)
    return False

def set_trade_amount(driver: webdriver.Chrome, amount: int, timeout: int = 10) -> bool:
    if not driver: return False
    try:
        logger.info(f"Quotex: Setting trade amount to {amount}...")
        amount_field = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, AMOUNT_INPUT_CSS))
        )
        # طريقة أكثر موثوقية لتنظيف الحقل وإدخال القيمة
        amount_field.click() # انقر أولاً لتنشيط الحقل
        # مسح الحقل باستخدام JavaScript إذا فشلت الطرق العادية
        driver.execute_script("arguments[0].value = '';", amount_field) 
        amount_field.send_keys(str(amount))
        # تأكد من أن القيمة تم تعيينها بشكل صحيح (اختياري ولكن جيد)
        # time.sleep(0.1) # اسمح للواجهة بالتحديث
        # if amount_field.get_attribute("value") != str(amount):
        #    logger.warning(f"Quotex: Amount field value mismatch. Expected {amount}, got {amount_field.get_attribute('value')}. Retrying with JS.")
        #    driver.execute_script(f"arguments[0].value = '{str(amount)}'; arguments[0].dispatchEvent(new Event('input'));", amount_field)

        logger.info(f"Quotex: Amount {amount} set.")
        time.sleep(0.2)
        return True
    except TimeoutException:
        logger.error(f"Quotex: Timeout setting amount {amount}. Amount field not found.", exc_info=True)
    except Exception as e:
        logger.error(f"Quotex: Error setting amount {amount}: {e}", exc_info=True)
    return False

def place_trade(driver: webdriver.Chrome, asset: str, direction: str, amount: int, duration: str, timeout: int = 20) -> bool:
    if not driver: return False
    try:
        logger.info(f"Quotex: Attempting trade -> Asset: {asset}, Dir: {direction.upper()}, Amt: {amount}, Dur: {duration}")
        
        # 1. اختر الأصل
        if not select_asset(driver, asset, timeout): # asset هو asset_quotex_symbol
            logger.error(f"Quotex: Failed to select asset {asset}. Trade aborted.")
            return False
        
        # 2. حدد مدة الصفقة (مثال لـ 1m)
        if duration == "1m":
            if not set_trade_duration(driver, "1m", timeout):
                 logger.error(f"Quotex: Failed to set 1m duration for {asset}. Trade aborted.")
                 return False
        else:
            logger.error(f"Quotex: Duration '{duration}' not directly supported. Trade for {asset} aborted.")
            return False

        # 3. حدد مبلغ الصفقة
        if not set_trade_amount(driver, amount, timeout):
            logger.error(f"Quotex: Failed to set trade amount for {asset}. Trade aborted.")
            return False

        # 4. انقر على زر CALL أو PUT
        button_css = CALL_BUTTON_CSS if direction.lower() == "call" else PUT_BUTTON_CSS
        trade_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, button_css))
        )
        trade_button.click()
        logger.info(f"Quotex: {direction.upper()} button clicked for {asset}.")
        time.sleep(0.7) # انتظار قصير بعد النقر للسماح بمعالجة الطلب

        # يمكنك هنا محاولة التحقق من أن الصفقة ظهرت في قائمة الصفقات المفتوحة (جزء متقدم)
        # مثال: ابحث عن عنصر في واجهة المستخدم يشير إلى أن الصفقة قد تمت.
        # هذا الجزء يعتمد بشدة على واجهة Quotex.
        # WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.deal--open"))) # مثال
        logger.info(f"Quotex: Trade for {asset} - {direction.upper()} presumed placed successfully.")
        return True
        
    except TimeoutException:
        logger.error(f"Quotex: Timeout during place_trade for {asset}. A step (asset, duration, amount, or trade button) failed.", exc_info=True)
        # driver.save_screenshot(f"quotex_place_trade_timeout_{asset.replace('/','')}.png")
    except Exception as e:
        logger.error(f"Quotex: General error placing trade for {asset}: {type(e).__name__} - {e}", exc_info=True)
        # driver.save_screenshot(f"quotex_place_trade_error_{asset.replace('/','')}.png")
    return False

def close_browser(driver: webdriver.Chrome | None):
    if driver:
        try:
            logger.info("Quotex: Closing browser...")
            driver.quit()
            logger.info("Quotex: Browser closed.")
        except Exception as e:
            logger.error(f"Quotex: Error closing browser: {e}", exc_info=True)
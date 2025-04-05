import requests
import time
import ccxt
import json
from collections import deque
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import talib
import numpy as np
from datetime import datetime, timedelta

# Подключение к API Байбит
class BybitAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
        })

    def get_futures_pairs(self):
        try:
            markets = self.exchange.load_markets()
            futures_pairs = [symbol for symbol in markets if "USDT" in symbol]
            return futures_pairs
        except Exception as e:
            print(f"Ошибка при получении данных по рынкам: {e}")
            return []

    def get_ohlcv(self, symbol, timeframe='1m'):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe)
            volumes = [x[5] for x in ohlcv]  # Объемы торгов
            return ohlcv, volumes
        except Exception as e:
            print(f"Ошибка при получении данных OHLCV: {e}")
            return [], []

# Класс для обработки API новостей
class NewsAPIHandler:
    def __init__(self, api_keys):
        self.api_keys = deque(api_keys)
        self.current_key = self.api_keys[0]
        self.base_url = "https://newsapi.org/v2/everything"

    def get_news(self, query):
        today = datetime.utcnow().date()
        url = f"{self.base_url}?q={query}&from={today}&sortBy=publishedAt&apiKey={self.current_key}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Лимит превышен для API {self.current_key}. Переход к следующему ключу.")
            self.switch_api_key()
            return self.get_news(query)
        else:
            print(f"Ошибка при запросе новостей: {response.status_code}")
            return None

    def switch_api_key(self):
        self.api_keys.rotate(-1)
        self.current_key = self.api_keys[0]
        print(f"Используем следующий API-ключ: {self.current_key}")

# Функция отправки сообщений в Telegram
def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=params)
    if response.status_code == 200:
        print(f"Сообщение успешно отправлено в Telegram: {message}")
    else:
        print(f"Ошибка при отправке сообщения в Telegram: {response.status_code}")

# Анализ настроений новостей
def analyze_sentiment(news_data):
    analyzer = SentimentIntensityAnalyzer()
    positive_count = 0
    negative_count = 0

    for article in news_data["articles"]:
        sentiment = analyzer.polarity_scores(article["title"] + " " + article["description"])  # Анализируем не только заголовок, но и описание статьи
        if sentiment['compound'] >= 0.05:
            positive_count += 1
        elif sentiment['compound'] <= -0.05:
            negative_count += 1

    return positive_count, negative_count

# Основной класс бота
class TradingBot:
    def __init__(self, bybit_api, news_api_handler, telegram_token, telegram_chat_id):
        self.bybit_api = bybit_api
        self.news_api_handler = news_api_handler
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.signal_accuracy = {"total_signals": 0, "hit_tp": 0, "hit_sl": 0}
        self.analyzed_symbols = set()  # Храним монеты, для которых уже был отправлен сигнал

    def analyze_market(self, symbol, timeframe):
        if symbol in self.analyzed_symbols:
            return  # Если сигнал для этой монеты уже был отправлен, пропускаем

        print(f"Анализируем рынок для {symbol} на таймфрейме {timeframe}...")

        # Получаем данные о рынке
        ohlcv, volumes = self.bybit_api.get_ohlcv(symbol, timeframe)

        if not ohlcv:
            print(f"Ошибка при получении данных для {symbol} на {timeframe}")
            return

        # Преобразуем список закрытых цен в numpy.ndarray
        closes = np.array([x[4] for x in ohlcv])  # Закрытие цен
        high_prices = np.array([x[2] for x in ohlcv])  # Высокие цены
        low_prices = np.array([x[3] for x in ohlcv])  # Низкие цены

        # Технический анализ (используем SMA для анализа тренда)
        indicators = self.calculate_indicators(closes, high_prices, low_prices)
        trend = "Восходящий" if indicators['sma_short'][-1] > indicators['sma_long'][-1] else "Нисходящий"
        print(f"Тренд для {symbol}: {trend}")

        # Рассчитываем ATR (волатильность)
        atr = indicators['atr'][-1]

        # Точка входа ТВХ (используем цену закрытия последней свечи)
        entry_price = closes[-1]

        # Расчет TP и SL на основе ATR
        take_profit, stop_loss = self.calculate_tp_sl(entry_price, atr, trend)

        # Генерация сигнала на основе тренда
        signal_type = "Лонг" if trend == "Восходящий" else "Шорт"

        # Получаем новости по монете
        news_data = self.news_api_handler.get_news(symbol)
        positive_count, negative_count = 0, 0
        if news_data:
            # Анализируем настроения новостей
            positive_count, negative_count = analyze_sentiment(news_data)

        # Отправка сигнала и новостей в Telegram
        self.send_signal_to_telegram(symbol, signal_type, take_profit, stop_loss, entry_price, news_data)

        # Добавляем монету в список проанализированных
        self.analyzed_symbols.add(symbol)

        # Увеличиваем счетчики сигналов
        self.signal_accuracy["total_signals"] += 1
        if self.check_price_hit_tp_sl(symbol, entry_price, take_profit, stop_loss):
            self.signal_accuracy["hit_tp"] += 1
        else:
            self.signal_accuracy["hit_sl"] += 1

    def calculate_indicators(self, closes, high_prices, low_prices):
        indicators = {}
        indicators['sma_short'] = talib.SMA(closes, timeperiod=50)
        indicators['sma_long'] = talib.SMA(closes, timeperiod=200)
        indicators['ema'] = talib.EMA(closes, timeperiod=50)
        indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = talib.BBANDS(closes, timeperiod=20)
        indicators['cci'] = talib.CCI(high_prices, low_prices, closes, timeperiod=14)
        indicators['atr'] = talib.ATR(high_prices, low_prices, closes, timeperiod=14)
        indicators['rsi'] = talib.RSI(closes, timeperiod=14)  # Добавлен RSI
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)  # Добавлен MACD
        return indicators

    def calculate_tp_sl(self, entry_price, atr, trend):
        if trend == "Восходящий":
            take_profit = entry_price + atr
            stop_loss = entry_price - atr
        else:
            take_profit = entry_price - atr
            stop_loss = entry_price + atr
        return take_profit, stop_loss

    def send_signal_to_telegram(self, symbol, signal_type, take_profit, stop_loss, entry_price, news_data):
        message = f"Пара: {symbol}\n"
        message += f"Сигнал: {signal_type}\n"
        message += f"ТП: {take_profit}\n"
        message += f"СЛ: {stop_loss}\n"
        message += "\nНовости:\n"
        for article in news_data["articles"][:3]:  # Берем только первые 3 статьи
            message += f"- {article['title']} ({article.get('url', 'No URL provided')})\n"  # Добавляем ссылку на новость

        send_telegram_message(message, self.telegram_token, self.telegram_chat_id)

    def check_price_hit_tp_sl(self, symbol, entry_price, take_profit, stop_loss):
        # Проверка достижения ТП или СЛ (можно заменить на более сложную логику)
        # Здесь подразумевается вызов функции, которая проверяет исторические данные
        # или текущую цену, чтобы определить, достиг ли сигнал ТП или СЛ
        return True  # Для примера всегда возвращаем True (ТП достигнут)

    def run(self):
        timeframes = ['1m', '5m', '15m']
        while True:
            try:
                for symbol in self.bybit_api.get_futures_pairs():
                    for timeframe in timeframes:
                        self.analyze_market(symbol, timeframe)
                time.sleep(60)  # Пауза между анализами
            except Exception as e:
                print(f"Ошибка в процессе анализа: {e}")
                time.sleep(60)

# Пример использования
api_keys_news = [
    "96029ec57793487cbbb0d41c1380875f", "410b796a9ad649c6b17deb3587f3854c",
    "07a1ebac282d4f758fb0e43f669c19fb", "69073c8d21bc4e0cb5d5f34845da5dc4",
    "e28e76e5b39344f48040ea9d16b53f4e", "d28e811cfcd84a558474af41d0f53922",
    "dc1c9fa46bbc49c5ac5dfccc7afda06e", "89438c38720a48ec81955302cd74d093",
    "dd3c066114ef4742be0704d7cca73ce8", "03437182e9b94476830b1c16edc77408"
]

# Вставьте ваш Telegram Token и Chat ID
telegram_token = "7942010489:AAHRHUaBYK11YO53W_EJuFvSeNwBteOh-oc"
telegram_chat_id = "-1002626815058"

bybit_api = BybitAPI(api_key="GuGo1f9JMMOOVDDG1T", api_secret="mwNCTRHSKLWSsg1Tox2gs5jBUc3zjChFtKQP")
news_api_handler = NewsAPIHandler(api_keys=api_keys_news)
trading_bot = TradingBot(bybit_api, news_api_handler, telegram_token, telegram_chat_id)

# Запуск бота
trading_bot.run()

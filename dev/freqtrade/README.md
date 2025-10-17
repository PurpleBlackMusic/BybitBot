# FreqAI Sidecar for Bybit Spot Guardian

Эта директория содержит минимальный каркас для запуска `freqtrade` с модулем FreqAI в режиме sidecar. Боковой сервис обучает модели на журналах сделок Guardian и регулярно отправляет предсказания обратно в ядро бота через HTTP.

## 1. Поднять Freqtrade/FreqAI как сайдкар

```bash
cd dev/freqtrade
docker compose up -d
```

Компоновка `docker-compose.yml` запускает контейнер `freqtradeorg/freqtrade:stable` со включённым FreqAI. Внутрь пробрасываются:

- каталог `bybit_app/_data` — для доступа к журналам сделок и статусам;
- файл конфигурации `config/config.json`;
- пользовательская директория `user_data` с кастомным провайдером признаков.

По умолчанию частота запросов к API Guardian — раз в 5 минут, порт REST-интерфейса freqtrade — `8080`.

## 2. Экспорт признаков из Guardian

Внутри приложения появился FastAPI-сервис `bybit_app.utils.freqai.api`. Его можно запустить командой:

```bash
uvicorn bybit_app.utils.freqai.api:app --host 0.0.0.0 --port 8099
```

Основные маршруты:

- `GET /features` — JSON с признаками по каждой монете (волатильность, объём, импульс, RSI/EMA, стакан и т. д.).
- `GET /predictions` — последние прогнозы, сохранённые sidecar'ом.
- `POST /predictions` — точка, куда freqtrade отправляет свежие вероятности/EV.
- `GET /health` — диагностический статус (флаги `freqai_enabled`, состояние WS, топ-пары по EV).

Провайдер признаков лежит в `user_data/freqai/data_providers/bybitbot_features.py`: он обращается к `GET /features` и готовит dataframe для FreqAI.

## 3. Обучение и инференс

- Журналы сделок (`executions.jsonl`) и OHLCV кэш автоматически пробрасываются в контейнер.
- Конфигурация `config/config.json` включает планировщик: ежедневное обучение (`freqai.train_cron`) и предсказания каждые N минут (`freqai.predict_interval_min`).
- После инференса скрипт `bybitbot_features.py` отправляет `POST /predictions` с ключами `probability` и `ev_bps` для каждой пары. Guardian сразу подхватывает эти значения в сканере и заменяет ими внутреннюю логистическую модель.

## 4. Решение и исполнение

`scan_market_opportunities` теперь учитывает:

- `freqai_override` в `model_metrics` — источник, уверенность, горизонт, «сырые» метаданные;
- `entry["freqai"]`, `probability_source` и `ev_source` — используемые в SignalExecutor.

Как только предсказания превышают пороги `ai_buy_threshold`/`ai_sell_threshold` и `ai_min_ev_bps`, автоматизация создаёт ордера через существующий модуль Bybit v5 (без ccxt). Все ограничения по риску (кап на позицию, волатильность, стопы) остаются прежними.

## 5. Бэктест и Hyperopt

Перед раскаткой в прод рекомендуется прогнать стратегию в offline-режиме freqtrade:

```bash
# внутри контейнера freqtrade
freqtrade backtesting --config /freqtrade/config/config.json --strategy FreqaiHyperStrategy
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --config /freqtrade/config/config.json
```

В результате можно подобрать оптимальные пороги вероятности, горизонт и набор признаков, не рискуя деньгами на реальном рынке.

## Файлы

- `docker-compose.yml` — запуск sidecar.
- `config/config.json` — базовый конфиг (API ключи, расписание, частота обучения).
- `user_data/freqai/data_providers/bybitbot_features.py` — пример провайдера, который стучится к `/features` и отдаёт pandas DataFrame с признаками.

> ⚠️ Не забудьте заполнить реальные API ключи и URL вашего Guardian, а также выставить `freqai_enabled=true` в настройках бота.

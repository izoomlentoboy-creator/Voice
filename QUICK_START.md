# EchoFlow 1.0 - Быстрый старт (для приватного репозитория)

Поскольку ваш репозиторий приватный, установка выполняется в 2 простых шага:

---

## Шаг 1: Настройка SSH-ключа на сервере (ОДИН РАЗ)

Подключитесь к серверу и выполните:

```bash
# Создайте SSH-ключ
ssh-keygen -t ed25519 -C "your_email@example.com"
# Нажимайте Enter на все вопросы

# Скопируйте публичный ключ
cat ~/.ssh/id_ed25519.pub
```

**Добавьте этот ключ в GitHub:**
1. Откройте: https://github.com/izoomlentoboy-creator/Voice/settings/keys
2. Нажмите "Add deploy key"
3. Вставьте скопированный ключ
4. Дайте ему имя (например, "Production Server")
5. ✅ Поставьте галочку "Allow write access" (если нужно)
6. Нажмите "Add key"

**Проверьте подключение:**
```bash
ssh -T git@github.com
```

Должно быть: `Hi izoomlentoboy-creator! You've successfully authenticated...`

---

## Шаг 2: Установка EchoFlow (ОДНА КОМАНДА)

```bash
git clone git@github.com:izoomlentoboy-creator/Voice.git && cd Voice && chmod +x install.sh && ./install.sh
```

**Вот и всё!** Скрипт автоматически:
- Запустится в фоновом режиме (screen)
- Проверит все требования
- Установит зависимости
- Скачает датасет 17.9 ГБ
- Обучит модель
- Запустит API на порту 8000

---

## Мониторинг установки

```bash
# Подключиться к процессу установки
screen -r echoflow_install

# Или следить за логом
tail -f echoflow_install.log
```

**Отключиться от screen (не останавливая процесс):**
Нажмите `Ctrl+A`, затем `D`

---

## Проверка после установки

```bash
# Проверить статус
curl http://localhost:8000/health

# Или откройте в браузере
http://YOUR_SERVER_IP:8000
```

---

## Управление сервисом

```bash
# Статус
sudo systemctl status echoflow

# Перезапуск
sudo systemctl restart echoflow

# Логи
sudo journalctl -u echoflow -f
```

---

## Время установки

- **Быстрый интернет:** ~1-1.5 часа
- **Средний интернет:** ~2-3 часа
- **Медленный интернет:** ~4-6 часов

Основное время — загрузка датасета (17.9 ГБ) и обучение модели.

---

## Если что-то пошло не так

**Ошибка SSH:** Вернитесь к Шагу 1, проверьте `ssh -T git@github.com`

**Недостаточно места:** Нужно минимум 40 ГБ свободного места

**Процесс завис:** Проверьте логи: `tail -f echoflow_install.log`

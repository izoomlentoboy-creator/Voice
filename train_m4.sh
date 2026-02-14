#!/bin/bash
# EchoFlow 2.0 - Одна команда для обучения на MacBook Air M4
# Автоматическая настройка, установка зависимостей и обучение в screen
# Автор: EchoFlow Team
# Дата: 2026-02-14

set -e  # Остановка при критических ошибках

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для красивого вывода
print_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_error() {
    echo -e "${RED}❌ ОШИБКА: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  ВНИМАНИЕ: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Проверка macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "Этот скрипт предназначен только для macOS!"
    exit 1
fi

# Проверка Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    print_warning "Обнаружен не Apple Silicon процессор. Скрипт оптимизирован для M-серии."
fi

print_step "🚀 EchoFlow 2.0 - Автоматическое обучение на MacBook Air M4"
echo "Этот скрипт выполнит:"
echo "  1. Установку всех зависимостей (Homebrew, Python, библиотеки)"
echo "  2. Настройку окружения для Apple Silicon"
echo "  3. Создание screen-сессии для обучения"
echo "  4. Запуск обучения модели (работает в фоне)"
echo ""
echo "Вы сможете:"
echo "  - Закрыть терминал - обучение продолжится"
echo "  - Проверить прогресс: screen -r echoflow"
echo "  - Отключиться от screen: Ctrl+A затем D"
echo ""
read -p "Продолжить? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Отменено пользователем."
    exit 0
fi

# Шаг 1: Проверка и установка Homebrew
print_step "📦 Шаг 1/8: Проверка Homebrew"
if ! command -v brew &> /dev/null; then
    echo "Homebrew не найден. Устанавливаю..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Добавляем Homebrew в PATH для Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    print_success "Homebrew установлен"
else
    print_success "Homebrew уже установлен"
fi

# Шаг 2: Установка Python 3.11
print_step "🐍 Шаг 2/8: Проверка Python 3.11"
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 не найден. Устанавливаю..."
    brew install python@3.11
    print_success "Python 3.11 установлен"
else
    print_success "Python 3.11 уже установлен"
    python3.11 --version
fi

# Шаг 3: Установка screen
print_step "🖥️  Шаг 3/8: Проверка screen"
if ! command -v screen &> /dev/null; then
    echo "screen не найден. Устанавливаю..."
    brew install screen
    print_success "screen установлен"
else
    print_success "screen уже установлен"
fi

# Шаг 4: Установка дополнительных зависимостей
print_step "🔧 Шаг 4/8: Установка системных зависимостей"
echo "Устанавливаю ffmpeg, sox, libsndfile..."
brew install ffmpeg sox libsndfile portaudio 2>/dev/null || print_warning "Некоторые пакеты уже установлены"
print_success "Системные зависимости готовы"

# Шаг 5: Создание виртуального окружения
print_step "🌐 Шаг 5/8: Настройка Python окружения"
VENV_DIR="$HOME/.echoflow_venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Виртуальное окружение уже существует. Пересоздаю..."
    rm -rf "$VENV_DIR"
fi

python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
print_success "Виртуальное окружение создано: $VENV_DIR"

# Обновляем pip
pip install --upgrade pip setuptools wheel

# Шаг 6: Установка PyTorch для Apple Silicon
print_step "🔥 Шаг 6/8: Установка PyTorch для Apple Silicon (MPS)"
echo "Устанавливаю PyTorch с поддержкой Metal Performance Shaders..."
pip install torch torchvision torchaudio
print_success "PyTorch установлен"

# Шаг 7: Установка остальных зависимостей
print_step "📚 Шаг 7/8: Установка библиотек ML"
echo "Устанавливаю transformers, librosa, scikit-learn..."

# Создаём requirements.txt если его нет
cat > requirements.txt << 'EOF'
transformers>=4.35.0
librosa>=0.10.0
soundfile>=0.12.0
torch>=2.0.0
torchaudio>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=2.0.0
EOF

pip install -r requirements.txt
print_success "Все библиотеки установлены"

# Шаг 8: Создание директорий
print_step "📁 Шаг 8/8: Подготовка к обучению"
mkdir -p dataset checkpoints logs

# Проверка наличия train.py
if [ ! -f "train.py" ]; then
    print_error "Файл train.py не найден!"
    print_warning "Убедитесь, что вы находитесь в директории Voice"
    exit 1
fi

print_success "Директории созданы"

# Создаём скрипт для запуска в screen
SCREEN_SCRIPT="$HOME/.echoflow_train_script.sh"
cat > "$SCREEN_SCRIPT" << 'SCREENEOF'
#!/bin/bash
# Внутренний скрипт для запуска обучения в screen

# Активируем виртуальное окружение
source "$HOME/.echoflow_venv/bin/activate"

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Цвета
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 EchoFlow 2.0 - Обучение началось!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Дата начала: $(date)"
echo "Устройство: MacBook Air M4 (Apple Silicon)"
echo "Ускоритель: Metal Performance Shaders (MPS)"
echo ""
echo "Параметры обучения:"
echo "  - Эпохи: 50"
echo "  - Batch size: 8 (оптимизировано для M4)"
echo "  - Learning rate: 2e-5"
echo "  - Mixed precision: Да"
echo ""
echo "Ожидаемое время: 150-200 часов (без GPU)"
echo ""
echo "Управление screen-сессией:"
echo "  - Отключиться: Ctrl+A затем D"
echo "  - Вернуться: screen -r echoflow"
echo "  - Завершить: screen -X -S echoflow quit"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Запускаем обучение
python3 train.py \
    --data_dir ./dataset \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_workers 4 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --early_stopping_patience 10 \
    --device mps 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Обучение завершено!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Дата завершения: $(date)"
echo "Модель сохранена в: checkpoints/best.pt"
echo "Логи доступны в: logs/"
echo ""
echo "Нажмите любую клавишу для выхода..."
read -n 1
SCREENEOF

chmod +x "$SCREEN_SCRIPT"

# Копируем скрипт в текущую директорию
cp "$SCREEN_SCRIPT" ./run_training.sh

# Завершающая информация
echo ""
print_step "🎯 Всё готово к запуску!"
echo ""
echo "Сейчас будет создана screen-сессия 'echoflow' и начнётся обучение."
echo ""
echo -e "${YELLOW}Важные команды:${NC}"
echo "  📊 Проверить прогресс:    screen -r echoflow"
echo "  🔌 Отключиться от screen: Ctrl+A затем D"
echo "  🛑 Остановить обучение:   screen -X -S echoflow quit"
echo "  📋 Список screen-сессий:  screen -ls"
echo ""
echo -e "${GREEN}После запуска вы можете:${NC}"
echo "  ✅ Закрыть терминал - обучение продолжится"
echo "  ✅ Выключить дисплей - обучение продолжится"
echo "  ✅ Перевести Mac в сон - обучение ОСТАНОВИТСЯ (не рекомендуется)"
echo ""
echo -e "${YELLOW}Рекомендация:${NC}"
echo "  Настройте Mac: Системные настройки → Энергосбережение → Запретить автоматический сон"
echo ""
read -p "Запустить обучение сейчас? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_warning "Обучение не запущено."
    echo "Для запуска позже выполните:"
    echo "  screen -dmS echoflow bash ./run_training.sh"
    exit 0
fi

# Проверяем, не запущена ли уже сессия
if screen -list | grep -q "echoflow"; then
    print_error "Screen-сессия 'echoflow' уже существует!"
    echo "Подключитесь к ней: screen -r echoflow"
    echo "Или удалите: screen -X -S echoflow quit"
    exit 1
fi

# Запускаем обучение в screen
print_step "🚀 Запуск обучения в screen-сессии 'echoflow'"
screen -dmS echoflow bash ./run_training.sh

sleep 2

# Проверяем, что сессия создана
if screen -list | grep -q "echoflow"; then
    echo ""
    print_success "Обучение успешно запущено!"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ Screen-сессия 'echoflow' активна${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Для просмотра прогресса выполните:"
    echo -e "  ${BLUE}screen -r echoflow${NC}"
    echo ""
    echo "Обучение работает в фоне. Можете закрыть терминал."
    echo ""
else
    print_error "Не удалось создать screen-сессию!"
    echo "Попробуйте запустить вручную:"
    echo "  screen -dmS echoflow bash ./run_training.sh"
    exit 1
fi

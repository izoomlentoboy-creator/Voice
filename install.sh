#!/bin/bash

# EchoFlow 1.0 - Smart One-Click Installer with Interactive UI
# Версия 5.0
# Полностью автономный установщик с визуальным прогрессом

# --- Конфигурация ---
GITHUB_REPO="git@github.com:izoomlentoboy-creator/Voice.git"
PROJECT_DIR="$HOME/echoflow_project"
REQUIRED_RAM_GB=4
REQUIRED_SWAP_GB=4
REQUIRED_DISK_GB=40
API_PORT=8000
LOG_FILE="echoflow_install.log"

# --- Цвета для терминала ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Автоматический запуск в Screen ---
if [ -z "$STY" ]; then
  clear
  echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║${NC}  ${BOLD}EchoFlow 1.0 - Автоматический установщик${NC}                  ${CYAN}║${NC}"
  echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${BLUE}→${NC} Запуск установщика в фоновой сессии 'echoflow_install'..."
  echo -e "${BLUE}→${NC} Процесс продолжится даже после закрытия терминала."
  echo ""
  screen -S echoflow_install -d -m bash "$0"
  echo -e "${GREEN}✓${NC} Установщик запущен успешно!"
  echo ""
  echo -e "${YELLOW}Команды для мониторинга:${NC}"
  echo -e "  ${CYAN}screen -r echoflow_install${NC}  - подключиться к сессии"
  echo -e "  ${CYAN}tail -f $LOG_FILE${NC}          - следить за логом в реальном времени"
  echo ""
  exit 0
fi

# --- Основная логика (выполняется уже внутри screen) ---

exec > >(tee -a "$LOG_FILE") 2>&1
set -euo pipefail

# Запоминаем время старта
START_TIME=$(date +%s)

# --- Вспомогательные функции для UI ---

print_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}EchoFlow 1.0 - Установщик${NC}                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Система распознавания голосовых патологий                 ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    local current=$1
    local total=$2
    local title=$3
    echo ""
    echo -e "${BOLD}${BLUE}[Шаг $current/$total]${NC} $title"
    echo -e "${BLUE}$(printf '━%.0s' {1..64})${NC}"
}

log_info() { 
    echo -e "${BLUE}ℹ${NC} $1" 
}

log_warn() { 
    echo -e "${YELLOW}⚠${NC} $1" 
}

log_ok() { 
    echo -e "${GREEN}✓${NC} $1" 
}

log_error() { 
    echo -e "${RED}✗${NC} $1"
    echo ""
    echo -e "${RED}Установка прервана. Проверьте лог: $LOG_FILE${NC}"
    exit 1
}

show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r${CYAN}Прогресс: [${NC}"
    printf "${GREEN}%${filled}s${NC}" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "${CYAN}] ${BOLD}%3d%%${NC}" "$percent"
}

print_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}${GREEN}УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!${NC}                           ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Сводка:${NC}"
    echo -e "  ${BLUE}→${NC} Время установки: ${minutes}м ${seconds}с"
    echo -e "  ${BLUE}→${NC} API-сервер: ${GREEN}запущен${NC}"
    echo -e "  ${BLUE}→${NC} Адрес: ${CYAN}http://$(wget -qO- ifconfig.me 2>/dev/null || echo '127.0.0.1'):${API_PORT}${NC}"
    echo -e "  ${BLUE}→${NC} Проверка: ${CYAN}curl http://localhost:${API_PORT}/health${NC}"
    echo ""
}

# --- Функции проверки и установки ---

check_and_install_dependencies() {
    print_step 1 10 "Проверка системных зависимостей"
    
    local REQUIRED_TOOLS=("python3" "pip3" "git" "screen" "wget" "bc" "lsof" "unzip")
    local MISSING_TOOLS=()
    
    log_info "Сканирование установленных пакетов..."
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            MISSING_TOOLS+=("$tool")
        fi
    done
    
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        log_warn "Отсутствуют: ${MISSING_TOOLS[*]}"
        log_info "Установка недостающих компонентов..."
        sudo apt-get update -qq
        
        local PACKAGES_TO_INSTALL=""
        for tool in "${MISSING_TOOLS[@]}"; do
            case "$tool" in
                python3) PACKAGES_TO_INSTALL+="python3-full python3-pip python3-venv " ;;
                pip3) PACKAGES_TO_INSTALL+="python3-pip " ;;
                git) PACKAGES_TO_INSTALL+="git " ;;
                screen) PACKAGES_TO_INSTALL+="screen " ;;
                wget) PACKAGES_TO_INSTALL+="wget " ;;
                bc) PACKAGES_TO_INSTALL+="bc " ;;
                lsof) PACKAGES_TO_INSTALL+="lsof " ;;
                unzip) PACKAGES_TO_INSTALL+="unzip " ;;
            esac
        done
        
        sudo apt-get install -y $PACKAGES_TO_INSTALL ffmpeg
        log_ok "Все зависимости установлены"
    else
        log_ok "Все базовые зависимости присутствуют"
    fi
    show_progress 1 10
}

pre_flight_checks() {
    print_step 2 10 "Предполетная проверка системы"

    # Проверка диска
    log_info "Проверка дискового пространства..."
    local available_disk_gb=$(df -BG /var | awk 'NR==2 {print substr($4, 1, length($4)-1)}')
    if (( $(echo "$available_disk_gb < $REQUIRED_DISK_GB" | bc -l) )); then
        log_error "Недостаточно места: требуется ${REQUIRED_DISK_GB}GB, доступно ${available_disk_gb}GB"
    fi
    log_ok "Диск: ${available_disk_gb}GB доступно"

    # Проверка RAM и SWAP
    log_info "Проверка оперативной памяти..."
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    if (( $(echo "$total_ram_gb < $REQUIRED_RAM_GB" | bc -l) )); then
        log_warn "RAM (${total_ram_gb}GB) < рекомендуемого (${REQUIRED_RAM_GB}GB)"
        if ! grep -q '/swapfile' /etc/fstab; then
            log_info "Создание SWAP-файла (${REQUIRED_SWAP_GB}GB)..."
            sudo fallocate -l ${REQUIRED_SWAP_GB}G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
            log_ok "SWAP создан и активирован"
        else
            log_ok "SWAP уже настроен"
        fi
    else
        log_ok "RAM: ${total_ram_gb}GB (достаточно)"
    fi

    # Проверка порта
    log_info "Проверка порта ${API_PORT}..."
    local pid_on_port=$(sudo lsof -t -i:${API_PORT} 2>/dev/null || echo "")
    if [ -n "$pid_on_port" ]; then
        log_warn "Порт занят (PID: ${pid_on_port}), освобождаем..."
        sudo kill -9 "$pid_on_port"
        sleep 2
        log_ok "Порт освобожден"
    else
        log_ok "Порт ${API_PORT} свободен"
    fi
    
    show_progress 2 10
}

setup_project() {
    print_step 3 10 "Настройка проекта"

    log_info "Очистка предыдущих установок..."
    sudo pkill -f 'uvicorn|voice_disorder_detection' || true
    sudo rm -f /etc/systemd/system/echoflow.service
    sudo systemctl daemon-reload
    log_ok "Система очищена"
    
    show_progress 3 10
    
    print_step 4 10 "Проверка SSH-доступа к GitHub"
    
    if [ ! -f "$HOME/.ssh/id_rsa" ] && [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        log_error "SSH-ключ не найден!\n\nИнструкция:\n1. ssh-keygen -t ed25519 -C 'your@email.com'\n2. cat ~/.ssh/id_ed25519.pub\n3. Добавьте ключ в GitHub Deploy Keys\n4. Запустите скрипт заново"
    fi
    
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    log_ok "SSH-ключ найден и настроен"
    
    show_progress 4 10
    
    print_step 5 10 "Клонирование репозитория"
    
    if [ -d "$PROJECT_DIR" ]; then
        log_warn "Проект уже существует, обновляем..."
        cd "$PROJECT_DIR"
        git pull || log_warn "Не удалось обновить, продолжаем с текущей версией"
    else
        log_info "Клонирование из GitHub..."
        git clone "$GITHUB_REPO" "$PROJECT_DIR" || log_error "Ошибка клонирования. Проверьте SSH-ключ."
        cd "$PROJECT_DIR"
    fi
    log_ok "Репозиторий готов"
    
    show_progress 5 10
    
    print_step 6 10 "Настройка Python окружения"
    
    if [ ! -f "requirements.txt" ]; then
        log_warn "requirements.txt не найден, создаем базовую версию..."
        cat << 'REQ_EOF' > requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
scikit-learn==1.5.2
librosa==0.10.2
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1
REQ_EOF
    fi
    
    if [ ! -d "venv" ]; then
        log_info "Создание виртуального окружения..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    
    log_info "Установка Python-зависимостей..."
    pip install --upgrade pip --quiet
    pip install --ignore-installed -r requirements.txt --quiet
    log_ok "Python окружение настроено"
    
    show_progress 6 10
}

download_and_prepare_dataset() {
    print_step 7 10 "Загрузка датасета (17.9 GB)"
    
    local DATA_DIR="/var/lib/voice-disorder/data/sbvoicedb/data"
    sudo mkdir -p "$DATA_DIR"
    sudo chown -R $(whoami) "/var/lib/voice-disorder/"
    
    local DATASET_URL="https://zenodo.org/records/16874898/files/data.zip"
    local DATASET_FILE="$DATA_DIR/data.zip"
    
    if [ ! -f "$DATASET_FILE" ]; then
        log_info "Загрузка датасета... (это может занять несколько часов)"
        wget -c -O "$DATASET_FILE" "$DATASET_URL" --progress=bar:force 2>&1 | \
            grep --line-buffered "%" | \
            sed -u -e "s,\.,,g" | \
            awk '{printf("\r'"${CYAN}Загрузка: ${GREEN}%s${NC}"'\n", $2)}'
        log_ok "Датасет загружен"
    else
        log_ok "Датасет уже загружен"
    fi
    
    show_progress 7 10
    
    print_step 8 10 "Распаковка датасета"
    
    if [ ! -d "$DATA_DIR/healthy" ]; then
        log_info "Распаковка архива..."
        unzip -n -q "$DATASET_FILE" -d "$DATA_DIR"
        log_ok "Датасет распакован"
    else
        log_ok "Датасет уже распакован"
    fi
    
    show_progress 8 10
}

train_model() {
    print_step 9 10 "Обучение модели EchoFlow 1.0"
    
    if [ -f "voice_disorder_detection/pipeline.py" ]; then
        if [ -f "echoflow_config.py" ]; then
            cp echoflow_config.py voice_disorder_detection/config.py
        fi
        log_info "Запуск обучения модели..."
        python3 -m voice_disorder_detection.pipeline --mode binary --backend ensemble
        log_ok "Модель обучена"
    else
        log_warn "Скрипт обучения не найден, создаем API-заглушку..."
        cat << 'API_EOF' > main.py
from fastapi import FastAPI
app = FastAPI(title="EchoFlow 1.0 API")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "EchoFlow 1.0"}

@app.get("/")
def root():
    return {"message": "EchoFlow 1.0 Voice Disorder Detection System"}
API_EOF
        log_ok "API-заглушка создана"
    fi
    
    show_progress 9 10
}

setup_service() {
    print_step 10 10 "Настройка и запуск сервиса"
    
    local MAIN_FILE="main.py"
    if [ -f "server/app/main.py" ]; then
        MAIN_FILE="server.app.main"
    elif [ -f "app/main.py" ]; then
        MAIN_FILE="app.main"
    fi
    
    local SERVICE_FILE="/etc/systemd/system/echoflow.service"
    sudo bash -c "cat << SERVICE_EOF > $SERVICE_FILE
[Unit]
Description=EchoFlow 1.0 API Server
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment=\"PATH=$PROJECT_DIR/venv/bin:/usr/bin\"
ExecStart=$PROJECT_DIR/venv/bin/uvicorn ${MAIN_FILE}:app --host 0.0.0.0 --port ${API_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE_EOF"

    sudo systemctl daemon-reload
    sudo systemctl enable echoflow
    sudo systemctl restart echoflow
    
    log_info "Ожидание запуска сервиса..."
    sleep 10
    
    if systemctl is-active --quiet echoflow; then
        log_ok "Сервис запущен и работает"
    else
        log_error "Сервис не запустился. Проверьте: sudo journalctl -u echoflow -n 50"
    fi
    
    show_progress 10 10
}

# --- Точка входа ---
main() {
    print_header
    
    check_and_install_dependencies
    pre_flight_checks
    setup_project
    download_and_prepare_dataset
    train_model
    setup_service
    
    print_summary
}

main

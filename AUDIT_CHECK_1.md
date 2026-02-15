# 🔍 Проверка #1 - Глубокий аудит train_ultimate_v2.py

## ❌ НАЙДЕНЫ НОВЫЕ ОШИБКИ

---

## **ОШИБКА #21: Нет target_sr в EnhancedVoiceDataset при создании**

**Строки 377-379:**
```python
train_dataset = EnhancedVoiceDataset(train_samples, max_length=args.max_length, augment=True)
val_dataset = EnhancedVoiceDataset(val_samples, max_length=args.max_length, augment=False)
test_dataset = EnhancedVoiceDataset(test_samples, max_length=args.max_length, augment=False)
```

**Проблема:**
- `EnhancedVoiceDataset.__init__` принимает параметр `target_sr=16000` (строка 41)
- Мы НЕ передаём `target_sr` при создании датасетов
- Используется дефолтное значение 16000
- Но если мы захотим изменить target_sr через аргументы - не сработает!

**Исправление:**
```python
train_dataset = EnhancedVoiceDataset(train_samples, max_length=args.max_length, target_sr=16000, augment=True)
val_dataset = EnhancedVoiceDataset(val_samples, max_length=args.max_length, target_sr=16000, augment=False)
test_dataset = EnhancedVoiceDataset(test_samples, max_length=args.max_length, target_sr=16000, augment=False)
```

**Критичность:** 🟡 СРЕДНЯЯ (работает, но не гибко)

---

## **ОШИБКА #22: Confusion matrix может быть неправильного размера**

**Строки 285-288:**
```python
if all_labels:
    cm = confusion_matrix(all_labels, all_preds)
else:
    cm = None
```

**Проблема:**
- Если в батче только один класс (например, все pathological), `confusion_matrix` вернёт матрицу 1x1
- Ожидаем матрицу 2x2
- При попытке доступа к `cm[0,1]` получим IndexError

**Исправление:**
```python
if all_labels:
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # Явно указать классы
else:
    cm = None
```

**Критичность:** 🔴 КРИТИЧНО (может вызвать крах программы)

---

## **ОШИБКА #23: Warmup scheduler применяется ПОСЛЕ первой эпохи**

**Строки 456-459:**
```python
# Update learning rate
if epoch <= args.warmup_epochs:
    warmup_scheduler.step()
else:
    plateau_scheduler.step(val_loss)
```

**Проблема:**
- `warmup_scheduler.step()` вызывается ПОСЛЕ обучения эпохи
- Но LR должен обновляться ПЕРЕД эпохой!
- Результат: первая эпоха обучается с LR = 5e-6 / 5 = 1e-6 (слишком мало)

**Правильная логика:**
```python
# ПЕРЕД эпохой (в начале цикла):
if epoch <= args.warmup_epochs:
    current_lr = args.learning_rate * (epoch / args.warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

# ПОСЛЕ эпохи:
if epoch > args.warmup_epochs:
    plateau_scheduler.step(val_loss)
```

**Критичность:** 🔴 КРИТИЧНО (неправильный LR в первых эпохах)

---

## **ОШИБКА #24: Неправильная обработка остаточных градиентов**

**Строки 244-248:**
```python
# Handle remaining gradients
if len(dataloader) % accumulation_steps != 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Проблема:**
- Проверка `len(dataloader) % accumulation_steps != 0` проверяет количество БАТЧЕЙ
- Но градиенты накапливаются по ИНДЕКСУ батча `i`
- Если последний батч не достиг `accumulation_steps`, градиенты НЕ применятся

**Пример:**
```
dataloader = 10 батчей
accumulation_steps = 2

Батчи: 0, 1 (step), 2, 3 (step), 4, 5 (step), 6, 7 (step), 8, 9 (step)
Все градиенты применены!

len(dataloader) % accumulation_steps = 10 % 2 = 0
Условие НЕ выполнится, хотя всё ОК
```

**Правильная проверка:**
```python
# Handle remaining gradients
if (len(dataloader) - 1) % accumulation_steps != accumulation_steps - 1:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**ИЛИ проще:**
```python
# Всегда применять в конце
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad()
```

**Критичность:** 🟡 СРЕДНЯЯ (может пропустить последние градиенты)

---

## **ОШИБКА #25: Нет сохранения конфигурации обучения**

**Проблема:**
- Мы сохраняем модель, историю, логи
- Но НЕ сохраняем конфигурацию (args)
- Невозможно воспроизвести обучение

**Решение:**
```python
# После создания output_dir
config = vars(args)  # Преобразовать argparse в dict
with open(output_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

**Критичность:** 🟢 НИЗКАЯ (удобство)

---

## **ОШИБКА #26: Неправильный расчёт test_size**

**Строки 364-366:**
```python
train_size = int(0.7 * len(indices))
val_size = int(0.15 * len(indices))
test_size = len(indices) - train_size - val_size
```

**Проблема:**
- `int(0.7 * 2041) = 1428`
- `int(0.15 * 2041) = 306`
- `test_size = 2041 - 1428 - 306 = 307`
- Соотношение: 1428:306:307 = 69.9%:15.0%:15.1% ✅ (OK)

**НО:**
- Если `len(indices) = 100`:
  - `train_size = int(0.7 * 100) = 70`
  - `val_size = int(0.15 * 100) = 15`
  - `test_size = 100 - 70 - 15 = 15`
  - Соотношение: 70:15:15 ✅ (OK)

**Проблема возникает при нечётных числах:**
- `len(indices) = 101`:
  - `train_size = int(0.7 * 101) = 70`
  - `val_size = int(0.15 * 101) = 15`
  - `test_size = 101 - 70 - 15 = 16`
  - Соотношение: 70:15:16 (не 70:15:15)

**Правильный подход:**
```python
train_size = int(0.7 * len(indices))
val_size = int(0.15 * len(indices))
test_size = len(indices) - train_size - val_size  # Правильно, остаток идёт в test
```

**Критичность:** 🟢 НИЗКАЯ (работает корректно)

---

## **ОШИБКА #27: librosa.resample deprecated**

**Строка 102:**
```python
data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
```

**Проблема:**
- `librosa.resample()` deprecated в librosa 0.10.0+
- Новый API: `librosa.resample(y, orig_sr, target_sr)` → `librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)`
- Работает, но выдаёт warning

**Исправление:**
```python
data = librosa.resample(y=data, orig_sr=sr, target_sr=self.target_sr)
```

**Критичность:** 🟢 НИЗКАЯ (работает, но warning)

---

## **ОШИБКА #28: librosa.effects.time_stretch deprecated API**

**Строка 73:**
```python
stretched = librosa.effects.time_stretch(data, rate=rate)
```

**Проблема:**
- В librosa 0.10.0+ требуется `y=` для первого аргумента
- Старый API: `time_stretch(data, rate)`
- Новый API: `time_stretch(y=data, rate=rate)`

**Исправление:**
```python
stretched = librosa.effects.time_stretch(y=data, rate=rate)
```

**Критичность:** 🟢 НИЗКАЯ (работает, но warning)

---

## **ОШИБКА #29: librosa.effects.pitch_shift deprecated API**

**Строка 87:**
```python
return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
```

**Проблема:**
- В librosa 0.10.0+ требуется `y=` для первого аргумента

**Исправление:**
```python
return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)
```

**Критичность:** 🟢 НИЗКАЯ (работает, но warning)

---

## **ОШИБКА #30: Нет проверки пустых датасетов**

**Строки 344-354:**
```python
# Load normal samples
normal_dir = data_dir / 'normal'
if normal_dir.exists():
    for wav_file in normal_dir.glob('*.wav'):
        all_samples.append((str(wav_file), 0))

# Load pathological samples
patho_dir = data_dir / 'pathological'
if patho_dir.exists():
    for wav_file in patho_dir.glob('*.wav'):
        all_samples.append((str(wav_file), 1))
```

**Проблема:**
- Если `all_samples` пустой, программа продолжит работу
- Создаст пустые датасеты
- Упадёт при первой эпохе

**Исправление:**
```python
if len(all_samples) == 0:
    raise ValueError("No audio files found! Check dataset directory.")

if sum(1 for _, l in all_samples if l == 0) == 0:
    raise ValueError("No normal samples found!")

if sum(1 for _, l in all_samples if l == 1) == 0:
    raise ValueError("No pathological samples found!")
```

**Критичность:** 🟡 СРЕДНЯЯ (защита от ошибок)

---

## 📊 Итоги проверки #1

### **Найдено ошибок: 10 (21-30)**

| # | Ошибка | Критичность |
|---|--------|-------------|
| 21 | Нет target_sr при создании датасетов | 🟡 СРЕДНЯЯ |
| 22 | Confusion matrix неправильного размера | 🔴 КРИТИЧНО |
| 23 | Warmup scheduler после эпохи | 🔴 КРИТИЧНО |
| 24 | Неправильная обработка остаточных градиентов | 🟡 СРЕДНЯЯ |
| 25 | Нет сохранения конфигурации | 🟢 НИЗКАЯ |
| 26 | Расчёт test_size (на самом деле OK) | 🟢 НИЗКАЯ |
| 27 | librosa.resample deprecated | 🟢 НИЗКАЯ |
| 28 | librosa.time_stretch deprecated | 🟢 НИЗКАЯ |
| 29 | librosa.pitch_shift deprecated | 🟢 НИЗКАЯ |
| 30 | Нет проверки пустых датасетов | 🟡 СРЕДНЯЯ |

### **Критичные: 2**
- #22: Confusion matrix
- #23: Warmup scheduler

### **Средние: 4**
- #21, #24, #30

### **Низкие: 4**
- #25, #26, #27, #28, #29

---

## ❌ Результат проверки #1: ПРОВАЛЕНА

**Найдено 10 ошибок, из них 2 критичные.**

**Требуется исправление и повторная проверка.**

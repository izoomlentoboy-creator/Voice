# EchoFlow 1.0 - API Documentation для iOS

**Версия API:** 1.0  
**Base URL:** `http://YOUR_SERVER_IP:8000/api/v1`  
**Дата:** 13 февраля 2026

Эта документация описывает REST API для интеграции EchoFlow 1.0 с iOS-приложением.

---

## 📋 Общая информация

### Формат данных
- **Request:** `multipart/form-data` (для загрузки аудио-файлов)
- **Response:** `application/json`

### Аутентификация
В текущей версии аутентификация не требуется. Для production рекомендуется добавить API keys.

### Rate Limiting
- **Лимит:** 60 запросов в минуту на один IP-адрес
- **HTTP Status при превышении:** `429 Too Many Requests`

---

## 🎤 Endpoints

### 1. Анализ голоса

**Основной endpoint** для отправки аудио-записей и получения результатов анализа.

#### Request

```http
POST /api/v1/analyze
Content-Type: multipart/form-data
```

**Parameters:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `audio_a` | File (WAV) | ✅ Да | Запись гласной "А" (5-7 сек) |
| `audio_i` | File (WAV) | ✅ Да | Запись гласной "И" (5-7 сек) |
| `audio_u` | File (WAV) | ✅ Да | Запись гласной "У" (5-7 сек) |
| `user_id` | String | ❌ Нет | UUID устройства для истории |
| `gender` | String | ❌ Нет | Пол: `"m"` или `"w"` |
| `age` | Integer | ❌ Нет | Возраст пользователя |
| `app_version` | String | ❌ Нет | Версия приложения (для аналитики) |
| `device_model` | String | ❌ Нет | Модель устройства (для аналитики) |

**Требования к аудио-файлам:**
- Формат: WAV, 16-bit PCM
- Частота дискретизации: 16000 Hz (рекомендуется)
- Каналы: Моно (1 канал)
- Длительность: 3-10 секунд
- Максимальный размер файла: 10 MB

#### Response (Success)

**HTTP Status:** `200 OK`

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "verdict": "healthy",
    "verdict_label": "Голос в норме",
    "confidence": 0.89,
    "confidence_percent": 89,
    "abstain": false
  },
  "details": {
    "pitch_stability": {
      "status": "good",
      "label": "Стабильность высоты: хорошо",
      "score": 0.92
    },
    "harmonic_quality": {
      "status": "good",
      "label": "Гармоническое качество: хорошо",
      "score": 0.88
    },
    "voice_steadiness": {
      "status": "good",
      "label": "Ровность голоса: хорошо",
      "score": 0.85
    },
    "spectral_clarity": {
      "status": "good",
      "label": "Спектральная чистота: хорошо",
      "score": 0.91
    },
    "breath_support": {
      "status": "good",
      "label": "Поддержка дыхания: хорошо",
      "score": 0.87
    }
  },
  "recommendation": "Ваш голос в норме. Признаков голосовых расстройств не обнаружено.",
  "ood_warning": false,
  "processing_time_ms": 234,
  "timestamp": "2026-02-13T15:30:45.123Z"
}
```

**Response Fields:**

| Поле | Тип | Описание |
|------|-----|----------|
| `analysis_id` | String (UUID) | Уникальный ID анализа |
| `result.verdict` | String | Вердикт: `"healthy"`, `"pathological"`, `"abstain"` |
| `result.verdict_label` | String | Читаемая метка вердикта |
| `result.confidence` | Float | Уверенность модели (0.0 - 1.0) |
| `result.confidence_percent` | Integer | Уверенность в процентах (0 - 100) |
| `result.abstain` | Boolean | `true` если модель воздержалась от вердикта |
| `details` | Object | Детальные показатели по категориям |
| `details.*.status` | String | Статус: `"good"`, `"moderate"`, `"poor"` |
| `details.*.label` | String | Читаемая метка категории |
| `details.*.score` | Float | Оценка категории (0.0 - 1.0) |
| `recommendation` | String | Текстовая рекомендация для пользователя |
| `ood_warning` | Boolean | Предупреждение о данных вне распределения |
| `processing_time_ms` | Integer | Время обработки в миллисекундах |
| `timestamp` | String (ISO 8601) | Время анализа |

#### Response (Error)

**HTTP Status:** `400 Bad Request`

```json
{
  "detail": "Missing required audio file: audio_a"
}
```

**HTTP Status:** `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "audio_a"],
      "msg": "File too large (max 10MB)",
      "type": "value_error"
    }
  ]
}
```

**HTTP Status:** `500 Internal Server Error`

```json
{
  "status": "error",
  "message": "Внутренняя ошибка сервера. Попробуйте позже."
}
```

---

### 2. История анализов

Получить историю анализов для конкретного пользователя.

#### Request

```http
GET /api/v1/history?user_id={user_id}&limit={limit}
```

**Query Parameters:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `user_id` | String | ✅ Да | UUID устройства |
| `limit` | Integer | ❌ Нет | Количество записей (по умолчанию: 10, макс: 100) |

#### Response (Success)

**HTTP Status:** `200 OK`

```json
{
  "user_id": "ios-abc123def456",
  "total": 5,
  "analyses": [
    {
      "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2026-02-13T15:30:45.123Z",
      "verdict": "healthy",
      "verdict_label": "Голос в норме",
      "confidence_percent": 89
    },
    {
      "analysis_id": "660e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2026-02-10T10:15:30.456Z",
      "verdict": "pathological",
      "verdict_label": "Возможны нарушения",
      "confidence_percent": 76
    }
  ]
}
```

---

### 3. Детали анализа

Получить полные детали конкретного анализа по ID.

#### Request

```http
GET /api/v1/analysis/{analysis_id}
```

**Path Parameters:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `analysis_id` | String (UUID) | ID анализа |

#### Response (Success)

**HTTP Status:** `200 OK`

Возвращает тот же объект, что и `/analyze` endpoint.

#### Response (Error)

**HTTP Status:** `404 Not Found`

```json
{
  "detail": "Analysis not found"
}
```

---

### 4. Health Check

Проверка статуса API.

#### Request

```http
GET /api/v1/health
```

#### Response (Success)

**HTTP Status:** `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0"
}
```

---

## 📱 Пример интеграции (Swift)

### Отправка аудио на анализ

```swift
import Foundation

struct VoiceAnalysisRequest {
    let audioA: URL  // Local file URL
    let audioI: URL
    let audioU: URL
    let userId: String
    let deviceModel: String
    let appVersion: String
}

struct VoiceAnalysisResponse: Codable {
    let analysisId: String
    let result: AnalysisResult
    let details: [String: CategoryDetail]
    let recommendation: String
    let processingTimeMs: Int
    let timestamp: String
    
    enum CodingKeys: String, CodingKey {
        case analysisId = "analysis_id"
        case result, details, recommendation
        case processingTimeMs = "processing_time_ms"
        case timestamp
    }
}

struct AnalysisResult: Codable {
    let verdict: String
    let verdictLabel: String
    let confidence: Double
    let confidencePercent: Int
    let abstain: Bool
    
    enum CodingKeys: String, CodingKey {
        case verdict
        case verdictLabel = "verdict_label"
        case confidence
        case confidencePercent = "confidence_percent"
        case abstain
    }
}

struct CategoryDetail: Codable {
    let status: String
    let label: String
    let score: Double
}

class EchoFlowAPI {
    let baseURL = "http://YOUR_SERVER_IP:8000/api/v1"
    
    func analyzeVoice(request: VoiceAnalysisRequest, completion: @escaping (Result<VoiceAnalysisResponse, Error>) -> Void) {
        let url = URL(string: "\(baseURL)/analyze")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add audio files
        if let audioAData = try? Data(contentsOf: request.audioA) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"audio_a\"; filename=\"a.wav\"\r\n")
            body.append("Content-Type: audio/wav\r\n\r\n")
            body.append(audioAData)
            body.append("\r\n")
        }
        
        if let audioIData = try? Data(contentsOf: request.audioI) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"audio_i\"; filename=\"i.wav\"\r\n")
            body.append("Content-Type: audio/wav\r\n\r\n")
            body.append(audioIData)
            body.append("\r\n")
        }
        
        if let audioUData = try? Data(contentsOf: request.audioU) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"audio_u\"; filename=\"u.wav\"\r\n")
            body.append("Content-Type: audio/wav\r\n\r\n")
            body.append(audioUData)
            body.append("\r\n")
        }
        
        // Add metadata
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"user_id\"\r\n\r\n")
        body.append("\(request.userId)\r\n")
        
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"device_model\"\r\n\r\n")
        body.append("\(request.deviceModel)\r\n")
        
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"app_version\"\r\n\r\n")
        body.append("\(request.appVersion)\r\n")
        
        body.append("--\(boundary)--\r\n")
        
        urlRequest.httpBody = body
        
        let task = URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "EchoFlow", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                let decoder = JSONDecoder()
                let result = try decoder.decode(VoiceAnalysisResponse.self, from: data)
                completion(.success(result))
            } catch {
                completion(.failure(error))
            }
        }
        
        task.resume()
    }
}

// Helper extension for Data
extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
```

### Использование

```swift
let api = EchoFlowAPI()

let request = VoiceAnalysisRequest(
    audioA: URL(fileURLWithPath: "/path/to/a.wav"),
    audioI: URL(fileURLWithPath: "/path/to/i.wav"),
    audioU: URL(fileURLWithPath: "/path/to/u.wav"),
    userId: UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
    deviceModel: UIDevice.current.model,
    appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
)

api.analyzeVoice(request: request) { result in
    switch result {
    case .success(let response):
        print("Analysis ID: \(response.analysisId)")
        print("Verdict: \(response.result.verdictLabel)")
        print("Confidence: \(response.result.confidencePercent)%")
        print("Recommendation: \(response.recommendation)")
        
        // Update UI with results
        DispatchQueue.main.async {
            // Update your UI here
        }
        
    case .failure(let error):
        print("Error: \(error.localizedDescription)")
        
        // Show error to user
        DispatchQueue.main.async {
            // Show error alert
        }
    }
}
```

---

## 🔒 Рекомендации по безопасности

### Для Production

1. **HTTPS:** Используйте SSL/TLS сертификаты (Let's Encrypt)
2. **API Keys:** Добавьте аутентификацию через API keys
3. **Rate Limiting:** Настройте более строгие лимиты
4. **Input Validation:** Проверяйте размер и формат файлов на клиенте
5. **Error Handling:** Не показывайте пользователям технические детали ошибок

### Пример с API Key (будущая версия)

```swift
urlRequest.setValue("Bearer YOUR_API_KEY", forHTTPHeaderField: "Authorization")
```

---

## 📊 Коды ответов

| HTTP Status | Описание |
|-------------|----------|
| `200 OK` | Успешный запрос |
| `400 Bad Request` | Неверные параметры запроса |
| `404 Not Found` | Ресурс не найден |
| `422 Unprocessable Entity` | Ошибка валидации данных |
| `429 Too Many Requests` | Превышен лимит запросов |
| `500 Internal Server Error` | Внутренняя ошибка сервера |

---

## 🧪 Тестирование API

### С помощью curl

```bash
# Health check
curl http://YOUR_SERVER_IP:8000/api/v1/health

# Analyze voice
curl -X POST http://YOUR_SERVER_IP:8000/api/v1/analyze \
  -F "audio_a=@/path/to/a.wav" \
  -F "audio_i=@/path/to/i.wav" \
  -F "audio_u=@/path/to/u.wav" \
  -F "user_id=test-user-123" \
  -F "app_version=1.0-test"

# Get history
curl "http://YOUR_SERVER_IP:8000/api/v1/history?user_id=test-user-123&limit=5"
```

### Интерактивная документация

Откройте в браузере: `http://YOUR_SERVER_IP:8000/docs`

Swagger UI позволяет тестировать все endpoints прямо в браузере.

---

**Версия документа:** 1.0  
**Последнее обновление:** 13 февраля 2026  
**Поддержка:** Для вопросов по API обращайтесь к разработчикам

# TinyMLS

TinyMLS is not yet Machine Learning-based Spellchecker

## Chạy server

Yêu cầu cài dependency trong `requirements.txt`.

```bash
# Dùng thư mục trained_model/ (mặc định)
python main.py

# Dùng file .tinymls (tự động, không cần config riêng)
python main.py model.tinymls
```

Biến môi trường hỗ trợ:

- `HOST`: mặc định `0.0.0.0`
- `PORT`: mặc định `8000`
- `DEBUG`: nhận `1`, `true`, `yes` để bật debug

Ví dụ:

```bash
HOST=127.0.0.1 PORT=8000 python main.py model.tinymls
```

Chế độ chạy:

- `DEBUG=true`: dùng Flask dev server
- mặc định: ưu tiên `waitress` để chạy ổn định lâu dài trên máy

## Cấu trúc thư mục

```
TinyMLS/
├── .github/workflows/  # CI pipeline (GitHub Actions)
│   └── test.yml
├── tests/              # Black-box API test suite (pytest)
│   ├── conftest.py     # Fixtures: mini model build từ embedded corpus
│   └── test_api.py     # 32 tests qua Flask test client
├── data/               # Dữ liệu (corpus + dictionary)
│   ├── corpus/         # Dữ liệu văn bản thô (.txt) để train
│   └── wordlist.dic    # Từ điển tiếng Việt chuẩn
├── builder.py          # Xây dựng N-gram language model từ corpus
├── trained_model/      # Model artifacts (unigrams.trie, bigrams.trie, ...)
├── model_pkg.py        # Export/Import model thành 1 file .tinymls duy nhất
├── checker.py          # Inference engine (NGramSpellChecker)
├── config.py           # SpellCheckerConfig dataclass
├── config.json         # Runtime configuration
├── keyboard.py         # QWERTY keyboard layout
├── telex.py            # Telex encoding conversion
├── api.py              # Flask REST API
└── main.py             # Entry point
```

## Kiểm thử

```bash
# Cài pytest
pip install pytest

# Chạy toàn bộ test suite (32 tests)
pytest tests/ -v

# Chạy test riêng theo category
pytest tests/ -k "Category1" -v    # Close match
pytest tests/ -k "Category2" -v    # Telex
pytest tests/ -k "Category4" -v    # QWERTY
pytest tests/ -k "Personalization" -v
pytest tests/ -k "ErrorHandling" -v
```

Model test được build từ embedded corpus (11 câu, 33 từ vựng) mỗi lần chạy, không ảnh hưởng đến model chính. Personalization data tự động dọn sau suite.

## API

Server đọc cấu hình từ `config.json`. Frontend không truyền path hay file cấu hình lên API.

Base URL mặc định:

```text
http://localhost:8000
```

### `GET /api/health`

Kiểm tra server đang hoạt động.

Ví dụ:

```bash
curl http://localhost:8000/api/health
```

Response:

```json
{
  "status": "ok",
  "checker_loaded": true,
  "build_in_progress": false,
  "active_requests": 0,
  "last_load_error": null
}
```

### `POST /api/check`

Nhận văn bản đầu vào và trả về các gợi ý sửa lỗi.

Request body:

- `text`: chuỗi cần kiểm tra, bắt buộc
- `top_k`: số gợi ý cần trả về, mặc định `5`

Các path cấu hình (`stats_path`, `dict_path`) được lấy từ `config.json`/`config.py` phía server, frontend không truyền lên.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "toi dang go tieng viet",
    "top_k": 3
  }'
```

Response thành công:

```json
{
  "text": "toi dang go tieng viet",
  "top_k": 3,
  "best_correction": "toi đang gõ tiếng việt",
  "processing_ms": 12.4,
  "suggestions": [
    "toi đang gõ tiếng việt",
    "tôi đang gõ tiếng việt",
    "toi đang go tiếng việt"
  ]
}
```

Response lỗi:

```json
{
  "error": "Thiếu trường 'text'."
}
```

### `POST /api/build`

Build lại bộ thống kê ngôn ngữ từ corpus.

Request body:

- `workers`: số worker xử lý, mặc định `1`
- `recursive`: `true` nếu muốn đọc đệ quy file `.txt` từ thư mục con, mặc định `false`

`data/corpus/` là thư mục input cố định phía server. Các đường dẫn lấy từ `config.json` hoặc dùng mặc định trong `config.py`.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/build \
  -H "Content-Type: application/json" \
  -d '{
    "workers": 4,
    "recursive": true
  }'
```

Response thành công:

```json
{
  "message": "Xây dựng thống kê hoàn tất.",
  "logs": "..."
}
```

## Tích hợp frontend

Frontend chỉ cần gọi HTTP JSON:

```js
const response = await fetch("http://localhost:8000/api/check", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: userInput,
    top_k: 5,
  }),
});

const data = await response.json();
```

Server đã bật CORS `*`, nên có thể gọi trực tiếp từ frontend chạy domain khác trong môi trường phát triển.

### `POST /api/export`

Export model thành file `.tinymls` duy nhất. Các thành phần trong file không cố định — chỉ gồm những gì tồn tại trong thư mục stats (vd: bỏ qua `trigrams.trie` nếu không có).

Request body:

- `output`: đường dẫn file đầu ra, mặc định `model.tinymls`

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{"output": "my_model.tinymls"}'
```

## Model packages (`.tinymls`)

File `.tinymls` là zip chứa toàn bộ model thành 1 file duy nhất.

### Sử dụng

```python
from config import SpellCheckerConfig
from checker import NGramSpellChecker

# Load trực tiếp từ .tinymls (không giải nén thủ công)
cfg = SpellCheckerConfig(stats_path='model.tinymls')
checker = NGramSpellChecker(cfg)
checker.correct_sentence('toi dang go tieng viet')
checker.close()  # dọn temp files
```

### `dict_path`

- Mặc định `None` — không tự động tìm file từ điển
- Nếu load từ `.tinymls` có chứa `dictionary.dic`, checker tự động dùng
- Nếu chỉ định `dict_path` rõ ràng → ưu tiên dùng file đó

```python
# Dùng dictionary.dic trong package (nếu có)
cfg = SpellCheckerConfig(stats_path='model.tinymls')

# Ghi đè: dùng file riêng
cfg = SpellCheckerConfig(stats_path='model.tinymls', dict_path='data/wordlist.dic')
```

## Export model (CLI)

```bash
# Export to file
python main.py --export model.tinymls

# Hoặc dùng công cụ riêng
python model_pkg.py export --stats trained_model --output model.tinymls
python model_pkg.py export --stats trained_model --dict data/wordlist.dic -o model.tinymls
python model_pkg.py extract model.tinymls -o my_model
```

## Ghi chú vận hành

- Checker được preload khi server khởi động để giảm độ trễ ở request đầu tiên.
- Trong lúc build thống kê, request check mới sẽ chờ build hoàn tất rồi mới xử lý.
- API giới hạn input `text` tối đa 2000 ký tự cho mỗi request.
- Builder mặc định chỉ đọc file `.txt` ở thư mục cấp 1. Dùng `recursive=true` để đọc đệ quy vào thư mục con.
- Builder chấp nhận cả từ tiếng Việt và tiếng Anh (từ chỉ gồm chữ cái) vào vocabulary, giúp model không sửa nhầm từ ngoại lai.

## Cá nhân hóa (Personalization)

Hệ thống hỗ trợ hai cơ chế cá nhân hóa, hoạt động độc lập với model static N-gram:

### 1. Học từ văn bản (Learn from text)

Người dùng gửi một đoạn văn bản tự do (bài báo, tài liệu chuyên ngành, ...). Hệ thống tự động trích xuất:

- **Từ mới**: mỗi từ đều được thêm vào danh sách ưu tiên, được cộng `priority_score` (mặc định 5.0) khi xuất hiện trong candidate
- **Ngữ cảnh N-gram**: mọi bigram và trigram liền kề đều được ghi nhận, khi context khớp lại cũng được cộng `priority_score`

Không phụ thuộc tần suất — từ chỉ xuất hiện 1 lần vẫn có boost tương đương từ xuất hiện nhiều lần.

Ví dụ: paste câu `"Bệnh nhân được chỉ định uống thuốc kháng sinh"` → hệ thống học:
- Priority: `bệnh`, `nhân`, `chỉ`, `định`, `uống`, `thuốc`, `kháng`, `sinh`, ...
- Contexts: `bệnh nhân`, `nhân chỉ`, `chỉ định`, `định uống`, `uống thuốc`, `thuốc kháng`, `kháng sinh`, ...
- Bigram contexts: `bệnh nhân chỉ`, `nhân chỉ định`, `chỉ định uống`, `định uống thuốc`, `uống thuốc kháng`, `thuốc kháng sinh`, ...

Khi scoring:
- Từ `thuốc` (trong văn bản) → +`priority_score`
- Context `uống thuốc` khớp → +`priority_score` nữa (flat, không frequency)

### 2. Bộ nhớ thói quen (Personalization memory)

Lưu lịch sử lựa chọn candidate của người dùng theo context N-gram. Khi người dùng chọn một từ gợi ý (qua `/api/learn`), hệ thống ghi nhận cặp `(context, word)` với mọi cấp độ ngữ cảnh.

Ví dụ context `["đã", "uống", "thuốc"]` tạo 2 entry trong memory:
- `uống thuốc` (unigram context)
- `đã uống thuốc` (bigram context)

Khi scoring, mỗi cấp context khớp đều được cộng `boost_factor * count` (mặc định 2.0 * số lần, có trọng số theo tần suất).

Khác với học từ văn bản (flat boost), bộ nhớ thói quen tích luỹ dần theo số lần người dùng chọn candidate.

### Lưu trữ

Dữ liệu được lưu trong `data/personalization/{hash}/` — hash SHA-256 (32 ký tự hex) từ passphrase người dùng.

```
data/personalization/
└── {sha256_hash}/
    ├── memory.json        # Bộ nhớ thói quen (context word → count, có trọng số)
    ├── learned.json       # Từ và context học từ văn bản (flat, không trọng số)
    └── ...
```
    └── dict/
        └── {filename}.txt # Từ điển ưu tiên (mỗi dòng một từ)
```

### Cấu hình (`config.json` / `config.py`)

| Tham số                     | Mặc định | Mô tả                                                |
| --------------------------- | -------- | ---------------------------------------------------- |
| `max_personal_memory_size`    | `10000`    | Số lượng cặp `(context, word)` tối đa (LRU eviction) |
| `priority_score`              | `5.0`      | Điểm cộng cho từ trong từ điển ưu tiên               |
| `boost_factor`                | `2.0`      | Hệ số nhân cho số lần chọn trong bộ nhớ              |
| `personalization_dir`         | `"data/personalization"` | Thư mục gốc chứa dữ liệu cá nhân hóa    |

### API

Tất cả endpoint personalization đều yêu cầu `passphrase` — do người dùng tự đặt, dùng để xác định danh tính và thư mục lưu trữ.

#### `POST /api/check` (mở rộng)

Thêm field `passphrase` để kích hoạt cá nhân hóa.

Request body:

- `text` (bắt buộc): chuỗi cần kiểm tra
- `top_k`: số gợi ý, mặc định `5`
- `passphrase` (tùy chọn): mật mã người dùng

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "uong thuooc",
    "passphrase": "mysecret"
  }'
```

Response bổ sung:

```json
{
  "text": "uong thuooc",
  "personalized": true,
  "best_correction": "uong thuốc",
  "suggestions": ["uong thuốc", "ung thư", "uong mau"],
  "processing_ms": 15.3
}
```

#### `POST /api/learn`

Học từ lựa chọn của người dùng. Context là luồng Viterbi đã đi qua (các từ đã được sửa), với từ cuối cùng là từ gợi ý được chọn.

Request body:

- `passphrase` (bắt buộc): mật mã người dùng
- `context` (bắt buộc): mảng các từ, tối thiểu 2 phần tử

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/learn \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "mysecret",
    "context": ["uong", "thuốc"]
  }'
```

Response:

```json
{ "status": "ok" }
```

Học với context dài hơn (N-gram):

```bash
curl -X POST http://localhost:8000/api/learn \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "mysecret",
    "context": ["đã", "uống", "thuốc"]
  }'
```

#### `POST /api/learn/text`

Gửi văn bản tự do để hệ thống học từ mới và ngữ cảnh (không phụ thuộc tần suất — từ xuất hiện 1 lần cũng được boost như nhau).

Request body:

- `passphrase` (bắt buộc)
- `text` (bắt buộc): văn bản cần học

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/learn/text \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "mysecret",
    "text": "Bệnh nhân được chỉ định uống thuốc kháng sinh và tái khám sau một tuần"
  }'
```

Response:

```json
{
  "status": "ok",
  "words_added": 15,
  "contexts_added": 27,
  "total_words": 15
}
```

#### `GET /api/profile`

Xem thông tin dữ liệu cá nhân hóa hiện tại.

Query parameter:

- `passphrase` (bắt buộc)

Ví dụ:

```bash
curl "http://localhost:8000/api/profile?passphrase=mysecret"
```

Response:

```json
{
  "user_hash": "652c7dc687d98c9889304ed2e408c74b",
  "learned_words": 15,
  "learned_contexts": 27,
  "memory_size": 3
}
```

#### `DELETE /api/personalization`

Xoá toàn bộ dữ liệu cá nhân hóa (cả từ/text học được lẫn bộ nhớ thói quen).

Request body:

- `passphrase` (bắt buộc)

Ví dụ:

```bash
curl -X DELETE http://localhost:8000/api/personalization \
  -H "Content-Type: application/json" \
  -d '{"passphrase": "mysecret"}'
```

Response:

```json
{ "status": "ok" }
```

### Luồng tích hợp frontend

1. **Học từ văn bản**: Khi người dùng nhập một đoạn văn bản chuyên ngành, gửi đến `/api/learn/text` để hệ thống học từ mới + context
2. Kiểm tra chính tả với `passphrase` qua `/api/check`
3. Khi người dùng chọn một gợi ý, gửi context (luồng Viterbi của gợi ý đó) đến `/api/learn`

```js
// Bước 0: Học từ văn bản (chỉ cần làm 1 lần cho mỗi domain)
await fetch("http://localhost:8000/api/learn/text", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    passphrase: userPassphrase,
    text: userDocument,
  }),
});

// Bước 1: Kiểm tra chính tả
const check = await fetch("http://localhost:8000/api/check", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: userInput,
    passphrase: userPassphrase,
    top_k: 5,
  }),
});
const { suggestions, best_correction } = await check.json();

// Bước 2: Người dùng chọn một gợi ý → học thói quen
const viterbiPath = selectedSuggestion.split(" ");
await fetch("http://localhost:8000/api/learn", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    passphrase: userPassphrase,
    context: viterbiPath,
  }),
});
```

### Học từ văn bản vs Bộ nhớ thói quen

| Đặc tính          | Học từ văn bản (`/api/learn/text`) | Bộ nhớ thói quen (`/api/learn`) |
| ----------------- | ----------------------------------- | ------------------------------- |
| Kích hoạt         | Paste văn bản                       | Chọn candidate từ gợi ý         |
| Tác dụng          | Thêm từ + context vào danh sách ưu tiên | Tăng dần điểm theo số lần chọn  |
| Trọng số tần suất | Không (flat)                        | Có (tích luỹ dần)               |
| Điểm cộng         | `priority_score` (5.0) cho từ + mỗi context khớp | `boost_factor * count` (2.0/lần) |
| Use case          | Học domain mới (y, luật, kỹ thuật)  | Học thói quen sửa lỗi cá nhân   |

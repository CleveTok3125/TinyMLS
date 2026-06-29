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

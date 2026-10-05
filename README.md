# TinyMLS

TinyMLS is not yet Machine Learning-based Spellchecker

## Chạy server

Yêu cầu cài dependency trong `requirements.txt`.

```bash
# Dùng thư mục trained_model/ (mặc định)
python main.py

# Dùng file .tinymls
python main.py model.tinymls

# Chỉ định socket khác
python main.py --socket /tmp/tinymls.sock
```

Socket đặt ở `$XDG_RUNTIME_DIR/tinymls.sock` — đúng theo XDG Base Directory
Specification, nơi dành cho socket và file chỉ sống trong phiên đăng nhập. Trên
hệ thống dùng systemd, `XDG_RUNTIME_DIR` là `/run/user/$UID`.

Nếu biến này không được đặt thì server báo lỗi rõ ràng và **không tự chọn chỗ
thay thế**, vì mọi vị trí khác đều không phải chuẩn. Đặt biến, hoặc truyền
`--socket` / `TINYMLS_SOCKET`:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
python main.py
```

Server nạp model trước khi lắng nghe, nên lần request đầu không phải chờ nạp.
Socket được đặt quyền `0600`: chỉ cùng user mới kết nối được.

Dừng server bằng `SIGINT` hoặc `SIGTERM`; socket file sẽ được xoá khi thoát.

## Giao thức

Mỗi frame gồm **4 byte độ dài big-endian**, theo sau là **msgpack map**.
Một request, một response, giữ nguyên thứ tự trên một connection đã mở, nên model
được giữ nóng suốt phiên làm việc.

Ký tự báo độ dài thay vì ký tự xuống dòng, vì một bản sửa có thể chứa bất kỳ ký tự
nào và payload vốn đã là nhị phân.

Request:

```json
{"op": "check", "text": "toi dang go tieng viet", "top_k": 5, "personalized": false}
```

Response:

```json
{"ok": true, "op": "check", "text": "toi dang go tieng viet", "top_k": 5,
 "best_correction": "toi đang gõ tiếng việt",
 "suggestions": ["toi đang gõ tiếng việt"],
 "personalized": false, "processing_ms": 9.42}
```

Khi lỗi: `{"ok": false, "error": "..."}`.

Frame dài hơn 1 MiB bị từ chối. Lỗi ở tầng framing đóng connection vì không còn
đồng bộ được luồng; lỗi của một request cụ thể chỉ trả về `ok: false` và giữ
connection.

### Operations

| `op`         | Trường             | Trả về                                                       |
| ------------ | ------------------ | ------------------------------------------------------------ |
| `ping`       | —                  | `status`                                                     |
| `check`      | `text`             | `best_correction`, `suggestions`, `personalized`, `processing_ms` |
| `learn`      | `context`          | `status`                                                     |
| `learn_text` | `text`             | `words_added`, `contexts_added`, `total_words`               |
| `profile`    | —                  | `learned_words`, `learned_contexts`, `memory_size`              |
| `clear`      | —                  | `status`                                                     |

`check` nhận `personalized: true` để dùng từ và thói quen đã học. Mặc định là
không.

`text` tối đa 2000 ký tự. `context` cần ít nhất 2 phần tử. `top_k` mặc định 5.

## Client CLI

```bash
python cli.py ping
python cli.py check "toi dang go tieng viet"
python cli.py check "uong thuoc" --personalized --top-k 3
python cli.py learn-text "$(cat tai-lieu.txt)"
python cli.py learn "đã uống thuốc"
python cli.py profile
python cli.py clear
```

Thêm `--json` để lấy phản hồi nguyên bản từ server. Cờ `--socket` nhận được ở
trước hoặc sau subcommand.

```bash
python cli.py check "toi dang go"
python cli.py check "toi dang go" --json
```

Mã thoát: `0` thành công, `1` server báo lỗi, `2` không kết nối được socket.

## Client thư viện

```python
from client import SpellCheckerClient
from paths import socket_path

with SpellCheckerClient(socket_path()) as conn:
    result = conn.check("toi dang go tieng viet", top_k=3)
    print(result["best_correction"], result["suggestions"])

    conn.learn_text("Bệnh nhân được uống thuốc kháng sinh")
    print(conn.profile())
```

`client.py` là bản tham chiếu. Client viết bằng ngôn ngữ khác làm theo
`protocol.py`: 4 byte độ dài big-endian rồi tới msgpack map.

### Về tương tranh

Mỗi connection một thread, nhưng các lời gọi checker được tuần tự hoá. Việc sửa
lỗi là tác vụ CPU-bound và bị GIL giới hạn, nên chạy song song chỉ thêm nhiễu
scheduling mà không tăng throughput.

## Cấu hình

`config.json` được đọc từ `$XDG_CONFIG_HOME/tinymls/config.json` nếu có, nếu không
thì từ `config.json` trong thư mục đang chạy. File trong XDG được ưu tiên, nên
cấu hình ở mức người dùng sẽ ghi đè bản đi kèm dự án.

## Build và export

```bash
# Xây dựng lại thống kê N-gram từ corpus
python main.py --build
python main.py --build --corpus /path/to/corpus --workers 4 --recursive

# Export model thành một file
python main.py --export model.tinymls

# Công cụ riêng
python model_pkg.py export --stats trained_model --dict data/wordlist.dic -o model.tinymls
python model_pkg.py extract model.tinymls -o my_model
```

Sau `--build` cần khởi động lại server để nạp model mới. Build chạy ở tiến trình
riêng nên không phối hợp với request đang chờ.

## Cấu trúc thư mục

```
TinyMLS/
├── .github/workflows/  # CI pipeline (GitHub Actions)
│   └── test.yml
├── tests/              # Test suite (pytest)
│   ├── conftest.py     # Fixtures: mini model + server trên socket tạm
│   ├── test_protocol.py# Framing, round trip, lỗi, thao tác
│   └── test_cli.py     # Parsing, output, mã thoát
├── data/               # Dữ liệu (corpus + dictionary)
│   ├── corpus/         # Dữ liệu văn bản thô (.txt) để train
│   └── wordlist.dic    # Từ điển tiếng Việt chuẩn
├── protocol.py         # Định nghĩa wire format
├── server.py           # Unix socket server
├── service.py          # Nghiệp vụ + cache
├── client.py           # Client tham chiếu
├── cli.py              # Client dòng lệnh
├── builder.py          # Xây dựng N-gram language model từ corpus
├── trained_model/      # Model artifacts
│   ├── unigrams.trie, bigrams.trie, trigrams.trie
│   ├── vocab.txt       # Từ điển, 1 từ/dòng, đã sắp xếp
│   └── language_stats_meta.json
├── model_pkg.py        # Export/Import model thành 1 file .tinymls duy nhất
├── checker.py          # Inference engine (NGramSpellChecker)
├── config.py           # SpellCheckerConfig dataclass
├── config.json         # Runtime configuration
├── keyboard.py         # QWERTY keyboard layout
├── personalization.py  # Bộ nhớ học được, lưu theo XDG
├── vietnamese.py       # Chuẩn hoá văn bản tiếng Việt
├── paths.py            # Đường dẫn theo XDG Base Directory
├── telex.py            # Telex encoding conversion
└── main.py             # Entry point
```

## Kiểm thử

```bash
pip install pytest

pytest tests/ -v
```

Bộ test dựng một model nhỏ từ corpus 11 câu nhúng sẵn, chạy server thật trên
socket tạm rồi gọi qua client. Dữ liệu personalization tự dọn sau suite.

```bash
pytest tests/ -k protocol -v   # framing và round trip
pytest tests/ -k cli -v         # dòng lệnh
```

## Model packages (`.tinymls`)

File `.tinymls` là zip chứa toàn bộ model thành một file.

```python
from config import SpellCheckerConfig
from checker import NGramSpellChecker

cfg = SpellCheckerConfig(stats_path="model.tinymls")
checker = NGramSpellChecker(cfg)
checker.correct_sentence("toi dang go tieng viet")
checker.close()  # dọn temp files
```

`dict_path`:

- Mặc định `None` — không tự động tìm file từ điển
- Nếu load từ `.tinymls` có chứa `dictionary.dic`, checker tự động dùng
- Nếu chỉ định `dict_path` rõ ràng → ưu tiên dùng file đó

```python
cfg = SpellCheckerConfig(stats_path="model.tinymls", dict_path="data/wordlist.dic")
```

## Ghi chú vận hành

- Model được nạp trước khi server lắng nghe.
- Server chỉ phục vụ tiến trình cùng user trên cùng máy.
- `personalized` là tham số của từng lời gọi `correct_sentence`, không phải trạng thái của checker. Cùng một checker phục vụ cả lời gọi có và không có cá nhân hóa, nên `personalized: true` không phát sinh lần nạp model thứ hai.
- Builder mặc định chỉ đọc file `.txt` ở thư mục cấp 1. Dùng `--recursive` để đọc đệ quy vào thư mục con.
- Builder chấp nhận cả từ tiếng Việt và tiếng Anh (từ chỉ gồm chữ cái) vào vocabulary, giúp model không sửa nhầm từ ngoại lai.
- Từ điển nằm ở `vocab.txt` (mỗi dòng một từ, đã sắp xếp), tách riêng khỏi `language_stats_meta.json` vì parse nhanh hơn 2.1× và nhỏ hơn 1.5 MB. Thứ tự từ trong file quyết định thứ tự ứng viên nên phải giữ nguyên khi sửa.
- Không có `vocab.txt` thì checker báo lỗi rõ ràng; cần build lại model.
- Checker giữ cache cho từ đứng trước (`_CONTEXT_INDEX_BUDGET` giới hạn theo tổng số từ kế tiếp được cache). Tăng ngân sách thì nhanh hơn nhưng tốn bộ nhớ; đo được 250.000 là mức không còn thrashing, 150.000 bắt đầu chậm lại rõ rệt.
- Bộ lọc độ dài ứng viên dùng `bytes.translate` trên một byte cho mỗi từ kế tiếp.

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

Lưu lịch sử lựa chọn candidate của người dùng theo context N-gram. Khi người dùng chọn một từ gợi ý (qua operation `learn`), hệ thống ghi nhận cặp `(context, word)` với mọi cấp độ ngữ cảnh.

Ví dụ context `["đã", "uống", "thuốc"]` tạo 2 entry trong memory:

- `uống thuốc` (unigram context)
- `đã uống thuốc` (bigram context)

Khi scoring, mỗi cấp context khớp đều được cộng `boost_factor * count` (mặc định 2.0 * số lần, có trọng số theo tần suất).

Khác với học từ văn bản (flat boost), bộ nhớ thói quen tích luỹ dần theo số lần người dùng chọn candidate.

### Lưu trữ

Dữ liệu nằm trong thư mục dữ liệu XDG của user.

```
$XDG_DATA_HOME/tinymls/personalization/     # mặc định ~/.local/share/tinymls/personalization/
├── memory.json        # Bộ nhớ thói quen (context word → count, có trọng số)
└── learned.json       # Từ và context học từ văn bản (flat, không trọng số)
```

### Cấu hình (`config.json` / `config.py`)

| Tham số                     | Mặc định | Mô tả                                                |
| --------------------------- | -------- | ---------------------------------------------------- |
| `max_personal_memory_size`    | `10000`    | Số lượng cặp `(context, word)` tối đa (LRU eviction) |
| `priority_score`              | `5.0`      | Điểm cộng cho từ trong từ điển ưu tiên               |
| `boost_factor`                | `2.0`      | Hệ số nhân cho số lần chọn trong bộ nhớ              |

### Operations

#### `check` (mở rộng)

Thêm field `personalized` để kích hoạt cá nhân hóa.

```python
conn.check("uong thuooc", personalized=True)
```

```json
{"ok": true, "op": "check", "text": "uong thuooc", "top_k": 5,
 "best_correction": "uong thuốc",
 "suggestions": ["uong thuốc"], "personalized": true, "processing_ms": 12.4}
```

#### `learn_text`

Học từ văn bản tự do: thêm từ mới và context. Không phụ thuộc tần suất — từ chỉ xuất
hiện 1 lần vẫn được boost như nhau.

```python
conn.learn_text("Bệnh nhân được chỉ định uống thuốc kháng sinh")
```

```json
{"ok": true, "op": "learn_text", "status": "ok",
 "words_added": 15, "contexts_added": 27, "total_words": 15}
```

#### `learn`

Ghi nhận lựa chọn của người dùng. Context là chuỗi từ đã sửa, từ cuối là gợi ý được
chọn; mọi cấp độ ngữ cảnh đều được ghi lại. Cần ít nhất 2 phần tử.

```python
conn.learn(["đã", "uống", "thuốc"])
```

```json
{"ok": true, "op": "learn", "status": "ok"}
```

#### `profile`

```python
conn.profile()
```

```json
{"ok": true, "op": "profile",
 "learned_words": 15, "learned_contexts": 27, "memory_size": 3}
```

`profile` không trả đường dẫn lưu trữ. Muốn biết vị trí thì suy ra từ quy ước XDG
nêu ở mục "Lưu trữ".

#### `clear`

Xoá toàn bộ dữ liệu cá nhân hóa: cả từ/context học được lẫn bộ nhớ thói quen.

```python
conn.clear()
```

```json
{"ok": true, "op": "clear", "status": "ok"}
```

### Luồng tích hợp

1. **Học từ văn bản**: khi người dùng có đoạn văn bản chuyên ngành, gửi `learn_text`
   một lần cho mỗi domain
2. Kiểm tra chính tả với `check`
3. Khi người dùng chọn một gợi ý, gửi `learn` với chuỗi từ của gợi ý đó

```python
from client import SpellCheckerClient
from paths import socket_path

with SpellCheckerClient(socket_path()) as conn:
    # Bước 0: học từ văn bản, chỉ cần làm 1 lần cho mỗi domain
    conn.learn_text(user_document)

    # Bước 1: kiểm tra chính tả
    result = conn.check(user_input, top_k=5)
    suggestions = result["suggestions"]

    # Bước 2: người dùng chọn một gợi ý để học thói quen
    conn.learn(selected_suggestion.split(" "))
```

Connection giữ mở xuyên suốt phiên, nên ba bước trên dùng chung một socket.

### Học từ văn bản vs Bộ nhớ thói quen

| Đặc tính          | Học từ văn bản (operation `learn_text`) | Bộ nhớ thói quen (operation `learn`) |
| ----------------- | ----------------------------------- | ------------------------------- |
| Kích hoạt         | Paste văn bản                       | Chọn candidate từ gợi ý         |
| Tác dụng          | Thêm từ + context vào danh sách ưu tiên | Tăng dần điểm theo số lần chọn  |
| Trọng số tần suất | Không (flat)                        | Có (tích luỹ dần)               |
| Điểm cộng         | `priority_score` (5.0) cho từ + mỗi context khớp | `boost_factor * count` (2.0/lần) |
| Use case          | Học domain mới (y, luật, kỹ thuật)  | Học thói quen sửa lỗi cá nhân   |

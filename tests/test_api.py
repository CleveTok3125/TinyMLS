class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["checker_loaded"] is True


class TestCategory1CloseMatch:
    """Lỗi chính tả thông thường — sai 1-3 ký tự, không phải phím kề."""

    def test_thuoc(self, client):
        """thuoc → thuốc (thiếu 's' trong Telex 'thuocs')."""
        resp = client.post("/api/check", json={"text": "thuoc"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "thuốc" in data["best_correction"]

    def test_uongs(self, client):
        """uongs → uống (thiếu 'o' trong Telex 'uoongs')."""
        resp = client.post("/api/check", json={"text": "uongs"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "uống" in data["best_correction"]

    def test_benh(self, client):
        """benh → bệnh (thiếu 'j' — tone suffix)."""
        resp = client.post("/api/check", json={"text": "benh"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "bệnh" in data["best_correction"]

    def test_nhan(self, client):
        """nhan → nhân (thiếu 'a' trong Telex 'nhaan')."""
        resp = client.post("/api/check", json={"text": "nhan"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "nhân" in data["best_correction"]


class TestCategory2Telex:
    """Lỗi Telex — raw ASCII codes trùng Telex form của từ đúng."""

    def test_ngayf(self, client):
        """ngayf → ngày (Telex('ngayf') == Telex('ngày') == 'ngayf')."""
        resp = client.post("/api/check", json={"text": "ngayf"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "ngày" in data["best_correction"]

    def test_thuocs(self, client):
        """thuocs → thuốc (Telex('thuocs') == Telex('thuốc'))."""
        resp = client.post("/api/check", json={"text": "thuocs"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "thuốc" in data["best_correction"]

    def test_nguwoif(self, client):
        """nguwoif → người (Telex('nguwoif') == Telex('người'))."""
        resp = client.post("/api/check", json={"text": "nguwoif"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "người" in data["best_correction"]


class TestCategory3Context:
    """Lỗi ngữ cảnh — N-gram context giúp chọn đúng candidate."""

    def test_uong_after_nhan(self, client):
        """'bệnh nhân uongws' → '...uống...' (bigram 'nhân uống' có sẵn)."""
        resp = client.post("/api/check", json={
            "text": "bệnh nhân uongws",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert "uống" in data["best_correction"]

    def test_thuoc_after_uong(self, client):
        """'uongws thuốc' → 'uống thuốc' (bigram 'uống thuốc' có sẵn)."""
        resp = client.post("/api/check", json={
            "text": "uongws thuốc",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert "uống" in data["best_correction"]


class TestCategory4Qwerty:
    """Lỗi bàn phím QWERTY — phím bấm nhầm kề nhau."""

    def test_thuocd(self, client):
        """thuocd → thuốc (d kề s: a s d f)."""
        resp = client.post("/api/check", json={"text": "thuocd"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "thuốc" in data["best_correction"]

    def test_ngayd(self, client):
        """ngayd → ngày (d kề f: a s d f)."""
        resp = client.post("/api/check", json={"text": "ngayd"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert "ngày" in data["best_correction"]


class TestCategory5Combined:
    """Lỗi kết hợp — nhiều loại lỗi trong cùng câu."""

    def test_multiple_typos(self, client):
        """benh nahan uoongd thuosc → bệnh nhân uống thuốc."""
        resp = client.post("/api/check", json={
            "text": "benh nahan uoongd thuosc",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        result = data["best_correction"]
        assert "bệnh" in result
        assert "nhân" in result
        assert "uống" in result
        assert "thuốc" in result


class TestCheckOptions:
    """Kiểm tra các tùy chọn của /api/check."""

    def test_top_k(self, client):
        resp = client.post("/api/check", json={
            "text": "toi la",
            "top_k": 3,
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert len(data["suggestions"]) == 3
        assert data["top_k"] == 3

    def test_already_correct(self, client):
        """Văn bản đã đúng → giữ nguyên."""
        resp = client.post("/api/check", json={
            "text": "ngày mai",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert "ngày" in data["best_correction"]
        assert "mai" in data["best_correction"]

    def test_not_personalized_by_default(self, client):
        """Không truyền passphrase → personalied = False."""
        resp = client.post("/api/check", json={"text": "ngayf"})
        data = resp.get_json()
        assert data["personalized"] is False

    def test_personalized_with_passphrase(self, client):
        """Có passphrase → personalied = True."""
        resp = client.post("/api/check", json={
            "text": "ngayf",
            "passphrase": "check-opt",
        })
        data = resp.get_json()
        assert data["personalized"] is True


class TestCheckErrors:
    """Lỗi đầu vào của /api/check."""

    def test_missing_text(self, client):
        resp = client.post("/api/check", json={})
        assert resp.status_code == 400

    def test_empty_text(self, client):
        resp = client.post("/api/check", json={"text": ""})
        assert resp.status_code == 400


class TestPersonalization:
    """Luồng personalization — mỗi test dùng passphrase riêng."""

    def test_learn_text(self, client):
        """Học từ văn bản → words_added > 0."""
        resp = client.post("/api/learn/text", json={
            "passphrase": "pt-learn-1",
            "text": "bệnh nhân uống thuốc mỗi ngày",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["words_added"] > 0
        assert data["contexts_added"] > 0

    def test_learn_text_idempotent(self, client):
        """Học cùng văn bản 2 lần → lần 2 không thêm gì."""
        passphrase = "pt-idem"
        client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "bệnh nhân uống thuốc",
        })
        resp = client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "bệnh nhân uống thuốc",
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["words_added"] == 0
        assert data["contexts_added"] == 0

    def test_profile_after_learn(self, client):
        """Sau khi học → profile có learned_words > 0."""
        passphrase = "pt-profile-1"
        client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "học viết tiếng việt",
        })
        resp = client.get("/api/profile", query_string={
            "passphrase": passphrase,
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["learned_words"] > 0
        assert data["learned_contexts"] > 0
        assert data["memory_size"] == 0

    def test_check_with_personalization(self, client):
        """Check với passphrase → personalied = True, correction đúng."""
        passphrase = "pt-check-1"
        client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "bệnh nhân uống thuốc",
        })
        resp = client.post("/api/check", json={
            "text": "benh nhan uongws thuosc",
            "passphrase": passphrase,
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["personalized"] is True
        assert "bệnh" in data["best_correction"]
        assert "nhân" in data["best_correction"]
        assert "uống" in data["best_correction"]
        assert "thuốc" in data["best_correction"]

    def test_learn_selection(self, client):
        """Học context selection → memory_size > 0."""
        passphrase = "pt-sel-1"
        resp = client.post("/api/learn", json={
            "passphrase": passphrase,
            "context": ["đi", "chơi", "về"],
        })
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"
        profile = client.get("/api/profile", query_string={
            "passphrase": passphrase,
        }).get_json()
        assert profile["memory_size"] > 0

    def test_check_after_learn_selection(self, client):
        """Selection làm tăng boost cho context đã biết."""
        passphrase = "pt-sel-2"
        client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "đi chơi về",
        })
        client.post("/api/learn", json={
            "passphrase": passphrase,
            "context": ["đi", "chơi", "về"],
        })
        resp = client.post("/api/check", json={
            "text": "đi choi về",
            "passphrase": passphrase,
        })
        assert resp.status_code == 200
        assert resp.get_json()["personalized"] is True

    def test_clear_personalization(self, client):
        """Xóa → profile reset, cache cleared."""
        passphrase = "pt-clear-1"
        client.post("/api/learn/text", json={
            "passphrase": passphrase,
            "text": "bệnh nhân uống thuốc",
        })
        resp = client.delete("/api/personalization", json={
            "passphrase": passphrase,
        })
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"
        profile = client.get("/api/profile", query_string={
            "passphrase": passphrase,
        }).get_json()
        assert profile["learned_words"] == 0
        assert profile["learned_contexts"] == 0
        assert profile["memory_size"] == 0

    def test_two_users_independent(self, client):
        """Hai user khác nhau có profile riêng biệt."""
        client.post("/api/learn/text", json={
            "passphrase": "user-a",
            "text": "bệnh nhân uống thuốc",
        })
        client.post("/api/learn/text", json={
            "passphrase": "user-b",
            "text": "học viết tiếng việt",
        })
        resp_a = client.get("/api/profile", query_string={"passphrase": "user-a"})
        resp_b = client.get("/api/profile", query_string={"passphrase": "user-b"})
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        data_a, data_b = resp_a.get_json(), resp_b.get_json()
        assert data_a["learned_words"] == 4
        assert data_b["learned_words"] == 4
        assert data_a["user_hash"] != data_b["user_hash"]


class TestErrorHandling:
    """Lỗi đầu vào các endpoint personalization."""

    def test_learn_text_missing_passphrase(self, client):
        resp = client.post("/api/learn/text", json={"text": "abc"})
        assert resp.status_code == 400

    def test_learn_text_missing_text(self, client):
        resp = client.post("/api/learn/text", json={"passphrase": "x"})
        assert resp.status_code == 400

    def test_learn_selection_short_context(self, client):
        resp = client.post("/api/learn", json={
            "passphrase": "x",
            "context": ["a"],
        })
        assert resp.status_code == 400

    def test_profile_missing_passphrase(self, client):
        resp = client.get("/api/profile")
        assert resp.status_code == 400

    def test_clear_missing_passphrase(self, client):
        resp = client.delete("/api/personalization", json={})
        assert resp.status_code == 400

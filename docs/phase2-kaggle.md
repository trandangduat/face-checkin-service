# Phase 2 · Nghiệm thu trên Kaggle CPU

**Kết luận tổng hợp:** Bản tích hợp **B (`3b8f8df33c27e7d21d5dcb56b91b77c261902b76`) đạt toàn bộ tiêu chí nghiệm thu đã chốt trước pilot trên Kaggle CPU**, không có lỗi oracle hay ghi dở dữ liệu nào across 100% lượt đo (`all_oracles_pass = True` ở 121/121 dòng chỉ số). Các thay đổi cải thiện rõ rệt hai điểm nghẽn chính (inference CPU và lọc SQL ở gallery hỗn hợp) cùng độ an toàn transaction, đồng thời giữ nguyên độ chính xác top-1/ngưỡng và hiệu năng đường đọc `E2` / gallery thuần.

## 1. Cấu hình cố định và SHA đối chứng

Toàn bộ phép đo A/B được chạy trong **cùng một phiên Kaggle CPU** (`session = a1281bb702ea4b06ad3ad2a86ac4b81c`, accelerator tắt), SQLite file riêng cho từng lượt, một Uvicorn worker, loopback `127.0.0.1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, dùng chung snapshot tạo từ A (`rng = 23020023`).

- **CPU & tài nguyên:** `Intel(R) Xeon(R) CPU @ 2.20GHz`, 4 vCPU (`affinity = [0, 1, 2, 3]`, `quota = 400000 100000`), RAM `32869296 kB` (~31,3 GiB), `Linux-6.18.55+-x86_64-with-glibc2.39`.
- **Runtime:** Python `3.12.3`, SQLite `3.45.1`, `numpy 2.2.6`, `insightface 2.0`, `onnxruntime 1.23.2` (`CPUExecutionProvider`), `uvicorn 0.53.0`, lock SHA-256 `ddaac31b49e3ee61f888fb44b63e81b5a4cd1a574dc520babd6aea22eb0f085b`.
- **Các SHA chính thức:**
  - **A (Base):** `247db09d98f4cedcc9ad22fdbd112ba328324aa9`
  - **Runner:** `4930c43e9548f8078ef92472a7b2c278a1c52479` (PR `#5`, 4/4 job CI xanh)
  - **Nhánh Thành (1/4 · inference):** `76488497cb388522e985737d75fcd081fc17ad40`
  - **Nhánh Sơn (2/4 · SQL matching):** `9c7618bad992d5700bd65c84bb2a3f0c6a5a9ad3`
  - **B (Tích hợp sau 3 merge 1/4, 2/4, 3/4):** `3b8f8df33c27e7d21d5dcb56b91b77c261902b76`

Xem dữ liệu thô từng lượt tại [kaggle-23020023-3b8f8df](benchmark/phase2/kaggle-23020023-3b8f8df/), [freeze.json](benchmark/phase2/kaggle-23020023-3b8f8df/freeze.json), [bảng A vs B](benchmark/phase2/kaggle-23020023-3b8f8df/integrated/AB-23020023-247db09-3b8f8df.csv), [bảng A vs Thành](benchmark/phase2/kaggle-23020023-3b8f8df/Thanh/AB-23020023-247db09-7648849.csv), [bảng A vs Sơn](benchmark/phase2/kaggle-23020023-3b8f8df/Son/AB-23020023-247db09-9c7618b.csv), [smoke cục bộ](benchmark/phase2/smoke-23020023-b34aaa8/) và [notebook tái lập](benchmark/phase2/kaggle-23020023.ipynb).

## 2. Kết quả A (`247db09`) vs B (`3b8f8df`) trên Kaggle CPU (3 cặp A–B / B–A / A–B)

| Ô tải | Chỉ số | A median `[min, max]` | B median `[min, max]` | B − A | Thay đổi | Oracle |
|---|---|---:|---:|---:|---:|---|
| **E1** (`C=1`) | `p95_ms` | 231,19 `[228,56, 232,08]` | 64,42 `[63,41, 65,02]` | −166,77 ms | **−72,13%** | PASS |
| **E1** (`C=1`) | `success_rps` | 4,83 `[4,77, 4,85]` | 16,28 `[16,26, 16,39]` | +11,45 RPS | **+237,25%** | PASS |
| **E1** (`C=2`) | `p95_ms` | 332,87 `[325,00, 358,78]` | 113,94 `[101,92, 115,40]` | −218,94 ms | **−65,77%** | PASS |
| **E1** (`C=4` · chính) | `p95_ms` | 706,78 `[698,13, 736,83]` | 249,03 `[234,88, 257,89]` | −457,75 ms | **−64,76%** | PASS |
| **E1** (`C=4` · chính) | `success_rps` | 7,41 `[7,34, 7,47]` | 22,55 `[22,34, 22,98]` | +15,14 RPS | **+204,20%** | PASS |
| **E1** (`C=4` · chính) | `rss_kib` | 1.011.400 `[963k, 1.015k]` | 524.012 `[516k, 573k]` | −487.388 KiB | **−48,19%** | PASS |
| **E1** (`C=8`) | `p95_ms` | 1.193,87 `[1.168,4, 1.201,3]` | 562,41 `[495,54, 824,27]` | −631,45 ms | **−52,89%** | PASS |
| **E2** (`C=1`) | `p95_ms` | 8,03 `[7,95, 8,09]` | 8,08 `[8,02, 8,13]` | +0,05 ms | +0,64% *(nhiễu)* | PASS |
| **E2** (`C=2`) | `p95_ms` | 15,57 `[15,56, 16,12]` | 15,58 `[15,13, 15,77]` | +0,01 ms | +0,07% *(nhiễu)* | PASS |
| **E2** (`C=4` · chính) | `p95_ms` | 48,70 `[47,86, 49,52]` | 48,68 `[48,24, 48,79]` | −0,02 ms | **−0,05%** | PASS |
| **E2** (`C=4` · chính) | `rps` | 101,88 `[101,16, 102,40]` | 100,94 `[100,66, 101,25]` | −0,94 RPS | −0,92% | PASS |
| **E2** (`C=8`) | `p95_ms` | 130,41 `[129,90, 131,26]` | 131,25 `[130,24, 132,34]` | +0,84 ms | +0,64% | PASS |
| **E3 pure** (`C=1`) | `p95_ms` | 260,07 `[251,41, 266,31]` | 153,75 `[149,38, 153,88]` | −106,32 ms | **−40,88%** | PASS |
| **E3 pure** (`C=4`) | `p95_ms` | 932,05 `[921,33, 993,48]` | 796,24 `[711,22, 804,34]` | −135,80 ms | **−14,57%** | PASS |
| **E3 mixed** (`C=1`) | `p95_ms` | 266,33 `[257,04, 271,59]` | 88,72 `[84,15, 124,65]` | −177,61 ms | **−66,69%** | PASS |
| **E3 mixed** (`C=4`) | `p95_ms` | 897,27 `[877,62, 926,05]` | 445,23 `[438,30, 446,03]` | −452,03 ms | **−50,38%** | PASS |
| **probe mixed** (`C=1`) | `p95_ms` | 78,55 `[77,52, 79,36]` | 20,85 `[20,01, 24,68]` | −57,70 ms | **−73,45%** | PASS |
| **probe mixed** (`C=1`) | `rps` | 30,77 `[29,72, 31,14]` | 46,72 `[46,20, 47,87]` | +15,95 RPS | **+51,85%** | PASS |
| **probe pure** (`C=1`) | `p95_ms` | 86,61 `[86,13, 87,10]` | 85,85 `[84,81, 86,96]` | −0,76 ms | **−0,87%** | PASS |
| **M1** (`C=1` · embed) | `p95_ms` | 224,59 `[222,47, 227,97]` | 53,94 `[53,27, 54,27]` | −170,64 ms | **−75,98%** | PASS |
| **M1** (`C=1` · embed) | `cold_init_ms` | 962,34 `[952,62, 993,27]` | 988,85 `[981,03, 998,68]` | +26,51 ms | **+2,76%** *(chậm hơn)* | PASS |

**Kết quả âm / không cải thiện cần ghi nhận rõ:**
- `cold_init_ms` ở `M1` tăng nhẹ từ `962,34` lên `988,85 ms` (`+2,76%`, tức chậm hơn ~26,5 ms) — không ảnh hưởng request nóng sau khởi động.
- Đường đọc `E2` gần như giữ nguyên (`−0,05%` ở `C=4`, dao động `+0,64%` ở `C=1` và `C=8`), còn `probe pure` giảm `0,87%` (gần bằng ngưỡng nhiễu giữa các lượt vì gallery 1.000 profile cùng model vẫn phải đọc toàn bộ 1.000 vector).

## 3. Quy kết nguyên nhân theo nhánh đơn lẻ và kiểm tra R1

| Nhánh / chủ sở hữu | Bằng chứng tách riêng (cùng A & session Kaggle CPU) | Kết luận quy kết & nghiệm thu |
|---|---|---|
| **Thành (`7648849`) · Inference** | `M1` hot `p95`: `228,28 → 53,81 ms` (**−76,43%**); `E1 C=4 p95`: `740,86 → 245,41 ms` (**−66,88%**); `probe mixed`: `78,71 → 79,10 ms` (`+0,50%`, không đổi). | Tối ưu nạp chỉ `detection` + `recognition` là nguyên nhân chính hạ p95 toàn request `E1` và giảm ~48% RSS server HTTP. **Đạt mục tiêu ≥10% (vượt ngưỡng lớn).** |
| **Sơn (`9c7618b`) · Matching** | `probe mixed C=1 p95`: `78,76 → 20,24 ms` (**−74,30%**); `probe pure C=1 p95`: `85,60 → 86,69 ms` (`+1,27%`, dao động nhỏ); `E1 C=4 p95`: `661,37 → 687,15 ms` (`+3,90%`, nằm trong nhiễu `658–725 ms`). | Lọc `model_name` + `embedding_dim` ngay trong SQL giảm mạnh chi phí giải mã/so khớp ở gallery hỗn hợp mà không làm chậm gallery thuần quá 10%. **Đạt mục tiêu ≥20% (vượt ngưỡng lớn).** |
| **Đạt (`3b8f8df`) · Transaction R1** | [R1 trên A](benchmark/phase2/kaggle-23020023-3b8f8df/integrated/R1-23020023-247db09d98f4cedcc9ad22fdbd112ba328324aa9.log): **FAIL 10/10** ở `business` (`MultipleFacesDetected` sau flush để lại `[(21..30, 'unmatched')]`). [R1 trên B](benchmark/phase2/kaggle-23020023-3b8f8df/integrated/R1-23020023-3b8f8df33c27e7d21d5dcb56b91b77c261902b76.log): **PASS 50/50** (10/10 `empty` 422 có 1 audit, 10/10 `unmatched` 404 có 1 audit, 10/10 `invalid` 400 có 0 audit, 10/10 `business` 422 có 0 audit, 10/10 `system` 500 có 0 audit). | Khắc phục triệt để lỗi ghi dở sau flush, bảo toàn 2 luồng audit hợp lệ. **Đạt 0 ghi dở, 100% PASS.** |

## 4. Giao thức và công thức đối chiếu

- Mỗi ô HTTP khởi động một Uvicorn worker và một bản sao snapshot độc lập; login, 5 warm-up và 2 giây nghỉ nằm ngoài cửa sổ đo, timeout 60 giây.
- Peak RSS HTTP là `VmRSS` lớn nhất lấy mẫu mỗi 50 ms; probe `M1`/`probe` ghi `VmHWM` của process riêng (`M1` nạp `get_embedder()`, còn `probe` gallery không nạp model để tránh cộng bộ nhớ ONNX).
- Mức giảm latency = `100 × (A − B) / A` (cột `change_percent` trong CSV mang dấu dương khi giảm độ trễ và âm khi tăng độ trễ); mức tăng RPS = `100 × (B − A) / A`.

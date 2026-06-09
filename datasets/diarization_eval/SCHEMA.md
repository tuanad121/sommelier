# Speaker Diarization Golden Schema

- `sample_id`: mã định danh duy nhất cho mỗi dòng trong bản podcast-only.
- `start`, `end`, `duration`: mốc thời gian chuẩn của segment.
- `speaker`: speaker hiện tại của segment trong pipeline.
- `gold_text`: transcript cuối cùng dùng làm gold. Nếu có `correct_script` thì lấy bản sửa; nếu không có thì giữ transcript hiện tại.

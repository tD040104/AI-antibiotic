<?php
declare(strict_types=1);

$successMessage = $_GET['success'] ?? '';
$errorMessage = $_GET['error'] ?? '';
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Hệ thống hỗ trợ điều trị kháng sinh</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            margin: 0;
            padding: 0;
            color: #222;
        }
        header {
            background: #0d47a1;
            color: #fff;
            padding: 20px;
            text-align: center;
        }
        main {
            max-width: 960px;
            margin: 30px auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(15, 44, 90, 0.08);
            padding: 30px 40px;
        }
        h1 {
            margin: 0;
            font-size: 24px;
        }
        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px 30px;
        }
        label {
            font-weight: 600;
            margin-bottom: 6px;
            display: block;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ccd7e4;
            border-radius: 6px;
            font-size: 15px;
            box-sizing: border-box;
        }
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .actions {
            grid-column: 1 / -1;
            text-align: right;
        }
        button {
            padding: 12px 26px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-primary {
            background: #1565c0;
            color: #fff;
        }
        .alert {
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 25px;
        }
        .alert-success {
            background: #e6f7ec;
            color: #0f704f;
        }
        .alert-error {
            background: #fdecea;
            color: #b00020;
        }
        footer {
            text-align: center;
            color: #6b7d95;
            font-size: 14px;
            padding: 20px 0 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Nhập thông tin bệnh nhân</h1>
        <p>Hệ thống multi-agent dự đoán kháng thuốc và gợi ý điều trị</p>
    </header>

    <main>
        <?php if ($successMessage): ?>
            <div class="alert alert-success"><?php echo htmlspecialchars($successMessage, ENT_QUOTES, 'UTF-8'); ?></div>
        <?php endif; ?>
        <?php if ($errorMessage): ?>
            <div class="alert alert-error"><?php echo htmlspecialchars($errorMessage, ENT_QUOTES, 'UTF-8'); ?></div>
        <?php endif; ?>

        <form method="post" action="process_form.php">
            <div>
                <label for="patient_name">Tên bệnh nhân</label>
                <input type="text" id="patient_name" name="patient_name" placeholder="Nguyễn Văn A" required>
            </div>
            <div>
                <label for="age">Tuổi</label>
                <input type="number" id="age" name="age" min="0" max="120" required>
            </div>
            <div>
                <label for="gender">Giới tính</label>
                <select id="gender" name="gender" required>
                    <option value="F">Nữ</option>
                    <option value="M">Nam</option>
                </select>
            </div>
            <div>
                <label for="souches">Chủng vi khuẩn (Souches)</label>
                <input type="text" id="souches" name="souches" placeholder="S123 Escherichia coli" required>
            </div>
            <div>
                <label for="diabetes">Tiểu đường</label>
                <select id="diabetes" name="diabetes" required>
                    <option value="Yes">Có</option>
                    <option value="No" selected>Không</option>
                </select>
            </div>
            <div>
                <label for="hypertension">Tăng huyết áp</label>
                <select id="hypertension" name="hypertension" required>
                    <option value="Yes">Có</option>
                    <option value="No" selected>Không</option>
                </select>
            </div>
            <div>
                <label for="hospital_before">Tiền sử nhập viện</label>
                <select id="hospital_before" name="hospital_before" required>
                    <option value="Yes">Có</option>
                    <option value="No" selected>Không</option>
                </select>
            </div>
            <div>
                <label for="infection_freq">Số lần nhiễm trùng/năm</label>
                <input type="number" id="infection_freq" name="infection_freq" step="0.1" min="0" value="0">
            </div>
            <div>
                <label for="collection_date">Ngày lấy mẫu</label>
                <input type="date" id="collection_date" name="collection_date" required>
            </div>
            <div class="full-width">
                <label for="notes">Ghi chú</label>
                <textarea id="notes" name="notes" placeholder="Thông tin bổ sung (nếu có)"></textarea>
            </div>
            <div class="actions">
                <button type="submit" class="btn-primary">Gửi dữ liệu &amp; Dự đoán</button>
            </div>
        </form>
    </main>

    <footer>
        &copy; <?php echo date('Y'); ?> Hệ thống hỗ trợ điều trị kháng sinh.
    </footer>
</body>
</html>



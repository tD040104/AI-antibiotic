<?php

declare(strict_types=1);

const DB_HOST = 'localhost';
const DB_NAME = 'antibiotic_ai';
const DB_USER = 'root';
const DB_PASS = '';
const DB_CHARSET = 'utf8mb4';

// Đường dẫn tới Python interpreter (có thể cần chỉnh sửa tùy máy)
const PYTHON_BINARY = 'python';

function get_db_connection(): PDO
{
    static $pdo = null;

    if ($pdo instanceof PDO) {
        return $pdo;
    }

    $dsn = sprintf('mysql:host=%s;dbname=%s;charset=%s', DB_HOST, DB_NAME, DB_CHARSET);
    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES => false,
    ];

    try {
        $pdo = new PDO($dsn, DB_USER, DB_PASS, $options);
    } catch (PDOException $e) {
        die('Không thể kết nối MySQL: ' . $e->getMessage());
    }

    return $pdo;
}

function run_python_prediction(array $patientData): array
{
    $scriptPath = realpath(__DIR__ . DIRECTORY_SEPARATOR . 'predict_patient.py');
    if ($scriptPath === false) {
        throw new RuntimeException('Không tìm thấy file predict_patient.py');
    }

    $inputJson = json_encode($patientData, JSON_UNESCAPED_UNICODE);
    if ($inputJson === false) {
        throw new RuntimeException('Không thể encode JSON từ dữ liệu bệnh nhân');
    }

    $command = sprintf(
        '"%s" "%s" --input-json %s',
        PYTHON_BINARY,
        $scriptPath,
        escapeshellarg($inputJson)
    );

    $output = shell_exec($command . ' 2>&1');

    if ($output === null) {
        throw new RuntimeException('Không thể chạy script Python. Kiểm tra cấu hình PYTHON_BINARY.');
    }

    $response = json_decode($output, true);
    if (!is_array($response)) {
        throw new RuntimeException('Phản hồi từ Python không phải dạng JSON: ' . $output);
    }

    if (($response['status'] ?? 'error') !== 'ok') {
        $message = $response['message'] ?? 'Không rõ lỗi';
        throw new RuntimeException('Python báo lỗi: ' . $message);
    }

    return $response['data'];
}



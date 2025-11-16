<?php
declare(strict_types=1);

require_once __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Location: index.php');
    exit;
}

function redirect_with_message(string $type, string $message): void
{
    header('Location: index.php?' . $type . '=' . urlencode($message));
    exit;
}

$patientName = trim($_POST['patient_name'] ?? '');
$age = (int)($_POST['age'] ?? 0);
$gender = strtoupper(trim($_POST['gender'] ?? 'F'));
$souches = trim($_POST['souches'] ?? '');
$diabetes = $_POST['diabetes'] ?? 'No';
$hypertension = $_POST['hypertension'] ?? 'No';
$hospitalBefore = $_POST['hospital_before'] ?? 'No';
$infectionFreq = (float)($_POST['infection_freq'] ?? 0);
$collectionDate = $_POST['collection_date'] ?? '';
$notes = trim($_POST['notes'] ?? '');

if ($patientName === '' || $age <= 0 || !in_array($gender, ['F', 'M'], true)) {
    redirect_with_message('error', 'Vui lòng nhập chính xác tên, tuổi và giới tính.');
}

if ($souches === '') {
    redirect_with_message('error', 'Vui lòng nhập thông tin chủng vi khuẩn (Souches).');
}

if ($collectionDate === '') {
    redirect_with_message('error', 'Vui lòng chọn ngày lấy mẫu.');
}

$ageGenderValue = sprintf('%d/%s', $age, $gender);

$patientData = [
    'age/gender' => $ageGenderValue,
    'Souches' => $souches,
    'Diabetes' => $diabetes,
    'Hypertension' => $hypertension,
    'Hospital_before' => $hospitalBefore,
    'Infection_Freq' => $infectionFreq,
    'Collection_Date' => $collectionDate,
    'Notes' => $notes,
];

// Thu thập dữ liệu từ form (ví dụ)
$patient = [
    'age' => $_POST['age'] ?? null,
    'sex' => $_POST['sex'] ?? null,
    // ...other fields...
];

// Nếu form gửi các trường riêng lẻ, ưu tiên lấy trực tiếp
$patient = [];
// Ví dụ: điều chỉnh theo tên trường thực tế của form nếu cần
if (!empty($_POST['age'])) $patient['age'] = $_POST['age'];
if (!empty($_POST['sex'])) $patient['sex'] = $_POST['sex'];
if (!empty($_POST['Souches'])) $patient['Souches'] = $_POST['Souches'];
if (!empty($_POST['Diabetes'])) $patient['Diabetes'] = $_POST['Diabetes'];
if (!empty($_POST['Hypertension'])) $patient['Hypertension'] = $_POST['Hypertension'];
if (!empty($_POST['Hospital_before'])) $patient['Hospital_before'] = $_POST['Hospital_before'];
if (!empty($_POST['Infection_Freq'])) $patient['Infection_Freq'] = $_POST['Infection_Freq'];
if (!empty($_POST['Collection_Date'])) $patient['Collection_Date'] = $_POST['Collection_Date'];
if (!empty($_POST['Notes'])) $patient['Notes'] = $_POST['Notes'];

// Nếu không có các trường trên nhưng có một chuỗi 'patient' (pseudo-JSON), chuyển nó thành mảng
function parse_pseudo_json($s) {
    $s = trim($s);
    $s = trim($s, "{} \t\n\r");
    // thay \/' thành / để khớp log
    $s = str_replace('\\/', '/', $s);
    if ($s === '') return [];

    // tách theo dấu phẩy (giả định values không chứa dấu phẩy)
    $parts = explode(',', $s);
    $res = [];
    foreach ($parts as $part) {
        $part = trim($part);
        if ($part === '') continue;
        // tách key:value lần đầu tiên
        $kv = preg_split('/\s*:\s*/', $part, 2);
        if (count($kv) < 2) continue;
        $k = trim($kv[0], " \t\n\r\"'{}");
        $v = trim($kv[1], " \t\n\r\"'{}");
        // bỏ khoảng trắng dư và dùng string cho mọi giá trị
        $res[$k] = $v;
    }
    return $res;
}

if (empty($patient) && isset($_POST['patient']) && is_string($_POST['patient'])) {
    $raw = $_POST['patient'];
    // thử decode nếu là JSON hợp lệ
    $try = json_decode($raw, true);
    if (json_last_error() === JSON_ERROR_NONE && is_array($try)) {
        $patient = $try;
    } else {
        // parse pseudo-JSON dạng { key : value , ... }
        $patient = parse_pseudo_json($raw);
    }
}

// Nếu vẫn rỗng, bạn có thể thu thập tất cả POST và dùng làm fallback
if (empty($patient)) {
    foreach ($_POST as $k => $v) {
        $patient[$k] = $v;
    }
}

// Mã hóa JSON an toàn
$json = json_encode($patient, JSON_UNESCAPED_UNICODE);

// Mã hóa base64 để tránh vấn đề escaping khi truyền qua stdin
$encoded = base64_encode($json);

// Đường dẫn tới Python và script
$python = 'python'; // nếu cần thì thay bằng đường dẫn tuyệt đối tới python.exe
$script = __DIR__ . DIRECTORY_SEPARATOR . 'predict_patient.py';

// Thiết lập descriptors để truyền stdin/stdout/stderr
$descriptorspec = [
    0 => ['pipe', 'r'],
    1 => ['pipe', 'w'],
    2 => ['pipe', 'w']
];

// Gọi Python, truyền cờ --b64 và gửi chuỗi base64 qua stdin
// Nếu có file trạng thái orchestrator (được lưu bởi main.save_state), truyền đường dẫn đó
$state_file = realpath(__DIR__ . '/../models/orchestrator_state.joblib');
$state_arg = '';
if ($state_file && file_exists($state_file)) {
    $state_arg = ' ' . escapeshellarg('--state_path=' . $state_file);
}

$cmd = escapeshellcmd($python) . ' ' . escapeshellarg($script) . ' ' . escapeshellarg('--b64') . $state_arg;
$process = proc_open($cmd, $descriptorspec, $pipes);

if (is_resource($process)) {
    fwrite($pipes[0], $encoded);
    fclose($pipes[0]);

    $output = stream_get_contents($pipes[1]);
    fclose($pipes[1]);

    $err = stream_get_contents($pipes[2]);
    fclose($pipes[2]);

    $return_value = proc_close($process);

    // Ghi log debug (nếu cần)
    $dbg = [
        'time' => date('c'),
        'cmd' => $cmd,
        'return' => $return_value,
        'stdout' => $output,
        'stderr' => $err,
        'sent_json' => $json
    ];
    @file_put_contents(__DIR__ . '/../logs/process_form_debug.log', json_encode($dbg, JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT)."\n", FILE_APPEND);

    if ($return_value !== 0) {
        // Hiển thị stderr rõ hơn
        $msg = "Không thể chạy dự đoán. Python trả lỗi (return={$return_value}). STDERR: " . trim($err) . " STDOUT: " . trim($output);
        echo "<div class=\"alert alert-danger\">" . htmlspecialchars($msg) . "</div>";
    } else {
        if (trim($output) === '') {
            // Python trả mã 0 nhưng không có nội dung -> hiển thị stderr nếu có và báo debug
            $msg = "Python chạy nhưng không trả kết quả. STDERR: " . trim($err) . " (Kiểm tra logs/process_form_debug.log và logs/predict_input.log)";
            echo "<div class=\"alert alert-danger\">" . htmlspecialchars($msg) . "</div>";
        } else {
            header('Content-Type: application/json; charset=utf-8');
            // Nếu output là JSON, in trực tiếp; nếu không, bọc trong <pre> để xem raw
            $trim = trim($output);
            if ((strpos($trim, '{') === 0) || (strpos($trim, '[') === 0)) {
                echo $output;
            } else {
                echo "<pre>" . htmlspecialchars($output) . "</pre>";
            }
        }
    }
} else {
    echo "<div class=\"alert alert-danger\">Không thể khởi tạo tiến trình Python.</div>";
}

try {
    $prediction = run_python_prediction($patientData);
} catch (RuntimeException $e) {
    redirect_with_message('error', 'Không thể chạy dự đoán: ' . $e->getMessage());
}

$pdo = get_db_connection();

$stmt = $pdo->prepare(
    'INSERT INTO patient_records 
    (patient_name, age_gender, gender, souches, diabetes, hypertension, hospital_before, infection_freq, collection_date, notes, prediction_json, probabilities_json, recommendations_json, explanation_json, report_text) 
    VALUES 
    (:patient_name, :age_gender, :gender, :souches, :diabetes, :hypertension, :hospital_before, :infection_freq, :collection_date, :notes, :prediction_json, :probabilities_json, :recommendations_json, :explanation_json, :report_text)'
);

$stmt->execute([
    ':patient_name' => $patientName,
    ':age_gender' => $ageGenderValue,
    ':gender' => $gender,
    ':souches' => $souches,
    ':diabetes' => $diabetes === 'Yes' ? 1 : 0,
    ':hypertension' => $hypertension === 'Yes' ? 1 : 0,
    ':hospital_before' => $hospitalBefore === 'Yes' ? 1 : 0,
    ':infection_freq' => $infectionFreq,
    ':collection_date' => $collectionDate,
    ':notes' => $notes,
    ':prediction_json' => json_encode($prediction['predictions'], JSON_UNESCAPED_UNICODE),
    ':probabilities_json' => json_encode($prediction['probabilities'], JSON_UNESCAPED_UNICODE),
    ':recommendations_json' => json_encode($prediction['recommendations'], JSON_UNESCAPED_UNICODE),
    ':explanation_json' => json_encode($prediction['explanation'], JSON_UNESCAPED_UNICODE),
    ':report_text' => $prediction['report'] ?? '',
]);

redirect_with_message('success', 'Đã lưu thành công và tạo dự đoán cho bệnh nhân.');
?>



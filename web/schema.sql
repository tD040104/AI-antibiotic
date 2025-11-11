CREATE DATABASE IF NOT EXISTS `antibiotic_ai` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `antibiotic_ai`;

CREATE TABLE IF NOT EXISTS `patient_records` (
    `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `patient_name` VARCHAR(255) NOT NULL,
    `age_gender` VARCHAR(32) NOT NULL,
    `gender` ENUM('M', 'F') NOT NULL,
    `souches` VARCHAR(255) NOT NULL,
    `diabetes` TINYINT(1) NOT NULL DEFAULT 0,
    `hypertension` TINYINT(1) NOT NULL DEFAULT 0,
    `hospital_before` TINYINT(1) NOT NULL DEFAULT 0,
    `infection_freq` DECIMAL(6,2) NOT NULL DEFAULT 0.00,
    `collection_date` DATE NOT NULL,
    `notes` TEXT NULL,
    `prediction_json` JSON NOT NULL,
    `probabilities_json` JSON NOT NULL,
    `recommendations_json` JSON NOT NULL,
    `explanation_json` JSON NOT NULL,
    `report_text` TEXT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    INDEX `idx_collection_date` (`collection_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



def detect_anomaly1(row, thresholds):
    if row['daily_freq'] > thresholds['daily_freq']:
        if row['amount'] > thresholds['amount']:
            if row['failed_count'] > thresholds['failed_count']:
                if row['mismatch'] > thresholds['mismatch']:
                    return 'Fraud'
                else:
                    return 'Fraud'
            else:
                if row['mismatch'] > thresholds['mismatch']:
                    if row['mismatch_ratio'] < 0.01 and row['failed_count'] == 0:
                        return 'Not Fraud'
                    else:
                        return 'Fraud'
                else:
                    return 'Not Fraud'
        else:
            if row['failed_count'] > thresholds['failed_count'] and row['mismatch'] > thresholds['mismatch']:
                return 'Fraud'
            else:
                return 'Not Fraud'
    else:
        if row['amount'] > thresholds['amount']:
            if row['failed_count'] > thresholds['failed_count'] or row['mismatch'] > thresholds['mismatch']:
                return 'Fraud'
            else:
                return 'Not Fraud'
        else:
            return 'Not Fraud'

def detect_anomaly2(row):
    if row['failed_count'] > 2 * row['avg_failed']:
        if row['fail_ratio'] > 0.7:
            if row['avg_fail_interval'] < 60:
                return 'Fraud'
            else:
                return 'Fraud'
        else:
            if row['avg_fail_interval'] < 60:
                return 'Fraud'
            else:
                return 'Not Fraud'
    else:
        if row['fail_ratio'] > 0.4:
            return 'Fraud'
        else:
            return 'Not Fraud'

def detect_anomaly3(row):
    if row['mismatch_ratio'] > 1.0:
        return 'Fraud'
    else:
        return 'Not Fraud'

def detect_combined_anomaly(row, thresholds):
    result1 = detect_anomaly1(row, thresholds)
    result2 = detect_anomaly2(row)
    result3 = detect_anomaly3(row)
    if 'Fraud' in [result1, result2, result3]:
        return 'Fraud'
    else:
        return 'Not Fraud'

if __name__ == "__main__":
    main()

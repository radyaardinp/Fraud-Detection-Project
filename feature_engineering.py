def feature_eng(df):
    df = df.copy()
    epsilon = 1e-6

    # Pastikan kolom waktu sudah dalam format datetime
    df['createdTime'] = pd.to_datetime(df['createdTime'], errors='coerce')
    df['updatedTime'] = pd.to_datetime(df['updatedTime'], errors='coerce')

    # Rasio-rasio
    df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
    df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
    df['settlement_ratio'] = df['settlementAmount'] / (df['amount'] + epsilon)
    df['amount_diff_inquiry'] = df['inquiryAmount'] - df['amount']
    df['net_amount'] = df['settlementAmount'] - df['feeAmount']

    # Fitur waktu
    df['trx_hour'] = df['createdTime'].dt.hour
    df['trx_day'] = df['createdTime'].dt.dayofweek
    df['is_weekend'] = df['trx_day'].apply(lambda x: 1 if x >= 5 else 0)
    df['trx_duration'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds()

    return df


def preprocess_form_input(data_dict):
    """
    Fungsi untuk input manual dari form (1 transaksi).
    Input: dictionary
    Output: dataframe hasil feature engineering
    """
    df = pd.DataFrame([data_dict])  # buat dataframe 1 baris
    df_eng = feature_eng(df)        # jalankan feature engineering
    return df_eng

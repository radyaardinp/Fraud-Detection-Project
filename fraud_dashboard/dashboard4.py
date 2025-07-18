def page_analysis():
    """Halaman 2: Analisis dan Visualisasi"""
    
    # Custom CSS untuk halaman analisis
    st.markdown("""
    <style>
        .analysis-header {
            font-size: 2rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2E86AB;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0px 20px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header dengan tombol back
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload", key="back_btn"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    with col2:
        st.markdown('<div class="analysis-header">🔍 Fraud Analysis Results</div>', unsafe_allow_html=True)
    
    # Check if data exists
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ No data found. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Load model components
    model, scaler, selected_features = dashboard.load_model_components()
    
    if model is None:
        st.error("❌ Failed to load model components")
        return
    
    # Get data from session
    original_df = st.session_state.uploaded_data
    
    # Preprocessing
    with st.spinner("🔄 Processing data..."):
        try:
            # Preprocess data
            preprocessed_df = preprocess_for_prediction(original_df)
            
            # Store preprocessed data in session (untuk visualisasi)
            st.session_state.preprocessed_data = preprocessed_df
            
            # Validate and predict
            if not dashboard.validate_data(preprocessed_df, selected_features):
                st.error("❌ Data validation failed")
                return
            
            # Perform prediction
            df_with_pred, X_scaled = dashboard.perform_prediction(preprocessed_df, model, scaler, selected_features)
            
            if df_with_pred is None:
                st.error("❌ Prediction failed")
                return
                
            # Store selected features data (untuk preview)
            selected_features_df = df_with_pred[selected_features + ['predicted_fraud', 'fraud_probability']]
            st.session_state.selected_features_data = selected_features_df
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            return
    
    # Show results
    st.markdown("---")
    
    # Metrics
    total_fraud, fraud_rate = dashboard.create_compact_metrics(df_with_pred)
    
    # Tabs untuk visualisasi - HANYA 2 TABS
    tab1, tab2 = st.tabs(["🔍 Details", "🧠 AI Explanation"])
    
    with tab1:
        st.markdown("### 🔍 Data Analysis Details")
        
        # 1. TAMPILKAN DATAFRAME UTAMA DULU (preprocessing + feature selection + prediction)
        st.markdown("#### 📊 Complete Analysis Results")
        st.info("💡 Data setelah preprocessing, feature selection, dan prediksi fraud")
        
        # Gabungkan data original dengan hasil prediksi untuk tampilan yang lebih lengkap
        complete_df = original_df.copy()
        complete_df['predicted_fraud'] = df_with_pred['predicted_fraud']
        complete_df['fraud_probability'] = df_with_pred['fraud_probability']
        
        # Tampilkan dataframe utama
        st.dataframe(complete_df, use_container_width=True)
        
        # Download button untuk complete data
        csv_complete = complete_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results",
            data=csv_complete,
            file_name='complete_fraud_analysis.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 2. VISUALISASI
        st.markdown("### 📊 Fraud Analysis Visualizations")
        
        # Visualisasi dalam grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Fraud Distribution")
            fig_pie = dashboard.create_fraud_pie_chart(df_with_pred)
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)
        
        with col2:
            st.markdown("#### 💳 Payment Source Distribution")
            fig_payment = dashboard.create_payment_source_chart(st.session_state.preprocessed_data)
            if fig_payment:
                st.pyplot(fig_payment, use_container_width=True)
                plt.close(fig_payment)
            else:
                st.info("No payment source data available")
        
        # Merchant fraud chart (full width)
        st.markdown("#### 🏪 Top Fraud Merchants")
        fig_merchant = dashboard.create_merchant_fraud_chart(st.session_state.preprocessed_data, df_with_pred)
        if fig_merchant:
            st.pyplot(fig_merchant, use_container_width=True)
            plt.close(fig_merchant)
        else:
            st.info("No merchant fraud data available")
        
        # Fraud details section
        st.markdown("---")
        detected_fraud = df_with_pred[df_with_pred['predicted_fraud'] == 1]
        
        if len(detected_fraud) > 0:
            st.markdown(f"### 🚨 Detected Fraud Transactions ({len(detected_fraud)} found)")
            
            # Show fraud data with original columns + predictions
            fraud_with_original = original_df.copy()
            fraud_with_original['predicted_fraud'] = df_with_pred['predicted_fraud']
            fraud_with_original['fraud_probability'] = df_with_pred['fraud_probability']
            
            # Filter hanya yang fraud
            fraud_display = fraud_with_original[fraud_with_original['predicted_fraud'] == 1]
            
            # Show fraud data
            st.dataframe(fraud_display, use_container_width=True)
            
            # Download button for fraud data
            csv_fraud = fraud_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Fraud Data Only",
                data=csv_fraud,
                file_name='fraud_transactions.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        else:
            st.success("🎉 No fraud transactions detected!")
    
    with tab2:
        st.markdown("### 🧠 AI Explanation (LIME)")
        st.markdown("Select a transaction to see why the AI made its prediction:")
        
        if len(df_with_pred) > 0:
            # Select transaction dengan lebih banyak opsi
            max_display = min(100, len(df_with_pred))
            
            # Filter options berdasarkan prediction
            fraud_indices = df_with_pred[df_with_pred['predicted_fraud'] == 1].index.tolist()
            non_fraud_indices = df_with_pred[df_with_pred['predicted_fraud'] == 0].index.tolist()
            
            # Selection method
            explanation_method = st.radio(
                "Choose explanation method:",
                ["Show All", "Show Only Fraud", "Show Only Non-Fraud"]
            )
            
            if explanation_method == "Show Only Fraud":
                available_indices = fraud_indices[:max_display]
            elif explanation_method == "Show Only Non-Fraud":
                available_indices = non_fraud_indices[:max_display]
            else:
                available_indices = list(range(min(max_display, len(df_with_pred))))
            
            if not available_indices:
                st.warning("⚠️ No transactions available for the selected filter")
                return
            
            # Select transaction
            idx_to_explain = st.selectbox(
                "Choose transaction to explain:",
                available_indices,
                format_func=lambda x: f"Transaction {x} - {'🚨 FRAUD' if df_with_pred.iloc[x]['predicted_fraud'] == 1 else '✅ NON-FRAUD'} (Prob: {df_with_pred.iloc[x]['fraud_probability']:.4f})"
            )
            
            # Show transaction details
            st.markdown("#### Transaction Details:")
            selected_transaction = df_with_pred.iloc[idx_to_explain]
            selected_original = original_df.iloc[idx_to_explain]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Prediction:** {'🚨 FRAUD' if selected_transaction['predicted_fraud'] == 1 else '✅ NON-FRAUD'}")
                st.write(f"**Probability:** {selected_transaction['fraud_probability']:.4f}")
            with col2:
                if 'merchantId' in selected_original:
                    st.write(f"**Merchant ID:** {selected_original['merchantId']}")
                if 'paymentSourceCode' in selected_original:
                    st.write(f"**Payment Source:** {selected_original['paymentSourceCode']}")
            
            # Show some feature values
            st.markdown("#### Key Features:")
            feature_sample = selected_features[:5]  # Show first 5 features
            for feature in feature_sample:
                if feature in selected_transaction:
                    st.write(f"**{feature}:** {selected_transaction[feature]}")
            
            # Generate LIME explanation
            if st.button("🔍 Generate AI Explanation", key="lime_btn", use_container_width=True):
                with st.spinner("Generating AI explanation..."):
                    lime_fig = dashboard.create_lime_explanation(
                        X_scaled, selected_features, model, idx_to_explain
                    )
                    if lime_fig:
                        st.pyplot(lime_fig, use_container_width=True)
                        plt.close(lime_fig)
                        
                        st.info("""
                        **How to read this explanation:**
                        - 🟢 Green bars: Features that contribute to NON-FRAUD prediction
                        - 🔴 Red bars: Features that contribute to FRAUD prediction
                        - Longer bars = stronger influence on the prediction
                        - Numbers show the actual impact values
                        """)
                        
                        # Additional explanation text
                        prediction_text = "FRAUD" if selected_transaction['predicted_fraud'] == 1 else "NON-FRAUD"
                        confidence = selected_transaction['fraud_probability']
                        
                        st.markdown(f"""
                        **Summary:**
                        - **Final Prediction:** {prediction_text}
                        - **Confidence Score:** {confidence:.4f}
                        - **Interpretation:** The model predicted this transaction as {prediction_text} 
                          with {confidence*100:.2f}% confidence based on the feature contributions shown above.
                        """)
                    else:
                        st.error("❌ Failed to generate explanation. Please try another transaction.")
        else:
            st.info("No transactions available for explanation")

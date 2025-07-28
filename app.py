import io
import tempfile
import streamlit as st
import pandas as pd
from synthetic_data_generator import DataSynthesizer
from matplotlib.backends.backend_pdf import PdfPages

# --- Core generation functions using a temp file path ---
def generate_data_by_file_bytes(file_bytes: bytes, augmentation_amount: int) -> pd.DataFrame:
    try:
        # Write uploaded bytes to a temp CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            path = tmp.name

        # Load and clean data
        df = pd.read_csv(path)
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Categorical: placeholder then fill
        df[cats] = df[cats].replace("", pd.NA).fillna("NULL")
        # Numeric: impute NaNs with median (instead of 0)
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(round(medians))

        # (Optional) debug  
        #print("in", df[['experianAppBustOutScoreV2']])

        # Save cleaned data for CTGAN
        df.to_csv(path, index=False)

        # Run CTGAN
        synth = DataSynthesizer(
            filename=path,
            categorical_features=cats,
            num_rows=augmentation_amount
        )
        print(augmentation_amount, "-->", cats)
        out = synth.generate_data()
        if out is None or not isinstance(out, pd.DataFrame):
            st.error("Synthetic data generation returned no data. Please check your inputs.")
            return pd.DataFrame()

        # Post-process: revert placeholders to empties, zeros→NA
        out[cats] = out[cats].replace("NULL", "")
        num_out_cols = out.select_dtypes(include=["number"]).columns.tolist()
        out[num_out_cols] = out[num_out_cols].replace(0, pd.NA)

        return out

    except Exception as e:
        st.error(f"File-based generation failed: {e}")
        return pd.DataFrame()


def generate_data_by_attribute_bytes(
    file_bytes: bytes,
    attribute: str,
    category_amounts: dict
) -> pd.DataFrame:
    try:
        # Write uploaded bytes to a temp CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            path = tmp.name

        # Load and clean data
        df = pd.read_csv(path)
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Categorical: placeholder then fill
        df[cats] = df[cats].replace("", pd.NA).fillna("NULL")
        # Numeric: impute NaNs with median
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

        # Ensure each category has ≥2 samples
        real_counts = df[df[attribute] != "NULL"][attribute].value_counts()
        if (real_counts < 2).any():
            st.error(
                "Cannot select this attribute as there is a data record with only one record "
                "for a category; please enrich the data or select another attribute."
            )
            return pd.DataFrame()

        # Save cleaned data for CTGAN
        df.to_csv(path, index=False)

        # Prepare conditional percentages
        amounts = {cat: int(val) for cat, val in category_amounts.items() if val is not None}
        total = sum(amounts.values())
        if total == 0:
            st.warning("Enter at least one positive augmentation amount.")
            return pd.DataFrame()
        cond_percents = {cat: amt / total for cat, amt in amounts.items()}

        # Run CTGAN with conditional settings
        synth = DataSynthesizer(
            filename=path,
            categorical_features=cats,
            num_rows=total,
            conditional_column=attribute,
            conditional_values_percent=cond_percents
        )
        out = synth.generate_data()
        if out is None or not isinstance(out, pd.DataFrame):
            st.error("Synthetic data generation returned no data. Please check conditional settings.")
            return pd.DataFrame()

        # Revert placeholders in output
        out[cats] = out[cats].replace("NULL", "")
        num_out_cols = out.select_dtypes(include=["number"]).columns.tolist()
        out[num_out_cols] = out[num_out_cols].replace(0, pd.NA)

        return out

    except Exception as e:
        st.error(f"Attribute-based generation failed: {e}")
        return pd.DataFrame()


# --- Streamlit callbacks ---
def _gen_file_callback(file_bytes, amt):
    with st.spinner("Generating synthetic data…"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            st.session_state.input_df = df
        except Exception as e:
            st.error(f"Failed to read input file: {e}")
            return

        out = generate_data_by_file_bytes(file_bytes, amt)
        if out.empty:
            st.error("No augmented data generated.")
        st.session_state.result_df = out

        # Precompute PDF metrics (unchanged)
        synth = DataSynthesizer(
            filename="",
            categorical_features=df.select_dtypes(include=["object", "category"]).columns.tolist(),
            num_rows=0
        )
        figs = synth._evaluate_syn_data(df, out)
        pdf_buf = None
        if figs:
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            pdf_buf.seek(0)
        st.session_state.pdf_buffer = pdf_buf

    st.session_state.view = 'results'


def _gen_attr_callback(file_bytes, attribute, cat_amounts):
    with st.spinner("Generating synthetic data…"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            st.session_state.input_df = df
        except Exception as e:
            st.error(f"Failed to read input file: {e}")
            return

        out = generate_data_by_attribute_bytes(file_bytes, attribute, cat_amounts)
        if out.empty:
            st.error("No augmented data generated.")
        st.session_state.result_df = out

        synth = DataSynthesizer(
            filename="",
            categorical_features=df.select_dtypes(include=["object", "category"]).columns.tolist(),
            num_rows=0
        )
        figs = synth._evaluate_syn_data(df, out)
        pdf_buf = None
        if figs:
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            pdf_buf.seek(0)
        st.session_state.pdf_buffer = pdf_buf

    st.session_state.view = 'results'


def _start_over_callback():
    for k in ('view', 'file_bytes', 'input_df', 'result_df', 'pdf_buffer'):
        st.session_state.pop(k, None)
    st.session_state.view = 'input'


# --- Custom CSS for button styling ---
def inject_custom_css():
    st.markdown("""
    <style>
      div.stDownloadButton > button:first-child,
      div.stButton > button:first-child {
        width: 170px;
        height: 45px;
        font-size: 14px;
        font-weight: 500;
        border-radius: 6px;
        border: 1px solid #ddd;
        transition: all 0.2s ease;
      }
      /* Download Report (light red) */
      .download-report div.stDownloadButton > button:first-child {
        background-color: #ffcccb;
        color: #8b0000;
        border-color: #ffaaaa;
      }
      /* Download Data (light green) */
      .download-data div.stDownloadButton > button:first-child {
        background-color: #ccffcc;
        color: #006400;
        border-color: #aaffaa;
      }
      /* Start Over (light grey) */
      .start-over div.stButton > button:first-child {
        background-color: #e0e0e0;
        color: #404040;
        border-color: #cccccc;
      }
    </style>
    """, unsafe_allow_html=True)


# --- App layout ---
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
inject_custom_css()

if 'view' not in st.session_state:
    st.session_state.view = 'input'

if st.session_state.view == 'input':
    st.header("Synthetic Data Generator")
    seed = st.file_uploader("Seed Data File:", type=["csv"])
    if seed:
        b = seed.read()
        st.session_state.file_bytes = b

        df = pd.read_csv(io.BytesIO(b))
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()

        approach = st.selectbox("Augmentation Approach:", ["Entire File", "By Attribute"])
        if approach == "Entire File":
            amt = st.number_input("Augmentation Amount:", min_value=1, value=100)
            st.button("Generate", on_click=_gen_file_callback, args=(b, amt))
        else:
            ai = st.text_input("Enter attribute name (case-insensitive):").strip()
            if ai:
                matches = [c for c in cats if c.lower() == ai.lower()]
                if not matches:
                    st.warning(f"No categorical column named '{ai}'.")
                else:
                    attr = matches[0]
                    vals = df[attr].dropna().unique()
                    tbl = pd.DataFrame({"Category": vals, "Augmentation Amount": [None] * len(vals)})
                    ed = st.data_editor(
                        tbl,
                        num_rows="fixed",
                        column_config={"Category": st.column_config.Column(disabled=True)},
                        hide_index=True
                    )
                    cam = dict(zip(ed["Category"], ed["Augmentation Amount"]))
                    st.button("Generate", on_click=_gen_attr_callback, args=(b, attr, cam))

elif st.session_state.view == 'results':
    st.header("Augmentation Results")

    df0 = st.session_state.input_df
    df1 = st.session_state.result_df

    if df0 is None or df1 is None:
        st.error("No data available. Please generate again.")
    else:
        # display
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original Data")
            st.dataframe(df0, height=400)
        with c2:
            st.subheader("Augmented Data")
            st.dataframe(df1, height=400)

        # ——— fix for compute_emd: make median-imputed copies for metrics ———
        df0_metrics = df0.copy()
        num_cols0 = df0_metrics.select_dtypes(include=["number"]).columns.tolist()
        df0_metrics[num_cols0] = df0_metrics[num_cols0].fillna(df0_metrics[num_cols0].median())

        df1_metrics = df1.copy()
        num_cols1 = df1_metrics.select_dtypes(include=["number"]).columns.tolist()
        df1_metrics[num_cols1] = df1_metrics[num_cols1].fillna(df1_metrics[num_cols1].median())
        # ————————————————————————————————————————————————————————————————

        cats = df0.select_dtypes(include=["object", "category"]).columns.tolist()
        emd = DataSynthesizer(filename="", categorical_features=cats, num_rows=0) \
            ._compute_emd(df0_metrics, df1_metrics)
        tvd = DataSynthesizer(filename="", categorical_features=cats, num_rows=0) \
            ._compute_tvd(df0_metrics, df1_metrics)
        jsd = DataSynthesizer(filename="", categorical_features=cats, num_rows=0) \
            ._compute_jsd(df0_metrics, df1_metrics)

        if emd is not None and tvd is not None and jsd is not None:
            st.markdown(f"""
                <div style="border:1px solid #ccc;
                             padding:12px;
                             background-color:#EEEEEE;
                             border-radius:6px;
                             margin:16px 0;
                             text-align:center;">
                  <span style="font-size:16px; font-weight:normal;">
                    Earth Mover Distance (EMD): {emd:.4f}
                    &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
                    Total Variation Distance (TVD): {tvd:.4f}
                    &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
                    Jensen–Shannon Distance (JSD): {jsd:.4f}
                  </span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Metric computation failed.")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6 = st.columns([6,1,1,2,2,2])
        with col4:
            st.markdown('<div class="download-report">', unsafe_allow_html=True)
            if st.session_state.pdf_buffer:
                st.download_button(
                    "📊 DQ Metrics",
                    data=st.session_state.pdf_buffer,
                    file_name="results.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("📊 DQ Metrics", disabled=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="download-data">', unsafe_allow_html=True)
            combined = pd.concat([df0, df1], ignore_index=True)
            csv = combined.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📁 Download Data",
                data=csv,
                file_name="combined_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col6:
            st.markdown('<div class="start-over">', unsafe_allow_html=True)
            st.button("🔄 Start Over", on_click=_start_over_callback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

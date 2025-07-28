import logging
from ctgan import CTGAN
from table_evaluator import TableEvaluator
import pandas as pd
import numpy as np
import ot
import matplotlib.pyplot as plt

# Configure logging to capture info and error messages with timestamps.
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSynthesizer:
    def __init__(
        self,
        filename,
        categorical_features,
        num_rows,
        conditional_column=None,
        conditional_values_percent=None,
        batch_size=100,
        epochs=20
    ):
        """
        Initialize the DataSynthesizer object with all required parameters.
        """
        self.filename = filename
        self.categorical_features = categorical_features
        self.num_rows = num_rows
        self.conditional_column = conditional_column
        self.conditional_values_percent = conditional_values_percent or {}
        self.batch_size = batch_size
        self.epochs = epochs

    def _evaluate_syn_data(self, real_df, synthetic_df):
        """
        Evaluate similarity between real and synthetic data and return a list of matplotlib Figures.
        """
        figs = []
        try:
            plt.close('all')
            sample_size = min(len(real_df), len(synthetic_df))
            if sample_size > 0:
                evaluator = TableEvaluator(
                    real_df.sample(sample_size, random_state=42),
                    synthetic_df.sample(sample_size, random_state=42),
                    cat_cols=self.categorical_features
                )
                evaluator.visual_evaluation()
                figs = [plt.figure(num) for num in plt.get_fignums()]
            else:
                logging.warning("Not enough data to perform evaluation sampling.")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
        return figs

    def _compute_emd(self, df_real, df_syn, weights_real=None, weights_syn=None, metric='euclidean'):
        """
        Compute and log the multivariate Earth Mover's Distance (EMD) between two datasets,
        first imputing any missing numeric values with column means.
        """
        try:
            # 1) Identify numeric columns
            numeric_cols = [c for c in df_real.columns if c not in self.categorical_features]

            # 2) Warn if there are missing values
            if df_real[numeric_cols].isna().any().any() or df_syn[numeric_cols].isna().any().any():
                logging.warning(
                    "Missing values found in numeric columns; "
                    "imputing with column means for EMD calculation."
                )

            # 3) Impute per‐dataset
            df1 = df_real.copy()
            df2 = df_syn.copy()
            df1[numeric_cols] = df1[numeric_cols].fillna(df1[numeric_cols].mean())
            df2[numeric_cols] = df2[numeric_cols].fillna(df2[numeric_cols].mean())

            # 4) One‐hot encode categoricals
            df1_enc = pd.get_dummies(df1, columns=self.categorical_features)
            df2_enc = pd.get_dummies(df2, columns=self.categorical_features)
            df1_enc, df2_enc = df1_enc.align(df2_enc, join='outer', axis=1, fill_value=0)

            # 5) Build distributions
            X = df1_enc.values.astype(np.float64)
            Y = df2_enc.values.astype(np.float64)
            n, m = X.shape[0], Y.shape[0]
            a = weights_real if weights_real is not None else np.ones(n) / n
            b = weights_syn if weights_syn is not None else np.ones(m) / m

            # 6) Distance matrix and EMD
            M = ot.dist(X, Y, metric=metric)
            M /= M.max() if M.max() != 0 else 1
            emd_value = ot.emd2(a, b, M)

            logging.info(f"Multivariate EMD: {emd_value:.6f}")
            return float(emd_value)

        except Exception as e:
            logging.error(f"Error computing EMD: {e}")
            return None

    def _compute_tvd(self, df_real, df_syn):
        """
        Compute Total Variation Distance (TVD) over categorical distributions (range [0,1]).
        """
        try:
            df1_cat = pd.get_dummies(df_real[self.categorical_features], columns=self.categorical_features)
            df2_cat = pd.get_dummies(df_syn[self.categorical_features], columns=self.categorical_features)
            df1_cat, df2_cat = df1_cat.align(df2_cat, join='outer', axis=1, fill_value=0)

            counts1 = df1_cat.sum().values
            counts2 = df2_cat.sum().values
            n = len(df_real)
            m = len(df_syn)
            f = len(self.categorical_features)

            p = counts1 / (n * f)
            q = counts2 / (m * f)

            tvd = 0.5 * np.abs(p - q).sum()
            logging.info(f"Total Variation Distance: {tvd:.6f}")
            return float(tvd)
        except Exception as e:
            logging.error(f"Error computing TVD: {e}")
            return None

    def _compute_jsd(self, df_real, df_syn):
        """
        Compute Jensen-Shannon Distance (JSD) over categorical distributions (range [0,1]).
        """
        try:
            df1_cat = pd.get_dummies(df_real[self.categorical_features], columns=self.categorical_features)
            df2_cat = pd.get_dummies(df_syn[self.categorical_features], columns=self.categorical_features)
            df1_cat, df2_cat = df1_cat.align(df2_cat, join='outer', axis=1, fill_value=0)

            counts1 = df1_cat.sum().values
            counts2 = df2_cat.sum().values
            n = len(df_real)
            m = len(df_syn)
            f = len(self.categorical_features)

            p = counts1 / (n * f)
            q = counts2 / (m * f)
            m_dist = 0.5 * (p + q)

            mask_p = p > 0
            mask_q = q > 0
            kl_pm = np.sum(p[mask_p] * np.log2(p[mask_p] / m_dist[mask_p]))
            kl_qm = np.sum(q[mask_q] * np.log2(q[mask_q] / m_dist[mask_q]))
            js_div = 0.5 * (kl_pm + kl_qm)
            jsd = np.sqrt(js_div)

            logging.info(f"Jensen-Shannon Distance: {jsd:.6f}")
            return float(jsd)
        except Exception as e:
            logging.error(f"Error computing JSD: {e}")
            return None

    def generate_data(self):
        """
        Main function to generate synthetic data and return a DataFrame
        with synthetic rows first, then the original rows.
        """
        try:
            data = pd.read_csv(self.filename)
            #print(data[['experianAppBustOutScoreV2']])
            generated_dfs = []
            used_rows = 0

            if self.conditional_column and self.conditional_values_percent:
                for value, percent in self.conditional_values_percent.items():
                    n_rows = int(self.num_rows * percent)
                    used_rows += n_rows
                    cond_df = data[data[self.conditional_column] == value]
                    if cond_df.empty:
                        continue
                    synth = CTGAN(batch_size=self.batch_size, epochs=self.epochs, verbose=True)
                    synth.fit(cond_df, self.categorical_features)
                    gen = synth.sample(n_rows)
                    gen[self.conditional_column] = value
                    generated_dfs.append(gen)

                remaining = self.num_rows - used_rows
                if remaining > 0:
                    other_df = data[~data[self.conditional_column].isin(self.conditional_values_percent.keys())]
                    if not other_df.empty:
                        synth = CTGAN(batch_size=self.batch_size, epochs=self.epochs, verbose=True)
                        synth.fit(other_df, self.categorical_features)
                        gen = synth.sample(remaining)
                        generated_dfs.append(gen)

                combined_df = pd.concat(generated_dfs, ignore_index=True)
                print("Conditional column counts:", combined_df[self.conditional_column].value_counts())
            else:
                synth = CTGAN(batch_size=self.batch_size, epochs=self.epochs, verbose=True)
                synth.fit(data, self.categorical_features)
                combined_df = synth.sample(self.num_rows)
                #print(combined_df[['experianAppBustOutScoreV2']])

            # --- Stack synthetic on top of original ---
            final_df = pd.concat([combined_df, data.reset_index(drop=True)], ignore_index=True)
            return final_df

        except Exception as e:
            logging.error(f"Error generating data: {e}")
            return None

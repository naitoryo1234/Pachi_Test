import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


# =============== データ読み込み & 前処理 ===============

def load_estimates(csv_path: str) -> pd.DataFrame:
    """CSV（1/x 形式）を読み込み。設定は int 化し、必要ならスイカ合算列を作る。"""
    df = pd.read_csv(csv_path)
    df["設定"] = df["設定"].astype(int)

    # スイカ合算(1/x) が無ければ、弱/強から作る（1 / (1/弱 + 1/強)）
    if "スイカ合算(1/x)" not in df.columns:
        if {"弱スイカ(1/x)", "強スイカ(1/x)"} <= set(df.columns):
            w = df["弱スイカ(1/x)"].astype(float)
            s = df["強スイカ(1/x)"].astype(float)
            df["スイカ合算(1/x)"] = 1.0 / (1.0 / w + 1.0 / s)
        else:
            raise ValueError("CSVに『スイカ合算(1/x)』が無く、弱/強からの合算も作れません。")

    return df


def estimates_to_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """1/x を確率 p に変換した DataFrame を返す（列名はそのまま）。"""
    prob_df = df.copy()
    # 変換候補（存在するものだけ変換）
    cand_cols = [
        "ベル(1/x)",
        "チェリー(1/x)",
        "弱スイカ(1/x)",
        "強スイカ(1/x)",
        "スイカ合算(1/x)",
        # 3枚役(1/x) は使わない方針なので変換不要
    ]
    for col in cand_cols:
        if col in prob_df.columns:
            prob_df[col] = 1.0 / prob_df[col].astype(float)
    return prob_df


# =============== 尤度計算（多項分布） ===============

def compute_posterior_multinomial(
    total_games: int,
    counts: Dict[str, int],
    prob_df: pd.DataFrame,
    ui_roles: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    役を相互排他とみなす多項分布のベイズ推定。
    - 使う役: ui_roles の列（p_i）＋ その他 p_other = 1 - Σ p_i
    - 尤度（比）: L ∝ Π p_i^{k_i} * p_other^{k_other}
      → log L = Σ k_i log p_i + k_other log p_other
    """
    rows: List[Dict] = []
    eps = 1e-12

    for _, row in prob_df.iterrows():
        ps = []
        ks = []
        for col, _label in ui_roles:
            p_i = float(row[col])            # すでに p = 1/x に変換済み
            p_i = min(max(p_i, eps), 1.0)   # 安全側クリップ
            k_i = float(counts[col])        # 観測回数（0 でもOK）
            ps.append(p_i)
            ks.append(k_i)

        p_sum = sum(ps)
        p_other = max(eps, 1.0 - p_sum)
        k_other = max(0.0, float(total_games) - sum(ks))

        log_like = 0.0
        for k_i, p_i in zip(ks, ps):
            # p_i > 0 はクリップ済み
            log_like += k_i * math.log(p_i)
        log_like += k_other * math.log(p_other)

        rows.append({"設定": int(row["設定"]), "log_like": log_like})

    post_df = pd.DataFrame(rows).sort_values("設定").reset_index(drop=True)
    # 数値安定化
    m = float(post_df["log_like"].max())
    w = np.exp(post_df["log_like"] - m)
    s = float(w.sum())
    post_df["posterior"] = (w / s) if s > 0 else (1.0 / len(post_df))
    return post_df


# =============== 参考統計（χ² など、任意） ===============

def compute_statistics(
    total_games: int,
    counts: Dict[str, int],
    prob_df: pd.DataFrame,
    ui_roles: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    参考用の χ²（各役を二項近似で個別に比較）と平方誤差。
    ベイズ判定は上の多項分布 posterior を用いる。
    """
    rows: List[Dict] = []
    for _, row in prob_df.iterrows():
        setting = int(row["設定"])
        chi2 = 0.0
        se = 0.0
        for col, _label in ui_roles:
            p_est = float(row[col])
            observed = float(counts[col])
            expected = max(1e-12, total_games * p_est)
            chi2 += (observed - expected) ** 2 / expected

            p_obs = observed / max(1, total_games)
            se += (p_obs - p_est) ** 2

        rows.append({
            "設定": setting,
            "chi2": chi2,
            "sq_error": se,
        })

    stat_df = pd.DataFrame(rows).sort_values("設定").reset_index(drop=True)
    # 確率っぽい指標は posterior に任せるのでここでは正規化しない
    return stat_df


# =============== Streamlit アプリ本体 ===============

def main() -> None:
    st.set_page_config(page_title="エヴァ 設定判別（推計版）", page_icon="🎰", layout="centered")
    st.title("エヴァ 設定判別（推計値ベース）")
    st.caption("本アプリは公開情報を基にした推計値を使用しています。公式値ではありません。参考目安としてご利用ください。")
    st.caption("数値は推計値であり、実機の挙動を保証しません。役は相互排他として扱い、多項分布ベイズで推定します。")

    # 日本語フォント
    try:
        rcParams["font.sans-serif"] = ["Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "TakaoGothic", "IPAexGothic", "DejaVu Sans"]
        rcParams["font.family"] = "sans-serif"
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 日本語フォントが無ければ Noto Sans JP を動的取得して登録（Streamlit Cloud 向け）
    def ensure_jp_font() -> bool:
        try:
            from matplotlib.font_manager import findfont, FontProperties, fontManager
            candidates = [
                "Noto Sans CJK JP",
                "IPAPGothic",
                "IPAexGothic",
                "TakaoGothic",
                "Yu Gothic",
                "Meiryo",
                "MS Gothic",
            ]
            for name in candidates:
                path = findfont(FontProperties(family=name), fallback_to_default=False)
                if path and isinstance(path, str) and len(path) > 0:
                    return True

            # 動的ダウンロード（軽量な Noto Sans JP）
            import urllib.request
            cache_dir = os.path.join(".fonts")
            os.makedirs(cache_dir, exist_ok=True)
            font_path = os.path.join(cache_dir, "NotoSansJP-Regular.otf")
            if not os.path.exists(font_path):
                url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansJP-Regular.otf"
                urllib.request.urlretrieve(url, font_path)
            font_manager.fontManager.addfont(font_path)
            rcParams["font.family"] = FontProperties(fname=font_path).get_name()
            rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            return False
        return False

    use_jp_plot_labels = ensure_jp_font()

    # データ読込
    try:
        raw_df = load_estimates("data/estimated_values.csv")
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        return

    prob_df = estimates_to_probabilities(raw_df)

    # ===== サイドバー入力 =====
    st.sidebar.header("入力")

    # セッション初期化（値保持用）
    if "total_games" not in st.session_state:
        st.session_state.total_games = 1000
    if "suika_mode" not in st.session_state:
        st.session_state.suika_mode = "合算（おすすめ）"
    if "q_suika" not in st.session_state:
        st.session_state.q_suika = 0.95

    # スイカの扱い切替：合算 or 弱のみ
    suika_mode = st.sidebar.radio(
        "スイカの扱い",
        options=("合算（おすすめ）", "弱のみ"),
        index=(0 if st.session_state.suika_mode.startswith("合算") else 1),
        help="実戦では合算カウントが一般的なので『合算』推奨。弱のみカウントの場合は『弱のみ』を選択。"
    , key="suika_mode")
    if suika_mode.startswith("合算"):
        suika_col = "スイカ合算(1/x)"
        suika_label = "スイカ（合算）"
    else:
        suika_col = "弱スイカ(1/x)"
        suika_label = "弱スイカ"

    # スイカ取得率 q：弱のみのときだけ出す（実戦補正）
    q_suika = 1.0
    if suika_mode.startswith("弱"):
        q_suika = st.sidebar.slider(
            "弱スイカ取得率 q（実戦補正）", min_value=0.80, max_value=1.00, value=st.session_state.q_suika, step=0.01,
            help="取りこぼし等で実効が下がる想定。期待確率 p を q 倍で補正します。",
            key="q_suika",
        )
        st.caption("弱のみ：取得率qで実戦補正中。判別が不安定なら合算にすると安定します。")
    else:
        st.caption("推奨：取りこぼし影響を受けにくい合算を使っています。")

    # 今回使う役（3枚役は除外）
    ui_roles: List[Tuple[str, str]] = [
        ("ベル(1/x)", "ベル"),
        ("チェリー(1/x)", "チェリー"),
        (suika_col, suika_label),
    ]

    total_games = st.sidebar.number_input("総ゲーム数", min_value=1, max_value=1_000_000, value=st.session_state.total_games, step=10, key="total_games")

    # 小役カウント入力（列名キーで持つ）
    counts: Dict[str, int] = {}
    for col, label in ui_roles:
        key_name = f"count_{col}"
        if key_name not in st.session_state:
            st.session_state[key_name] = 0
        counts[col] = int(
            st.sidebar.number_input(f"{label} 回数", min_value=0, max_value=1_000_000, value=st.session_state[key_name], step=1, key=key_name)
        )

    # クリアボタン（値はセッションで保持。更新ボタンは不要）
    col_btn2 = st.sidebar
    def _clear_all():
        st.session_state.total_games = 1000
        st.session_state.suika_mode = "合算（おすすめ）"
        st.session_state.q_suika = 0.95
        for col, _ in ui_roles:
            st.session_state[f"count_{col}"] = 0
    with col_btn2:
        if st.button("クリア", use_container_width=True):
            _clear_all()
            st.experimental_rerun()

    # バリデーション：相互排他ゆえ 合計 > 総G はNG
    sum_counts = sum(counts[col] for col, _ in ui_roles)
    if sum_counts > total_games:
        st.error("入力回数の合計が総ゲーム数を超えています。値を確認してください。")
        return

    # 0回も尤度に含まれる旨の注記
    zero_roles = [label for col, label in ui_roles if counts[col] == 0]
    if len(zero_roles) > 0:
        st.caption("注: 0回でも尤度に反映されます — " + ", ".join(zero_roles))

    # ===== 計算（確率表を必要に応じて補正） =====
    prob_eff = prob_df.copy()
    if suika_mode.startswith("弱"):
        col = "弱スイカ(1/x)"
        if col in prob_eff.columns:
            prob_eff[col] = np.clip(prob_eff[col] * q_suika, 1e-12, 1 - 1e-12)

    stat_df = compute_statistics(total_games, counts, prob_eff, ui_roles)
    post_df = compute_posterior_multinomial(total_games, counts, prob_eff, ui_roles)

    # ===== 表示（実測テーブル） =====
    obs_rows: List[Dict] = []
    for col, label in ui_roles:
        c = counts[col]
        p_obs = c / max(1, total_games)
        one_over = (1.0 / p_obs) if p_obs > 0 else np.nan
        obs_rows.append({
            "役": label,
            "回数": int(c),
            "実測(1/x)": one_over,
            "実測確率(%)": p_obs * 100.0,
        })
    observed_df = pd.DataFrame(obs_rows)

    st.subheader("実測（入力から計算）")
    st.dataframe(
        observed_df.style.format({
            "実測(1/x)": "{:.2f}",
            "実測確率(%)": "{:.3f}%",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    # 合算（今回は3役合算）
    total_k = sum(counts[col] for col, _ in ui_roles)
    p_obs_sum = total_k / max(1, total_games)
    one_over_sum = (1.0 / p_obs_sum) if p_obs_sum > 0 else np.nan
    observed_sum_df = pd.DataFrame([{
        "対象": "合算（ベル+チェリー+スイカ）",
        "回数": int(total_k),
        "実測(1/x)": one_over_sum,
        "実測確率(%)": p_obs_sum * 100.0,
    }])
    st.dataframe(
        observed_sum_df.style.format({
            "実測(1/x)": "{:.2f}",
            "実測確率(%)": "{:.3f}%",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    # ===== 結果 =====
    best_row = post_df.loc[post_df["posterior"].idxmax()]
    best_setting = int(best_row["設定"])

    st.subheader("結果")
    st.write(f"最も近い設定は **設定{best_setting}** です。")

    # 信頼度（トップと2位の差）
    ranked = post_df.sort_values("posterior", ascending=False).reset_index(drop=True)
    top_p = float(ranked.loc[0, "posterior"]) if len(ranked) > 0 else 0.0
    second_p = float(ranked.loc[1, "posterior"]) if len(ranked) > 1 else 0.0
    gap = top_p - second_p
    if gap >= 0.30:
        confidence_text, stars = "非常に高", "★★★★★"
    elif gap >= 0.20:
        confidence_text, stars = "高", "★★★★☆"
    elif gap >= 0.12:
        confidence_text, stars = "中", "★★★☆☆"
    elif gap >= 0.06:
        confidence_text, stars = "低", "★★☆☆☆"
    else:
        confidence_text, stars = "非常に低", "★☆☆☆☆"
    st.info(f"識別の信頼度: {stars}（{confidence_text}）")

    # 判別不能アラートと推奨G数
    if total_games < 1500 and gap < 0.06:
        st.warning("試行数が少なく、設定差が統計誤差に埋もれています（判定は参考程度）。スイカは合算を推奨。")
    elif suika_mode.startswith("弱") and gap < 0.06:
        st.warning("弱スイカのみ・取得率補正ありでも判別が弱いです。合算に切り替えると安定します。")

    # ===== 可視化：事後確率（Matplotlibに戻す） =====
    st.subheader("設定ごとの事後確率")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if use_jp_plot_labels:
        x_labels = [f"設定{int(s)}" for s in post_df["設定"].tolist()]
        y_label = "事後確率（%）"
    else:
        x_labels = [f"Setting {int(s)}" for s in post_df["設定"].tolist()]
        y_label = "Posterior probability (%)"
    y_vals = post_df["posterior"].to_numpy() * 100.0
    winner_idx = int(post_df["posterior"].idxmax())
    colors = ["#4C78A8" for _ in range(len(y_vals))]
    if 0 <= winner_idx < len(colors):
        colors[winner_idx] = "#d62728"
    ax.bar(x_labels, y_vals, color=colors)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", which="major", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(100.0, float(y_vals.max() * 1.2)))
    for idx, v in enumerate(y_vals):
        ax.text(idx, v + max(1.0, y_vals.max() * 0.02), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig, clear_figure=True)

    # ===== 詳細テーブル =====
    st.markdown("---")
    st.subheader("詳細テーブル（ベイズ事後確率 & 参考統計）")
    stats = compute_statistics(total_games, counts, prob_eff, ui_roles)
    display_df = post_df.merge(
        stats[["設定", "chi2", "sq_error"]],
        on="設定", how="left"
    )
    display_df["事後確率(%)"] = display_df["posterior"] * 100.0
    # 日本語見出しに置換
    display_df = display_df.rename(columns={
        "chi2": "カイ二乗値",
        "sq_error": "平方誤差（確率差の二乗）",
    })
    st.dataframe(
        display_df[["設定", "事後確率(%)", "カイ二乗値", "平方誤差（確率差の二乗）"]]
        .style.format({
            "事後確率(%)": "{:.2f}%",
            "カイ二乗値": "{:.3f}",
            "平方誤差（確率差の二乗）": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("推計の前提と注意"):
        st.write(
            "- 小役の出現は相互排他とみなし、多項分布ベイズで設定ごとの事後確率を算出します。\n"
            "- スイカは『合算』または『弱のみ』を切替可能（UIのラジオボタン）。\n"
            "- 0回の役も尤度に反映されます（起きなかったこと自体が情報）。\n"
            "- 入力回数の合計が総ゲーム数を超える値は無効です。\n"
            "- 数値は推計値（CSV）に基づく参考結果であり、実機の挙動を保証しません。\n"
        )


if __name__ == "__main__":
    main()



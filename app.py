import streamlit as st
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="PhishGuard — URL Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
.stButton > button {
    background: #00ff88; color: #0e0e10; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; border: none; border-radius: 6px;
    padding: 12px 32px; font-size: 14px; width: 100%;
}
.stButton > button:hover { background: #00cc6a; }
.result-box { border-radius: 10px; padding: 24px 28px; margin: 20px 0; border: 1px solid; }
.result-phishing { background: #1f0a0a; border-color: #ff4444; }
.result-legit    { background: #0a1f12; border-color: #00ff88; }
.result-title    { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; margin-bottom: 6px; }
.result-sub      { font-size: 13px; color: #888; font-family: 'IBM Plex Mono', monospace; }
.feature-row     { display: flex; justify-content: space-between; padding: 7px 0;
                   border-bottom: 1px solid #1e1e24; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
.feature-name    { color: #aaa; }
.feature-val-bad { color: #ff6666; font-weight: 600; }
.feature-val-good{ color: #00ff88; font-weight: 600; }
.feature-val-neutral { color: #555; }
.section-label   { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
                   letter-spacing: 0.1em; color: #555; text-transform: uppercase; margin: 24px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── Feature names (48 — must match training order) ──────────────────────────
FEATURE_NAMES = [
    'NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash',
    'NumDashInHostname','AtSymbol','TildeSymbol','NumUnderscore',
    'NumPercent','NumQueryComponents','NumAmpersand','NumHash',
    'NumNumericChars','NoHttps','RandomString','IpAddress',
    'DomainInSubdomains','DomainInPaths','HttpsInHostname',
    'HostnameLength','PathLength','QueryLength','DoubleSlashInPath',
    'NumSensitiveWords','EmbeddedBrandName','PctExtHyperlinks',
    'PctExtResourceUrls','ExtFavicon','InsecureForms',
    'RelativeFormAction','ExtFormAction','AbnormalFormAction',
    'PctNullSelfRedirectHyperlinks','FrequentDomainNameMismatch',
    'FakeLinkInStatusBar','RightClickDisabled','PopUpWindow',
    'SubmitInfoToEmail','IframeOrFrame','MissingTitle',
    'ImagesOnlyInForm','SubdomainLevelRT','UrlLengthRT',
    'PctExtResourceUrlsRT','AbnormalExtFormActionR',
    'ExtMetaScriptLinkRT','PctExtNullSelfRedirectHyperlinksRT'
]

RISK_FEATURES = {
    'AtSymbol','TildeSymbol','NoHttps','RandomString','IpAddress',
    'DomainInSubdomains','DomainInPaths','HttpsInHostname','DoubleSlashInPath',
    'NumSensitiveWords','EmbeddedBrandName','ExtFavicon','InsecureForms',
    'RelativeFormAction','ExtFormAction','AbnormalFormAction',
    'FrequentDomainNameMismatch','FakeLinkInStatusBar','RightClickDisabled',
    'PopUpWindow','SubmitInfoToEmail','IframeOrFrame','MissingTitle','ImagesOnlyInForm'
}

# ── Sample rows ──────────────────────────────────────────────────────────────
SAMPLE_PHISHING = dict(zip(FEATURE_NAMES, [
    3,2,4,98,5,2,1,0,2,1,3,2,0,8,1,1,0,1,1,1,
    28,42,24,1,2,1,0.8,0.7,1,1,0,1,1,0.6,1,1,1,1,0,1,0,1,1,1,1,1,1,1
]))
SAMPLE_LEGIT = dict(zip(FEATURE_NAMES, [
    2,1,2,32,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
    10,10,6,0,0,0,0.1,0.1,0,0,1,0,0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]))

# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "model/phishing_model.pkl"
    if os.path.exists(path):
        return joblib.load(path), False
    return None, True

def predict(bundle, features, is_dummy):
    vals = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)
    if is_dummy:
        risk = sum(features[f] for f in RISK_FEATURES if features.get(f, 0) > 0)
        raw  = min((risk * 4 + min(features['UrlLength'] / 15, 20)) / 120, 0.95)
        conf = max(raw, 0.05)
        label = 1 if conf > 0.45 else 0
        return label, conf if label else 1 - conf, None
    X_sc  = bundle['scaler'].transform(vals)
    label = int(bundle['model'].predict(X_sc)[0])
    proba = bundle['model'].predict_proba(X_sc)[0]
    imps  = getattr(bundle['model'], 'feature_importances_', None)
    return label, proba[label], imps

# ── Charts ───────────────────────────────────────────────────────────────────
def confidence_bar(conf, is_phishing):
    fig, ax = plt.subplots(figsize=(6, 0.5))
    fig.patch.set_facecolor('#0e0e10')
    ax.set_facecolor('#1a1a1f')
    color = '#ff4444' if is_phishing else '#00ff88'
    ax.barh(0, 1,    color='#222228', height=0.6, edgecolor='none')
    ax.barh(0, conf, color=color,    height=0.6, edgecolor='none')
    ax.set_xlim(0,1); ax.set_ylim(-0.5,0.5); ax.axis('off')
    ax.text(min(conf+0.02,0.93), 0, f"{conf*100:.1f}%",
            va='center', ha='left', color=color,
            fontsize=11, fontweight='bold', fontfamily='monospace')
    plt.tight_layout(pad=0)
    return fig

def feature_chart(features, importances=None):
    if importances is not None:
        idx    = np.argsort(importances)[::-1][:15]
        names  = [FEATURE_NAMES[i] for i in idx]
        vals   = [importances[i]   for i in idx]
        colors = ['#ff6666' if FEATURE_NAMES[i] in RISK_FEATURES else '#00ff88' for i in idx]
    else:
        names, vals, colors = [], [], []
        for f in FEATURE_NAMES:
            v = features[f]
            if f in ('UrlLength','HostnameLength','PathLength','QueryLength'):
                nv = min(v/200, 1.0)
            elif f in ('NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname',
                       'NumUnderscore','NumNumericChars','NumQueryComponents','NumAmpersand','NumSensitiveWords'):
                nv = min(v/10, 1.0)
            else:
                nv = min(float(v), 1.0)
            names.append(f); vals.append(nv)
            colors.append('#ff6666' if f in RISK_FEATURES and v > 0
                          else '#00ff88' if f not in RISK_FEATURES and v > 0
                          else '#2a2a30')

    short = [n[:22] for n in names]
    fig, ax = plt.subplots(figsize=(6, max(4, len(names)*0.33)))
    fig.patch.set_facecolor('#0e0e10'); ax.set_facecolor('#0e0e10')
    ax.barh(short[::-1], vals[::-1], color=colors[::-1], height=0.6, edgecolor='none')
    ax.set_xlim(0,1.1); ax.xaxis.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    for lbl in ax.get_yticklabels():
        lbl.set_color('#888'); lbl.set_fontfamily('monospace'); lbl.set_fontsize(8)
    plt.tight_layout(pad=1)
    return fig

# ── Layout ───────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ PhishGuard")
st.markdown("<p style='color:#555;font-family:IBM Plex Mono,monospace;font-size:13px;margin-top:-12px;'>Phishing URL Detection · Cybersecurity ML Project</p>", unsafe_allow_html=True)

bundle, is_dummy = load_model()

if is_dummy:
    st.info("⚠️ **Demo mode** — heuristic scorer active. Drop `phishing_model.pkl` in a `model/` folder to use the real trained model.", icon="🔧")

st.markdown("---")

# Sample buttons
st.markdown("<div class='section-label'>Quick samples</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
if c1.button("🔴 Load phishing sample"):
    st.session_state['features'] = SAMPLE_PHISHING.copy()
    st.session_state['ready']    = True
if c2.button("🟢 Load legitimate sample"):
    st.session_state['features'] = SAMPLE_LEGIT.copy()
    st.session_state['ready']    = True

# Manual input (key fields only)
st.markdown("<div class='section-label'>Or enter values manually</div>", unsafe_allow_html=True)
with st.expander("Expand to enter feature values"):
    cols = st.columns(3)
    manual = {}
    key_fields = [
        ('UrlLength',0), ('NumDots',0), ('SubdomainLevel',0),
        ('AtSymbol',0), ('NoHttps',0), ('IpAddress',0),
        ('NumSensitiveWords',0), ('IframeOrFrame',0), ('PopUpWindow',0),
    ]
    for i,(f,default) in enumerate(key_fields):
        manual[f] = cols[i%3].number_input(f, min_value=0, value=default, key=f"m_{f}")
    if st.button("Use these values"):
        full = {f: 0 for f in FEATURE_NAMES}
        full.update(manual)
        st.session_state['features'] = full
        st.session_state['ready']    = True

# Analyse
st.markdown("---")
if st.button("🔍 Analyse") and st.session_state.get('ready'):
    features = st.session_state['features']
    label, conf, imps = predict(bundle, features, is_dummy)
    is_phishing = label == 1

    color      = "#ff4444" if is_phishing else "#00ff88"
    box_class  = "result-phishing" if is_phishing else "result-legit"
    result_txt = "PHISHING DETECTED" if is_phishing else "LIKELY LEGITIMATE"
    icon       = "🚨" if is_phishing else "✅"

    st.markdown(f"""
    <div class='result-box {box_class}'>
        <div class='result-title' style='color:{color}'>{icon} {result_txt}</div>
        <div class='result-sub'>Confidence: {conf*100:.1f}% &nbsp;·&nbsp;
        Model: {"Heuristic (demo)" if is_dummy else "Random Forest"}</div>
    </div>""", unsafe_allow_html=True)

    st.pyplot(confidence_bar(conf, is_phishing), use_container_width=True)
    st.markdown("---")

    st.markdown("<div class='section-label'>Feature breakdown</div>", unsafe_allow_html=True)
    rows = ""
    for f in FEATURE_NAMES:
        v = features[f]
        cls = ("feature-val-bad"  if f in RISK_FEATURES and v > 0 else
               "feature-val-good" if f not in RISK_FEATURES and v > 0 else
               "feature-val-neutral")
        rows += f"<div class='feature-row'><span class='feature-name'>{f}</span><span class='{cls}'>{v}</span></div>"
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Feature visualisation</div>", unsafe_allow_html=True)
    st.pyplot(feature_chart(features, imps), use_container_width=True)
    st.markdown("<p style='font-size:11px;color:#444;font-family:monospace;text-align:center;'>Red = risk · Green = safe/neutral · Grey = absent</p>", unsafe_allow_html=True)

elif st.button("🔍 Analyse"):
    st.warning("Please load a sample or enter values first.")

st.markdown("---")
st.markdown("<p style='font-size:11px;color:#2a2a2a;font-family:monospace;text-align:center;'>PhishGuard · Cybersecurity ML Project</p>", unsafe_allow_html=True)

"""
Streamlit Demo App for Arabic Sentiment Analysis.

Run: streamlit run app/streamlit_app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

from src.inference import SentimentPredictor
from src.config import MODELS_DIR


# Page config
st.set_page_config(
    page_title="Arabic Sentiment Analysis",
    page_icon="🇸🇦",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1e3a5f;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { background-color: #d4edda; border: 2px solid #28a745; }
    .negative { background-color: #f8d7da; border: 2px solid #dc3545; }
    .neutral { background-color: #fff3cd; border: 2px solid #ffc107; }
    .stTextArea textarea { font-size: 18px; direction: rtl; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model (cached)."""
    model_dir = os.path.join(MODELS_DIR, "arabert-sentiment")
    if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
        return None
    return SentimentPredictor.from_pretrained(model_dir)


def create_gauge_chart(probabilities: dict) -> go.Figure:
    """Create a gauge chart for sentiment probabilities."""
    colors = {"negative": "#dc3545", "neutral": "#ffc107", "positive": "#28a745"}

    fig = go.Figure()

    labels = list(probabilities.keys())
    values = list(probabilities.values())
    bar_colors = [colors.get(l, "#666") for l in labels]

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
        textfont=dict(size=16),
    ))

    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=300,
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def main():
    st.markdown("<h1 class='main-title'>🇸🇦 Arabic Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#666;'>Powered by AraBERT — Fine-tuned for Arabic sentiment classification</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Load model
    predictor = load_model()

    if predictor is None:
        st.error("⚠️ No trained model found. Please run training first:")
        st.code("python scripts/train.py", language="bash")
        st.info("The model will be saved to `models/arabert-sentiment/`")
        return

    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "📝 Enter Arabic text:",
            height=120,
            placeholder="اكتب نصاً عربياً هنا...",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    # Example texts
    st.markdown("**Try these examples:**")
    examples = [
        "هذا المنتج ممتاز والجودة عالية جداً",
        "الخدمة سيئة جداً ولن أعود مرة أخرى",
        "الفيلم كان عادي، لا بأس به",
        "أحب هذا المطعم، الطعام لذيذ والأسعار معقولة",
        "تجربة مخيبة للآمال، لا أنصح بها أبداً",
    ]

    cols = st.columns(len(examples))
    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"Ex {i+1}", key=f"ex_{i}", use_container_width=True):
                text_input = example

    # Predict
    if analyze_btn or text_input:
        if text_input and text_input.strip():
            with st.spinner("Analyzing..."):
                result = predictor.predict(text_input)

            # Display result
            emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            color_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

            label = result["label"]
            confidence = result["confidence"]
            css_class = color_map.get(label, "neutral")
            emoji = emoji_map.get(label, "")

            st.markdown(
                f"""<div class='result-box {css_class}'>
                    <h2 style='text-align:center;margin:0;'>{emoji} {label.upper()}</h2>
                    <p style='text-align:center;margin:5px 0 0 0;font-size:18px;'>
                        Confidence: {confidence:.1%}
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Probability chart
            st.plotly_chart(
                create_gauge_chart(result["probabilities"]),
                use_container_width=True,
            )

            # Details
            with st.expander("🔧 Processing Details"):
                st.json({
                    "original_text": result["text"],
                    "processed_text": result["processed_text"],
                    "probabilities": result["probabilities"],
                })

    # Footer
    st.divider()
    st.markdown(
        """<p style='text-align:center;color:#999;font-size:12px;'>
        Built with AraBERT (aubmindlab/bert-base-arabertv2) • Fine-tuned on Arabic sentiment data
        <br>By <a href='https://github.com/salehA13'>Saleh Almansour</a>
        </p>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

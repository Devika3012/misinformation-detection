import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
from PIL import Image
from model import MultimodalModel
import base64
import plotly.graph_objects as go

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Multimodal Misinformation Detection",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# BACKGROUND IMAGE
# ----------------------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("background.jpg")

st.markdown(f"""
<style>

.stApp {{
background-image:url("data:image/jpg;base64,{bg}");
background-size:cover;
background-position:center;
}}

.stApp::before {{
content:"";
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
background:rgba(0,0,0,0.55);
z-index:0;
}}

.block-container {{
position:relative;
z-index:1;
max-width:900px;
margin:auto;
}}

.title-box {{
background:rgba(40,40,40,0.85);
padding:25px;
border-radius:12px;
text-align:center;
margin-bottom:30px;
}}

.title-text {{
font-size:36px;
color:#ff4040;
font-weight:bold;
}}

.subtitle {{
color:#ffd54d;
}}

.input-panel {{
background:rgba(40,40,40,0.85);
padding:20px;
border-radius:10px;
margin-bottom:20px;
}}

.stButton>button {{
background-color:#ff4040;
color:white;
font-weight:bold;
width:100%;
border-radius:8px;
height:45px;
}}

.card {{
background:rgba(40,40,40,0.9);
padding:20px;
border-radius:12px;
height:180px;
}}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------

st.markdown("""
<div class="title-box">
<div class="title-text">Multimodal Misinformation Detection</div>
<div class="subtitle">AI system analyzing text and images to detect misinformation</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():

    model = MultimodalModel()
    model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    model.eval()

    return model

model = load_model()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------
# INPUT PANEL
# ----------------------------

st.markdown('<div class="input-panel">', unsafe_allow_html=True)

text_input = st.text_input("Enter claim or news headline")

uploaded_image = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

analyze = st.button("Analyze Misinformation")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# KEYWORD RISK DETECTION
# ----------------------------

danger_keywords = [
    "vaccine", "covid", "cancer cure", "miracle cure",
    "infertility", "secret cure", "pharmaceutical conspiracy",
    "100% cure", "guaranteed cure", "government hiding"
]

credible_keywords = [
    "who report", "official report", "research study",
    "university study", "scientists say"
]

# ----------------------------
# PREDICTION
# ----------------------------

if analyze:

    if text_input == "" and uploaded_image is None:

        st.warning("Please provide text or image")

    else:

        # TEXT
        if text_input != "":

            tokens = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        else:

            input_ids = torch.zeros((1,10),dtype=torch.long)
            attention_mask = torch.zeros((1,10),dtype=torch.long)

        # IMAGE
        if uploaded_image:

            image = Image.open(uploaded_image).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

        else:

            image_tensor = torch.zeros((1,3,224,224))

        # MODEL
        with torch.no_grad():

            outputs = model(input_ids, attention_mask, image_tensor)
            probs = torch.softmax(outputs, dim=1)

        probs = probs.squeeze().tolist()

        # ----------------------------
        # RULE ADJUSTMENT
        # ----------------------------

        text_lower = text_input.lower()

        if any(word in text_lower for word in danger_keywords):

            probs = [0.10, 0.20, 0.70]

        elif any(word in text_lower for word in credible_keywords):

            probs = [0.70, 0.20, 0.10]

        pred = probs.index(max(probs))
        confidence = max(probs)
        confidence_percent = confidence * 100

        labels = [
            "True Information",
            "Mixed / Uncertain",
            "False / Misinformation"
        ]

# ----------------------------
# GAUGE
# ----------------------------

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_percent,
            number={'suffix':"%"},
            title={'text':"Model Confidence"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"#ff4040"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth':0
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color':"white"}
        )

        st.plotly_chart(fig,use_container_width=True)

# ----------------------------
# PREDICTION
# ----------------------------

        st.markdown(f"""
### Prediction  
**{labels[pred]}**
""")

# ----------------------------
# RISK LOGIC
# ----------------------------

        if confidence_percent < 40:

            risk="Low Risk"
            risk_color="#2ecc71"

            explanation="The system detected patterns associated with verified factual information."

            recommendation="Information appears credible, but still verify from trusted sources."

        elif confidence_percent < 70:

            risk="Moderate Risk"
            risk_color="#f1c40f"

            explanation="The claim contains ambiguous patterns that may indicate misleading information."

            recommendation="Verify this claim using reliable fact-check websites."

        else:

            risk="High Risk"
            risk_color="#e74c3c"

            explanation="The system detected patterns commonly associated with misinformation."

            recommendation="Avoid trusting or sharing this claim without verification."

# ----------------------------
# CARDS
# ----------------------------

        st.markdown("## Analysis Report")

        col1,col2,col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="card">
            <h4 style="color:white">Risk Level</h4>
            <h2 style="color:{risk_color}">{risk}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="card">
            <h4 style="color:white">Explanation</h4>
            <p style="color:#ddd">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class="card">
            <h4 style="color:white">Recommendation</h4>
            <p style="color:#ddd">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# IMAGE DISPLAY
# ----------------------------

        if uploaded_image:

            st.subheader("Uploaded Image")
            st.image(image,use_column_width=True)
import streamlit as st
from langchain_helper import generate_startup_info

st.header("🚀 Startup Name Generator")
st.markdown("**Created by Sivaprasath Munusamy**")  # Bold name

categories = ['AI', 'Health Tech', 'FinTech', 'EdTech', 'E-commerce', 'Gaming']
selected_category = st.selectbox("Choose your startup category:", categories)

if st.button("Generate"):
    with st.spinner("Generating startup name and domain..."):
        result = generate_startup_info(selected_category)
    st.subheader("Generated Startup Name & Domain")
    st.write(f"**Startup Name:** {result['startup_name']}")
    st.write(f"**Domain:** {result['domain_name']}")

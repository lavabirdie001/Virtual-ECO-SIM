import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face chatbot model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 18px;
        font-family: Arial, sans-serif;
        color: #333;
    }
    .highlight-box {
        background-color: #f9f9f9;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Page title and introduction
st.title("🌍 Ecosystem Simulation and Insights 🐾")
st.write("""
Welcome to the **Ecosystem Simulation Portal**! 🌱💧
Explore how various parameters influence ecosystems through interactive simulations and AI insights. 🌿🐾🦅
""")

# AI Chatbot Interface
st.sidebar.title("🤠 AI Ecosystem Chatbot")
user_query = st.sidebar.text_input("Ask the Chatbot 🌟", "Ask Me Anything ?")
if st.sidebar.button("Get Response"):
    try:
        response = chatbot(user_query, max_length=100, num_return_sequences=1)
        chatbot_response = response[0]['generated_text']
        st.sidebar.write("🤖 Chatbot Response:")
        st.sidebar.success(chatbot_response)
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Sidebar Parameters for Simulation
st.sidebar.title("🔧 Adjust Parameters")
plant_growth_rate = st.sidebar.slider("🌱 Plant Growth Rate", 0.01, 0.5, 0.2)
herbivore_birth_rate = st.sidebar.slider("🐇 Herbivore Birth Rate", 0.01, 0.3, 0.1)
predator_birth_rate = st.sidebar.slider("🦁 Predator Birth Rate", 0.01, 0.2, 0.05)
initial_plants = st.sidebar.slider("🌼 Initial Plant Population", 50, 500, 100)
initial_herbivores = st.sidebar.slider("🐐 Initial Herbivore Population", 10, 100, 30)
initial_predators = st.sidebar.slider("🦅 Initial Predator Population", 5, 50, 10)
time_steps = st.sidebar.slider("⏱ Simulation Duration (Steps)", 10, 200, 50)

# New Abiotic and Biotic Factors
water_availability = st.sidebar.slider("💧 Water Availability", 0.0, 1.0, 0.5)
temperature_variation = st.sidebar.slider("🌡 Temperature Variation (°C)", -10, 40, 25)
soil_quality = st.sidebar.slider("🌿 Soil Quality Index", 0.1, 1.0, 0.7)
human_impact = st.sidebar.slider("🚚 Human Impact Factor", 0.0, 1.0, 0.2)

# Function to run the simulation
def run_simulation(initial_plants, initial_herbivores, initial_predators, time_steps):
    plants, herbivores, predators = initial_plants, initial_herbivores, initial_predators
    plant_pop, herbivore_pop, predator_pop = [], [], []

    for t in range(time_steps):
        plants = max(plants + plants * plant_growth_rate * (1 + water_availability - 0.1 * human_impact) - herbivores * 0.01, 0)
        herbivores = max(herbivores + herbivores * herbivore_birth_rate * (1 + soil_quality - 0.05 * temperature_variation) - predators * 0.01, 0)
        predators = max(predators + predators * predator_birth_rate * (herbivores / (herbivores + 1)), 0)
        plant_pop.append(plants)
        herbivore_pop.append(herbivores)
        predator_pop.append(predators)

    return plant_pop, herbivore_pop, predator_pop

# Run the simulation
plant_pop, herbivore_pop, predator_pop = run_simulation(initial_plants, initial_herbivores, initial_predators, time_steps)

# Create a DataFrame for population data
population_data = pd.DataFrame({
    "Time": range(time_steps),
    "Plants": plant_pop,
    "Herbivores": herbivore_pop,
    "Predators": predator_pop
})

# Graphs
st.subheader("📊 Population Dynamics Over Time")
fig1 = px.line(population_data, x="Time", y=["Plants", "Herbivores", "Predators"],
               labels={"value": "Population Size", "variable": "Species"},
               title="📈 Population Dynamics")
st.plotly_chart(fig1)

# Advanced Insights
st.subheader("🔍 Advanced Ecosystem Relationships")
abiotic_vs_biotic = pd.DataFrame({
    "Abiotic Factor": ["Water Availability", "Temperature Variation", "Soil Quality", "Human Impact"],
    "Impact on Plants": [water_availability, 1 - temperature_variation / 40, soil_quality, 1 - human_impact],
    "Impact on Herbivores": [water_availability * 0.8, 1 - temperature_variation / 50, soil_quality * 0.7, 1 - human_impact * 0.6],
    "Impact on Predators": [water_availability * 0.5, 1 - temperature_variation / 60, soil_quality * 0.3, 1 - human_impact * 0.2]
})

fig2 = px.bar(abiotic_vs_biotic, x="Abiotic Factor", y=["Impact on Plants", "Impact on Herbivores", "Impact on Predators"],
              title="🌧️ Abiotic Factors and Their Impact on Biotic Components",
              barmode="group")
st.plotly_chart(fig2)

# Correlation Analysis
st.subheader("📈 Correlation Analysis")
correlation_data = pd.DataFrame({
    "Species": ["Plants", "Herbivores", "Predators"],
    "Population": [np.mean(plant_pop), np.mean(herbivore_pop), np.mean(predator_pop)]
})

fig3 = px.scatter(correlation_data, x="Species", y="Population", title="📊 Average Population of Species",
                  labels={"Population": "Average Population", "Species": "Species"})
st.plotly_chart(fig3)

# Cumulative Population Trends
population_data["Total Population"] = population_data["Plants"] + population_data["Herbivores"] + population_data["Predators"]
fig4 = px.area(population_data, x="Time", y="Total Population",
               title="📊 Cumulative Population Trends Over Time",
               labels={"x": "Time", "y": "Total Population"})
st.plotly_chart(fig4)

# AI and ML Insights Section
st.subheader("🧠 AI-Powered Ecosystem Insights")

# Generate summary insights
st.write("#### Key Highlights")
st.write(f"- **Average Plant Growth Rate**: {np.mean(plant_pop):.2f}")
st.write(f"- **Average Herbivore Population**: {np.mean(herbivore_pop):.2f}")
st.write(f"- **Average Predator Population**: {np.mean(predator_pop):.2f}")
st.write(f"- **Total Duration Simulated**: {time_steps} steps")

# User Feedback Section
st.subheader("💬 User Feedback")
feedback = st.text_area("Please provide your feedback on the simulation and chatbot responses:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")

# Footer
st.markdown("""
---

Made with Love ❤️ by **Prateek** and **Aditya**
""")
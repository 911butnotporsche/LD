# **Dataset Description**

This document provides a detailed description of the datasets shared on Zenodo, focusing on air quality data collected through **LoRa-based nodes** and a **Palas Fidas Frog sensor**. The dataset is structured to support machine learning-based calibration of low-cost sensors against research-grade instruments.

---

## **1. Dataset Structure and Format**

The dataset is provided in the **CSV (Comma-Separated Values)** format and contains multiple columns representing sensor measurements, environmental conditions, and particulate matter concentrations.

### **Columns Overview:**
- LoRa Particle Data
- LoRa Environmental Parameters
- Palas Fidas Frog Measurements
- Palas Environmental Parameters

The dataset spans time-series measurements collected over a defined period and is structured with timestamps, allowing analysis of temporal trends and correlations.

---

## **2. Variables and Definitions**

### **LoRa Particle Data**
- **`P1_lpo_loRa`**: Represents the total time (in milliseconds) during which the signal for particles larger than 1 μm remains low in a 15-second sampling period.  
  - Also referred to as the **> 1 μm LPO**.
  
- **`P1_ratio_loRa`**: Represents the proportion of time the sensor signal for particles larger than 1 μm is low during the sampling period.  
  - Also referred to as the **> 1 μm ratio**.

- **`P1_conc_loRa`**: Measures the PM concentration of particles larger than 1 μm (µg/m³).  
  - Also referred to as the **> 1 μm concentration**.

- **`P2_lpo_loRa`**: Represents the total time (in milliseconds) during which the signal for particles larger than 2.5 μm remains low in a 15-second sampling period.  
  - Also referred to as the **> 2.5 μm LPO**.

- **`P2_ratio_loRa`**: Represents the proportion of time the sensor signal for particles larger than 2.5 μm is low during the sampling period.  
  - Also referred to as the **> 2.5 μm ratio**.

- **`P2_conc_loRa`**: Measures the PM concentration of particles larger than 2.5 μm (µg/m³).  
  - Also referred to as the **> 2.5 μm concentration**.

---

### **LoRa Environmental Parameters**
- **`Temperature_loRa`**: Air temperature measured by the LoRa Node’s BME280 sensor (°C).
- **`Pressure_loRa`**: Atmospheric pressure measured by the LoRa Node’s BME280 sensor (hPa).
- **`Humidity_loRa`**: Relative humidity measured by the LoRa Node’s BME280 sensor (%).

---

### **Palas Fidas Frog Measurements**
- **`pm1Palas`**: PM1 concentration measured by the Palas sensor (µg/m³).
- **`pm2_5Palas`**: PM2.5 concentration measured by the Palas sensor (µg/m³).
- **`pm4Palas`**: PM4 concentration measured by the Palas sensor (µg/m³).
- **`pm10Palas`**: PM10 concentration measured by the Palas sensor (µg/m³).
- **`pmTotalPalas`**: Total particulate matter concentration measured by the Palas sensor (µg/m³).
- **`dCnPalas`**: Particle count density measured by the Palas sensor (particles/m³).

---

### **Palas Environmental Parameters**
- **`temperaturePalas`**: Air temperature measured by the Palas sensor (°C).
- **`humidityPalas`**: Relative humidity measured by the Palas sensor (%).
- **`pressurehPalas`**: Atmospheric pressure measured by the Palas sensor (hPa).

---

## **3. Purpose of the Dataset**

This dataset is designed for **calibrating low-cost LoRa-based air quality monitors** against the research-grade Palas Fidas Frog sensor. It enables:

- **Model training and evaluation**: Build regression models to predict PM concentrations from LoRa sensor readings.
- **Hyperparameter tuning**: Optimize the performance of machine learning models.
- **Performance validation**: Compare predictions from LoRa nodes to research-grade measurements.

---

## **4. Notes**
- **PM** stands for **Particulate Matter**.
- Particle concentration units:  
  - **µg/m³**: Micrograms per cubic meter  
  - **particles/m³**: Particle count per cubic meter

# Hybrid System for Predicting Mastitis and Foot and Mouth Disease in Dairy Cows


A web-based application for identifying and diagnosing common bovine diseases (Mastitis and Foot & Mouth Disease) using symptom analysis and image processing.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Impact and Benefits](#impact-and-benefits)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Bovine Disease Identification System is an innovative digital solution designed to assist farmers and veterinarians in early detection and management of two critical cattle diseases: Mastitis and Foot & Mouth Disease (FMD). This AI-powered web application combines symptom analysis with image processing to provide accurate, real-time disease diagnosis and treatment recommendations.

## Problem Statement

Cattle diseases cause significant economic losses worldwide:
- Mastitis costs the global dairy industry approximately $35 billion annually
- FMD outbreaks can lead to 20-50% milk yield reduction
- Small farmers often lack access to timely veterinary diagnostics
- Traditional diagnosis methods are time-consuming and require specialized expertise

## Features

### Disease Detection
- Mastitis diagnosis through udder analysis
- Foot and Mouth Disease (FMD) detection via:
  - Mouth symptoms analysis
  - Foot symptoms evaluation

### Multi-modal Input
- Text-based symptom reporting
- Image upload for visual analysis
  - Udder images for Mastitis
  - Mouth/Foot images for FMD

### Comprehensive Reporting
- Detailed diagnosis with confidence percentages
- Treatment suggestions based on severity levels:
  - High risk (Immediate action needed)
  - Medium risk (Monitoring recommended)
  - Low risk (Preventive measures)
- PDF report generation

### Additional Features
- Multi-language support (English, Hindi, Kannada)
- Interactive chatbot assistant
- Nearby veterinarian locator
- Responsive design for mobile devices

## Technologies Used

### Frontend
- HTML5, CSS3, JavaScript
- [i18next](https://www.i18next.com/) for internationalization
- [jsPDF](https://parall.ax/products/jspdf) for PDF generation
- Chart.js for data visualization

### Backend
- Python with Flask framework
- Machine learning models:
  - CNN for image classification
  - Random Forest for symptom analysis
- SQLite database


## System Architecture

```mermaid
graph TD
    A[User Interface] --> B{Input Type}
    B -->|Text| C[Symptom Analysis]
    B -->|Image| D[Image Processing]
    C --> E[Diagnosis Engine]
    D --> E
    E --> F[Result Generation]
    F --> G[Report/PDF]
    F --> H[Treatment Suggestions]
    G --> I[User]
    H --> I
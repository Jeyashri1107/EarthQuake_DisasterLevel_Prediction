# PRCP-1015: Earthquake Damage Prediction
Predicting the ordinal variable damage_grade, which represents a level of damage to the building that was hit by the earthquake. There are 3 grades of the damage: 1 represents low damage 2 represents a medium amount of damage 3 represents almost complete destruction

📌 Problem Statement

Task 1

- Prepare a complete data analysis report on the given dataset

Task 2

- Build a predictive model to estimate the damage_grade of buildings
- The target variable damage_grade represents the level of damage caused by the earthquake:
  - 1 → Low damage
  - 2 → Medium damage
  - 3 → Severe damage

Task 3

- Provide actionable suggestions to seismologists to reduce building damage during earthquakes

---

📂 Dataset Information

This project is based on a DrivenData competition dataset focused on predicting earthquake damage to buildings.

- Each row represents a building affected by the Gorkha earthquake
- The dataset contains 39 columns:
  - "building_id" → Unique identifier
  - 38 features describing structural, geographical, and usage details

🔗 Dataset Link
👉 https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1015-EquakeDamagePred.zip

---

📊 Feature Overview

🌍 Geographic Features

- "geo_level_1_id" → Region level 1 (0–30)
- "geo_level_2_id" → Region level 2 (0–1427)
- "geo_level_3_id" → Region level 3 (0–12567)

---

🏢 Building Structure

- "count_floors_pre_eq" → Number of floors before earthquake
- "age" → Age of the building
- "area_percentage" → Normalized building area
- "height_percentage" → Normalized building height

---

🧱 Construction Details

- "land_surface_condition" → Land condition (n, o, t)
- "foundation_type" → Foundation type (h, i, r, u, w)
- "roof_type" → Roof type (n, q, x)
- "ground_floor_type" → Ground floor type (f, m, v, x, z)
- "other_floor_type" → Upper floor construction (j, q, s, x)
- "position" → Building position (j, o, s, t)
- "plan_configuration" → Building layout (a, c, d, f, m, n, o, q, s, u)

---

🏗️ Superstructure Materials

- "has_superstructure_adobe_mud"
- "has_superstructure_mud_mortar_stone"
- "has_superstructure_stone_flag"
- "has_superstructure_cement_mortar_stone"
- "has_superstructure_mud_mortar_brick"
- "has_superstructure_cement_mortar_brick"
- "has_superstructure_timber"
- "has_superstructure_bamboo"
- "has_superstructure_rc_non_engineered"
- "has_superstructure_rc_engineered"
- "has_superstructure_other"

---

⚖️ Ownership & Usage

- "legal_ownership_status" → Ownership type (a, r, v, w)
- "count_families" → Number of families in the building

---

🏠 Secondary Usage

- "has_secondary_use"
- "has_secondary_use_agriculture"
- "has_secondary_use_hotel"
- "has_secondary_use_rental"
- "has_secondary_use_institution"
- "has_secondary_use_school"
- "has_secondary_use_industry"
- "has_secondary_use_health_post"
- "has_secondary_use_gov_office"
- "has_secondary_use_use_police"
- "has_secondary_use_other"

---

🎯 Objective

The main objective of this project is to:

- Analyze building characteristics and earthquake impact
- Build a robust predictive model for damage classification
- Provide insights to reduce structural damage risks in future earthquakes

---

🏷️ Domain

Earthquake Engineering / Disaster Management / Machine Learning

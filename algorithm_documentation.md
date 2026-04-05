**Algorithm 5.1 AI-Driven Multimodal Complaint Analysis and Routing Framework**

**Input:** Raw user complaint text $T_i$, physical evidence/documents $E_i$, and contextual metadata $M_i$.
**Output:** Predicted issue department $\hat{D}_i$, priority classification $P_i$, and unique tracking ID $ID_i$.

**Load Dataset**
1. Load dataset $D = \{x_i\}$ where each complaint instance is composed of $(T_i, E_i, M_i)$.
2. Assign target issue types $y_i$ (e.g., Accounts, Technical, Operations) for model training.

**Preprocess Data**
3. Remove noise, punctuation, and irrelevant symbols from text $T_i$.
4. Perform tokenization and English stopword removal to normalize text.
5. Ensure contextual field extraction via automated prompts (e.g., missing locations or vehicle numbers).

**Feature Extraction**
6. Convert processed text into (1,2)-gram TF-IDF vectors (max 7000 features):
   $F_{i} = \text{TF-IDF}(T_i)$
7. Construct feature matrix $F = \{F_i\}$ for all training document samples.

**Model Training**
8. Train Logistic Regression Classifier with balanced class weights using feature matrix $F$.
9. Learn generalized decision boundaries for complaint categories: $\hat{D}_i = f(F_i)$.

**Multi-Faceted Priority Scoring**
10. Extract keyword-based severity indicators $S_{text}$ from raw text to identify urgent terms.
11. Compute context-based multiplier $S_{context}$ derived from location data or urgency flags.
12. Evaluate bounds to assign priority level:
    $P_i \leftarrow \text{EvaluateContext}(S_{text}, S_{context}) \in \{\text{LOW, MEDIUM, HIGH}\}$

**Complaint Routing & Tracking**
13. Generate secure cryptographic tracking identifier $ID_i$ to ensure user transparency.
14. Persist relevant evidence paths securely and link $E_i$ to the primary $ID_i$.
15. Initialize dynamic state tracking lifecycle: $\text{Status}_{ID_i} \leftarrow \text{"Submitted"}$.

**Analytics & Alert Generation**
16. If $P_i == \text{HIGH}$, prioritize routing and trigger critical indicator on the department dashboard.
17. Aggregate active incident records to detect system-wide trends or recurring issue hotspots.

**Prediction**
18. Predict optimal resolution department:
    $\hat{D}_i \in \{\text{Target Department Set}\}$

**Evaluation**
19. Evaluate classification performance mapping via Accuracy, Precision, Recall, and F1-score logic.
20. **return** $\hat{D}_i, P_i$, and $ID_i$

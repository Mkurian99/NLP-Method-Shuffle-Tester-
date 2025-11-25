# NLP-Method-Shuffle-Tester-
This is a validation framework for testing the order sensitivity of mainstream NLP methods. It runs NLP method analysis (ID-IDF, Sentiment Analysis etc.) on original, word shuffled, and sentence shuffled versions to see how well the method indicates a loss or alteration of order-sensitive semantic structure like narrative form. The code is designed for direct drag and drop into google colab. The user then uploads an original copy of a text, a word shuffled version with random word order, and a sentence shuffle or line shuffle where word order within lines/sentences is maintained but the order of lines/sentences is itself random. If a method outputs similar results for all three, then that method is not sensitive to the narrative content or the order-level semantic processes. 


A comprehensive validation framework for testing whether NLP methods can detect semantic structure in text. Originally developed to validate Symbolic Entropy (SE), a novel framework that extends Shannon's 1948 information theory to measure meaning density.

**Initial Test Case**
The suite was first validated on KJV Genesis 1-3, comparing:
Original text
Word-shuffled version (complete structural destruction)
Sentence-shuffled version (local coherence preserved, discourse structure destroyed)

**What This Tests is...**
The 3-way shuffle comparison distinguishes between methods that detect:
Local coherence only — sensitive to word-shuffle but not sentence-shuffle
Discourse-level structure — sensitive to both shuffle types (what SE's Σ component measures)

A method that passes the word-shuffle test but fails the sentence-shuffle test only detects surface-level patterns. Methods sensitive to sentence-shuffle detect higher-order narrative architecture.

**The 8 Methods**
GPT-2 Perplexity Language model surprisal (how "expected" is this text?)
Sentiment Analysis Emotional valence distribution
TF-IDF Coherence Lexical similarity between consecutive sentences
Named Entity Recognition Entity density patterns
LDA Topic Modeling Topic assignment confidence
BERTScore Semantic similarity between consecutive windows
BERTopic Neural topic assignment probabilities
SE (Σ_total) Archetypal motif concentration via KL divergence from global baseline average

**Methodological Corrections**
This suite implements several corrections to ensure valid cross-condition comparison:
TF-IDF: Vectorizer fit on original only, then transform applied to all conditions
LDA: Dictionary and model trained on original only, inference on all conditions
BERTopic: fit_transform() on original, transform() on shuffled conditions
BERTScore: Uses fixed-size token windows instead of period-based sentence splitting (periods scatter randomly in word-shuffled text, creating invalid comparisons)
SE (Σ_total): Baseline calculated from original text, same window parameters for all conditions
Perplexity, Sentiment, NER: Pre-trained models (no fitting required)

**Interpretation**
Results are reported as Cohen's d effect sizes:
d ≥ 3.0: ✅✅ STRONG PASS — Unambiguous structural sensitivity
d ≥ 2.0: ✅ PASS — Clear structural sensitivity
1.0 ≤ d < 2.0: ~ BORDERLINE — Detects something, but not robustly
d < 1.0: ❌ FAIL — Cannot reliably distinguish structured from shuffled text

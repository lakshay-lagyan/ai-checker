
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            feature_dict = self._extract_features(text)
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    def _extract_features(self, text):
        features = {}
        
        # Basic statistics
        words = text.split()
        sentences = self._split_sentences(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Lexical diversity
        unique_words = len(set(words))
        features['lexical_diversity'] = unique_words / len(words) if words else 0
        
        # Punctuation analysis
        features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        features['exclamation_ratio'] = text.count('!') / len(text) if text else 0
        features['question_ratio'] = text.count('?') / len(text) if text else 0
        
        # AI pattern indicators
        ai_phrases = [
            'it is important to note', 'furthermore', 'moreover', 'consequently',
            'therefore', 'in conclusion', 'additionally', 'however', 'nevertheless'
        ]
        features['ai_phrase_count'] = sum(text.lower().count(phrase) for phrase in ai_phrases)
        features['ai_phrase_density'] = features['ai_phrase_count'] / len(words) if words else 0
        
        # Human pattern indicators
        human_phrases = [
            'i think', 'i believe', 'in my opinion', 'you know', 'like',
            'basically', 'actually', 'honestly', 'i mean'
        ]
        features['human_phrase_count'] = sum(text.lower().count(phrase) for phrase in human_phrases)
        features['human_phrase_density'] = features['human_phrase_count'] / len(words) if words else 0
        
        # Sentence length variance (AI tends to be more uniform)
        sentence_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_variance'] = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        features['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Coherence analysis
        features['coherence_score'] = self._calculate_coherence(sentences)
        
        # Temporal patterns
        features['pacing_consistency'] = self._calculate_pacing_consistency(sentence_lengths)
        
        # Cognitive complexity
        features['complex_word_ratio'] = sum(1 for w in words if len(w) > 7) / len(words) if words else 0
        features['simple_word_ratio'] = sum(1 for w in words if len(w) <= 4) / len(words) if words else 0
        
        # Transition words (AI uses more)
        transition_words = ['furthermore', 'moreover', 'however', 'therefore', 'consequently', 'additionally']
        features['transition_word_count'] = sum(text.lower().count(word) for word in transition_words)
        features['transition_density'] = features['transition_word_count'] / len(words) if words else 0
        
        # Repetition patterns
        features['word_repetition'] = self._calculate_repetition(words)
        
        # Formality score
        features['formality_score'] = self._calculate_formality(text)
        
        # Spoofing indicators
        features['mixed_register_score'] = self._detect_mixed_register(text)
        features['artificial_error_score'] = self._detect_artificial_errors(text)
        
        return features
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_coherence(self, sentences):
        """Calculate coherence between consecutive sentences"""
        if len(sentences) < 2:
            return 0.5
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
            coherence_scores.append(overlap)
        
        return np.mean(coherence_scores)
    
    def _calculate_pacing_consistency(self, sentence_lengths):
        """Calculate pacing consistency"""
        if len(sentence_lengths) < 2:
            return 1.0
        
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        cv = std_length / mean_length if mean_length > 0 else 0
        
        return 1 - min(cv, 1.0)
    
    def _calculate_repetition(self, words):
        """Calculate word repetition score"""
        if not words:
            return 0
        
        word_counts = Counter(words)
        repeated = sum(1 for count in word_counts.values() if count > 1)
        return repeated / len(word_counts)
    
    def _calculate_formality(self, text):
        """Calculate text formality score"""
        formal_indicators = ['utilize', 'commence', 'endeavor', 'facilitate', 'implement']
        informal_indicators = ['gonna', 'wanna', 'yeah', 'ok', 'cool']
        
        formal_count = sum(text.lower().count(word) for word in formal_indicators)
        informal_count = sum(text.lower().count(word) for word in informal_indicators)
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5
        
        return formal_count / total
    
    def _detect_mixed_register(self, text):
        """Detect mixed formal/informal register (humanization attempt)"""
        formal_count = sum(text.lower().count(word) for word in [
            'furthermore', 'moreover', 'consequently', 'therefore'
        ])
        casual_count = sum(text.lower().count(word) for word in [
            'super', 'kinda', 'sorta', 'basically', 'literally'
        ])
        
        if formal_count > 1 and casual_count > 1:
            return 1.0
        return 0.0
    
    def _detect_artificial_errors(self, text):
        """Detect artificially inserted errors"""
        words = text.split()
        if not words:
            return 0
        
        # Check for suspicious patterns
        potential_typos = sum(1 for w in words if len(w) > 4 and not w.isalpha() and w.isalnum())
        typo_rate = potential_typos / len(words)
        
        # Artificial typos tend to be in a specific range
        if 0.01 < typo_rate < 0.05:
            return 0.8
        return 0.2


class AITextDetectorModel:
    """
    Production-ready AI text detector using scikit-learn
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_extractor = None
        self.scaler = None
        self.training_history = {}
        
    def build_model(self):
        """Build ensemble model with multiple classifiers"""
        print("🏗️  Building ensemble model...")
        
        # Feature extraction pipeline
        self.feature_extractor = TextFeatureExtractor()
        
        # TF-IDF vectorizer for text content
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Scaler for custom features
        self.scaler = StandardScaler()
        
        # Ensemble of classifiers
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        
        lr_clf = LogisticRegression(
            C=10,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        svm_clf = SVC(
            C=10,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('gb', gb_clf),
                ('lr', lr_clf),
                ('svm', svm_clf)
            ],
            voting='soft',
            weights=[2, 2, 1, 1],
            n_jobs=-1
        )
        
        print("✅ Model architecture created")
        
    def prepare_features(self, texts, fit=False):
        """Prepare features from texts"""
        # Extract TF-IDF features
        if fit:
            tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.vectorizer.transform(texts).toarray()
        
        # Extract custom features
        custom_features = self.feature_extractor.transform(texts)
        
        # Scale custom features
        if fit:
            custom_features = self.scaler.fit_transform(custom_features)
        else:
            custom_features = self.scaler.transform(custom_features)
        
        # Combine all features
        combined_features = np.hstack([tfidf_features, custom_features])
        
        return combined_features
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        print("\n🎯 Training AI detector model...")
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        
        # Prepare features
        print("🔧 Extracting features...")
        X_train_features = self.prepare_features(X_train, fit=True)
        X_val_features = self.prepare_features(X_val, fit=False)
        
        print(f"✅ Feature shape: {X_train_features.shape}")
        
        # Train model
        print("🚀 Training ensemble model...")
        self.model.fit(X_train_features, y_train)
        
        # Validate
        print("\n📈 Evaluating model...")
        train_pred = self.model.predict(X_train_features)
        val_pred = self.model.predict(X_val_features)
        
        train_pred_proba = self.model.predict_proba(X_train_features)
        val_pred_proba = self.model.predict_proba(X_val_features)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        train_auc = roc_auc_score(y_train, train_pred_proba[:, 1])
        val_auc = roc_auc_score(y_val, val_pred_proba[:, 1])
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_pred, average='binary'
        )
        
        conf_matrix = confusion_matrix(y_val, val_pred)
        
        # Store training history
        self.training_history = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_auc': float(train_auc),
            'val_auc': float(val_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"📊 TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Val Accuracy:   {val_acc*100:.2f}%")
        print(f"Train AUC:      {train_auc:.4f}")
        print(f"Val AUC:        {val_auc:.4f}")
        print(f"Precision:      {precision*100:.2f}%")
        print(f"Recall:         {recall*100:.2f}%")
        print(f"F1 Score:       {f1*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"{'='*60}")
        
        return self.training_history
    
    def predict(self, texts):
        """Predict whether texts are AI-generated"""
        if isinstance(texts, str):
            texts = [texts]
        
        features = self.prepare_features(texts, fit=False)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return predictions, probabilities
    
    def save(self, path='models/'):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving model to {path}...")
        
        # Save model components
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(path / 'feature_extractor.pkl', 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        with open(path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("✅ Model saved successfully")
    
    def load(self, path='models/'):
        """Load model from disk"""
        path = Path(path)
        
        print(f"📥 Loading model from {path}...")
        
        with open(path / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(path / 'vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(path / 'feature_extractor.pkl', 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        with open(path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(path / 'training_history.json', 'r') as f:
            self.training_history = json.load(f)
        
        print("✅ Model loaded successfully")
        print(f"   Validation Accuracy: {self.training_history['val_accuracy']*100:.2f}%")


def create_training_data(num_samples=2000):
    """
    Create synthetic training data for demonstration
    In production, replace with real datasets
    """
    print("\n🔬 Creating training dataset...")
    
    # AI-generated patterns
    ai_templates = [
        "It is important to note that {topic}. Furthermore, {detail}. Consequently, {conclusion}. Moreover, {addition}. Therefore, the implications are {impact}.",
        "The analysis of {topic} reveals significant insights. Additionally, {detail}. Hence, {conclusion}. Furthermore, {addition}. This demonstrates {impact}.",
        "{topic} represents a critical consideration. Moreover, {detail}. Consequently, one must conclude that {conclusion}. Additionally, {addition}, which suggests {impact}.",
        "In examining {topic}, several factors emerge. Furthermore, {detail}. Therefore, {conclusion}. Moreover, {addition}. The resulting {impact} cannot be overlooked.",
        "It is evident that {topic} plays a crucial role. Additionally, {detail}. Thus, {conclusion}. Furthermore, {addition}, leading to {impact}."
    ]
    
    # Human-written patterns
    human_templates = [
        "I've been thinking about {topic}. {detail}, you know? I guess {conclusion}. {addition}. It's {impact}, honestly.",
        "So {topic} is interesting. Like, {detail}. I think {conclusion}. Also, {addition}. Pretty {impact} if you ask me.",
        "{topic}? Yeah, I have some thoughts. {detail}. My take is {conclusion}. Plus {addition}. Seems {impact} to me.",
        "Here's the thing about {topic}. {detail}. Honestly, {conclusion}. And {addition}. That's {impact}, right?",
        "Well, {topic}. {detail}. I mean, {conclusion}. {addition}. It's kind of {impact}."
    ]
    
    topics = [
        "artificial intelligence", "climate change", "education systems",
        "social media impact", "remote work", "sustainable technology",
        "healthcare innovation", "economic policy", "digital privacy"
    ]
    
    details = [
        "research shows significant progress", "experts have varying opinions",
        "data indicates clear trends", "studies reveal complex patterns",
        "evidence suggests multiple factors"
    ]
    
    conclusions = [
        "careful consideration is necessary", "further research is warranted",
        "immediate action should be taken", "balanced approaches work best",
        "we need better solutions"
    ]
    
    additions = [
        "stakeholders must collaborate", "policy changes are essential",
        "public awareness is growing", "technology plays a key role",
        "education remains fundamental"
    ]
    
    impacts = [
        "far-reaching", "significant", "transformative", "concerning", "promising"
    ]
    
    data = []
    
    # Generate AI samples
    for _ in range(num_samples // 2):
        template = np.random.choice(ai_templates)
        text = template.format(
            topic=np.random.choice(topics),
            detail=np.random.choice(details),
            conclusion=np.random.choice(conclusions),
            addition=np.random.choice(additions),
            impact=np.random.choice(impacts)
        )
        data.append({'text': text, 'label': 1, 'source': 'synthetic_ai'})
    
    # Generate human samples
    for _ in range(num_samples // 2):
        template = np.random.choice(human_templates)
        text = template.format(
            topic=np.random.choice(topics),
            detail=np.random.choice(details),
            conclusion=np.random.choice(conclusions),
            addition=np.random.choice(additions),
            impact=np.random.choice(impacts)
        )
        data.append({'text': text, 'label': 0, 'source': 'synthetic_human'})
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} training samples")
    print(f"   AI-generated: {len(df[df['label']==1])}")
    print(f"   Human-written: {len(df[df['label']==0])}")
    
    return df


def main():
    """Main training pipeline"""
    print("="*80)
    print("🤖 AI TEXT DETECTOR - SCIKIT-LEARN MODEL TRAINING")
    print("="*80)
    
    # Create synthetic data (replace with real data in production)
    print("\n⚠️  DEMO MODE: Using synthetic data")
    df = create_training_data(num_samples=2000)
    
    # Split data
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    detector = AITextDetectorModel()
    detector.build_model()
    history = detector.train(X_train, y_train, X_val, y_val)
    
    # Save model
    detector.save('models/')
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_texts = [
        "It is important to note that artificial intelligence has revolutionized various industries. Furthermore, machine learning algorithms demonstrate remarkable capabilities. Consequently, the future of technology appears promising.",
        "I think AI is pretty cool, you know? Like, it's changing everything. But I'm not sure if that's always good. What do you think?"
    ]
    
    predictions, probabilities = detector.predict(test_texts)
    
    for i, text in enumerate(test_texts):
        verdict = "AI-Generated" if predictions[i] == 1 else "Human-Written"
        confidence = probabilities[i][predictions[i]] * 100
        print(f"\nText {i+1}: {text[:80]}...")
        print(f"Verdict: {verdict}")
        print(f"Confidence: {confidence:.2f}%")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📦 Model files saved to 'models/' directory")
    print("🚀 Ready for deployment!")


if __name__ == "__main__":
    main()
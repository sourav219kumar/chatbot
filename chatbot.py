import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import requests  # For API integration

class AdvancedFAQChatbot:
    def __init__(self, topic: str = "general", config_path: str = "chatbot_config.json"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load SpaCy's transformer model for best accuracy
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            raise ImportError(
                "Required model 'en_core_web_trf' not found. Please install with:\n"
                "python -m spacy download en_core_web_trf"
            )
        
        self.topic = topic
        self.conversation_history: List[Dict] = []
        self.user_profile: Dict = {}
        
        # Enhanced FAQ database with support for multimedia and actions
        self.faqs = self._initialize_faq_database()
        
        # Initialize vectorizer and vectors
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self._initialize_question_vectors()
        
        # Sentiment analyzer (simple version - could use VADER or similar)
        self.positive_words = {"good", "great", "excellent", "happy", "satisfied"}
        self.negative_words = {"bad", "poor", "terrible", "unhappy", "angry"}
        
        # External service integration flags
        self.use_knowledge_graph = False
        self.use_translation = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "similarity_threshold": 0.4,
            "max_history": 5,
            "enable_sentiment": True,
            "enable_multilingual": False
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                try:
                    return {**default_config, **json.load(f)}
                except json.JSONDecodeError:
                    return default_config
        return default_config
    
    def _initialize_faq_database(self) -> Dict:
        """Initialize the FAQ database with advanced features"""
        return {
            "general": {
                "What are your opening hours?": {
                    "answer": "We're open Monday to Friday from 9 AM to 5 PM and Saturdays from 10 AM to 2 PM.",
                    "keywords": ["open", "hours", "time", "available"],
                    "actions": ["set_reminder"],
                    "metadata": {"category": "operations"}
                },
                "How do I contact customer support?": {
                    "answer": "You can reach us at support@example.com or call +1 (555) 123-4567 during business hours.",
                    "keywords": ["contact", "support", "help", "phone", "email"],
                    "actions": ["initiate_call", "compose_email"],
                    "metadata": {"priority": "high"}
                }
            },
            "technical": {
                "How do I reset my password?": {
                    "answer": "Visit our password reset page and follow the instructions. Would you like me to send you the link?",
                    "keywords": ["password", "reset", "login", "account"],
                    "actions": ["send_link"],
                    "metadata": {"security": True}
                }
            }
        }
    
    def _initialize_question_vectors(self):
        """Initialize question vectors with support for dynamic updates"""
        self.all_questions = []
        for category in self.faqs.values():
            self.all_questions.extend(category.keys())
        
        self.processed_questions = [self._advanced_preprocess(q) for q in self.all_questions]
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)
    
    def _advanced_preprocess(self, text: str) -> str:
        """Advanced text preprocessing with entity recognition and phrase detection"""
        doc = self.nlp(text.lower())
        processed_tokens = []
        
        # Handle noun chunks and entities
        for chunk in doc.noun_chunks:
            processed_tokens.append(chunk.text)
            
        # Add important verbs and adjectives
        for token in doc:
            if (token.pos_ in ["VERB", "ADJ"] and not token.is_stop) or token.ent_type_:
                processed_tokens.append(token.lemma_)
        
        # Remove duplicates while preserving order
        seen = set()
        return ' '.join([x for x in processed_tokens if not (x in seen or seen.add(x))])
    
    def _analyze_sentiment(self, text: str) -> float:
        """Basic sentiment analysis (replace with proper model in production)"""
        doc = self.nlp(text)
        sentiment_score = 0
        
        for token in doc:
            if token.lemma_ in self.positive_words:
                sentiment_score += 1
            elif token.lemma_ in self.negative_words:
                sentiment_score -= 1
                
        return sentiment_score / len(doc) if doc else 0
    
    def _handle_actions(self, question: str, user_input: str) -> Optional[str]:
        """Handle any actions associated with a question"""
        category = self._find_question_category(question)
        if not category:
            return None
            
        actions = self.faqs[category][question].get("actions", [])
        
        for action in actions:
            if action == "send_link" and "send" in user_input.lower():
                return "I've sent the password reset link to your email. Please check your inbox."
            elif action == "initiate_call" and "call" in user_input.lower():
                return "Initiating call to our support team now..."
        
        return None
    
    def _find_question_category(self, question: str) -> Optional[str]:
        """Find which category a question belongs to"""
        for category, questions in self.faqs.items():
            if question in questions:
                return category
        return None
    
    def _generate_alternative_questions(self, user_input: str) -> List[str]:
        """Suggest similar questions when no good match is found"""
        processed_input = self._advanced_preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        
        similarities = cosine_similarity(input_vector, self.question_vectors)
        similar_indices = similarities.argsort()[0][-3:][::-1]  # Top 3 matches
        
        return [self.all_questions[i] for i in similar_indices 
                if similarities[0,i] > self.config["similarity_threshold"]/2]
    
    def _log_conversation(self, user_input: str, response: str):
        """Log conversation with timestamp and metadata"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "sentiment": self._analyze_sentiment(user_input) if self.config["enable_sentiment"] else None
        }
        self.conversation_history.append(entry)
        
        # Keep only recent history
        if len(self.conversation_history) > self.config["max_history"]:
            self.conversation_history.pop(0)
    
    def _get_contextual_response(self, user_input: str) -> Optional[str]:
        """Use conversation history to provide contextual responses"""
        if not self.conversation_history:
            return None
            
        last_entry = self.conversation_history[-1]
        
        # Handle follow-up questions
        if "this" in user_input.lower() or "that" in user_input.lower():
            return f"Regarding your previous question, {last_entry['response']}"
            
        return None
    
    def get_response(self, user_input: str) -> str:
        """Generate intelligent responses with advanced features"""
        # Check for empty input
        if not user_input.strip():
            return "I didn't receive your question. Could you please repeat or rephrase it?"
        
        # Check conversation history for context
        contextual_response = self._get_contextual_response(user_input)
        if contextual_response:
            self._log_conversation(user_input, contextual_response)
            return contextual_response
        
        # Find the most similar FAQ question
        processed_input = self._advanced_preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        
        similarities = cosine_similarity(input_vector, self.question_vectors)
        most_similar_idx = similarities.argmax()
        similarity_score = similarities[0, most_similar_idx]
        
        # Handle low similarity scores
        if similarity_score < self.config["similarity_threshold"]:
            alternatives = self._generate_alternative_questions(user_input)
            if alternatives:
                response = "I'm not sure I understand. Did you mean:\n"
                response += "\n".join(f"- {q}" for q in alternatives)
            else:
                response = random.choice([
                    "Could you please rephrase your question?",
                    "I don't have information on that topic. Could you ask something else?",
                    "I'm still learning about this. Could you try a different question?"
                ])
            
            self._log_conversation(user_input, response)
            return response
        
        selected_question = self.all_questions[most_similar_idx]
        
        # Handle any actions associated with the question
        action_response = self._handle_actions(selected_question, user_input)
        if action_response:
            self._log_conversation(user_input, action_response)
            return action_response
        
        # Get the standard answer
        category = self._find_question_category(selected_question)
        answer = self.faqs[category][selected_question]["answer"]
        
        # Add follow-up prompt
        answer += "\n\nIs there anything else I can help you with?"
        
        self._log_conversation(user_input, answer)
        return answer
    
    def run(self):
        """Run the chatbot in interactive mode with enhanced interface"""
        print(f"\n⭐ Advanced FAQ Chatbot v2.0 - Topic: {self.topic} ⭐")
        print("Type your questions or 'quit' to exit. I can help with:")
        
        # Display top categories
        for category in self.faqs.keys():
            print(f"- {category.capitalize()}")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nAdvanced FAQ Chatbot: Thank you for chatting! Have a great day.")
                    break
                
                response = self.get_response(user_input)
                print(f"\nAdvanced FAQ Chatbot: {response}")
                
            except KeyboardInterrupt:
                print("\n\nAdvanced FAQ Chatbot: Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\nAdvanced FAQ Chatbot: I encountered an error. Please try again.")
                # In production: log the error properly
                print(f"[System Error: {str(e)}]")

if __name__ == "__main__":
    # Initialize with custom configuration
    chatbot = AdvancedFAQChatbot(
        topic="Technical Support",
        config_path="chatbot_config.json"
    )
    
    # Example of adding a question dynamically
    chatbot.faqs["general"]["Where is your headquarters located?"] = {
        "answer": "Our global headquarters is at 123 Tech Park, San Francisco, CA.",
        "keywords": ["location", "address", "headquarters", "office"],
        "metadata": {"verified": True}
    }
    
    chatbot.run()